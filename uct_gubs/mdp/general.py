import logging
import math
import random
import time

import numpy as np
from pddlgym.inference import check_goal
from pddlgym.structs import Literal

from uct_gubs import context, tree, pddl, rendering
from uct_gubs.mdp.types import ExtendedState

SQRT_TWO = math.sqrt(2)


def risk_exp_fn(lamb):
    return lambda cumcost: np.exp(lamb * cumcost)


def run_round(ctx: context.ProblemContext, s: ExtendedState,
              actions: frozenset, n_steps: int):
    mdp_tree = tree.new_tree(s, 0, actions)
    pi = {}

    start = time.perf_counter()

    cur_tree = mdp_tree
    depth = 0
    found_goal = False
    cumcost = s.cumcost
    n_updates = 0
    while True:
        if depth == n_steps:
            logging.info("reached maximum number of simulation steps. Exiting")
            break

        s = cur_tree.s
        logging.info("running uct-gubs for " +
                     f"state {rendering.text_render(ctx.env, s.literals)}" +
                     f" cost {s.cumcost}")

        if ctx.check_goal(s.literals):
            found_goal = True
            logging.info("found goal, exiting")
            break
        if len(cur_tree.valid_actions) == 0:
            logging.info("found deadend, exiting")

            # compute remaining cost
            future_cost = get_remaining_cost_at_deadend(
                s, ctx.cost_fn, actions, depth, ctx.horizon)
            cumcost += future_cost
            break

        cur_tree, pi, _n_updates = uct_gubs(ctx, cur_tree, actions, pi)
        logging.info("finished running uct-gubs for " +
                     f"state {rendering.text_render(ctx.env, s.literals)}" +
                     f" cost {s.cumcost}")

        # TODO -> Fix KeyError here
        a_best = pi[s]

        if depth == 0:
            action_initial_state = a_best

        logging.info(f"optimal action at initial state: {a_best}")
        logging.info(f"{mdp_tree.qs}{cur_tree.qs}")
        logging.info(
            f"value of optimal action at current state: {cur_tree.qs[a_best]}")
        cur_tree = sample_next_node(cur_tree, a_best, ctx.env)
        # compute cost of applying this action
        #  and accumulate that on variable 'cumcost'
        cost = ctx.cost_fn(s.literals, a_best)
        cumcost += cost
        n_updates += _n_updates

        depth += 1

    final_time = time.perf_counter() - start

    def pi_func(s):
        return pi[s]

    return (mdp_tree, pi_func, found_goal, cumcost, action_initial_state,
            final_time, n_updates)


def uct_gubs(ctx: context.ProblemContext, mdp_tree: tree.Tree,
             actions: frozenset, pi: dict[tuple[Literal, float], Literal]):

    logging.info("starting rollouts")

    start = time.perf_counter()
    n_updates = 0
    for i in range(ctx.n_rollouts):
        _, _, _, _n_updates = search(ctx, 0, actions, mdp_tree, pi)
        n_updates += _n_updates
    stop = time.perf_counter()
    logging.info(f"finished rollouts after {stop - start} seconds")

    return mdp_tree, pi, n_updates


def search(ctx: context.ProblemContext, depth, actions, mdp_tree: tree.Tree,
           pi):
    s = mdp_tree.s

    # if goal, return optimal value
    if check_goal(pddl.from_literals(s[0]),
                  ctx.env.problems[ctx.problem_index].goal):
        logging.debug("found goal on search")
        return mdp_tree, 0, True, 0

    # == or > ?
    if depth == ctx.horizon - 1:
        logging.debug("reached max horizon")
        return mdp_tree, 0, False, 0

    # if leaf, initialize children
    # TODO -> give option to use rollout policy and return instead of
    #   initializing values and continuing
    # TODO -> add options to disallow actions that only goes to a previous state that was already visited?
    #           - for example, in grid world domains (such as Navigation and Tireworld), moving right and then moving left on deterministic (and maybe even probabilistic) states
    if mdp_tree.is_leaf():
        # TODO -> verificar se depois do deadend ser inicializado ele ainda é uma folha e, portanto, cai nesse if
        mdp_tree.initialize_children(ctx, actions)
        if not mdp_tree.valid_actions:
            # if there aren't valid actions at the current state,
            #  then it is a deadend and its value is 0
            logging.debug("found leaf deadend on search")

            future_cost = get_remaining_cost_at_deadend(
                s, ctx.cost_fn, actions, depth, ctx.horizon)
            return mdp_tree, future_cost, False, 0

    if len(mdp_tree.valid_actions) == 0:
        # found deadend

        logging.debug("found deadend on search")
        exit("found deadend on search")
        future_cost = get_remaining_cost_at_deadend(s, ctx.cost_fn, actions,
                                                    depth, ctx.horizon)
        return mdp_tree, future_cost, False, 0
    elif len(mdp_tree.valid_actions) == 1:
        # if there's a single valid action, then it is taken
        a_best = next(iter(mdp_tree.valid_actions))
        logging.debug(f"taking only valid action {a_best}")
    else:
        # otherwise, it is chosen via the UCT equation
        a_best = uct_best_action(mdp_tree, ctx.exploration_constant,
                                 ctx.norm_exp_constant, ctx.action_tiebreaker)

    next_node = sample_next_node(mdp_tree, a_best, ctx.env)

    # TODO -> verify if cumcost is getting incremented correctly
    #           - for example, some q-values that get logged during the search
    #               change a little. This might be expected though since near
    #               the goal choices narrow down
    _, future_cost, has_goal, n_updates = search(ctx, depth + 1, actions,
                                                 next_node, pi)

    action_cost = ctx.cost_fn(s[0], a_best)

    # first fix:
    # cumcost = future_cost

    cumcost_fromnow = action_cost + future_cost
    cumcost_total = s[1] + action_cost + future_cost

    logging.debug(f"n_as: {mdp_tree.n_as}")
    n_a = mdp_tree.n_as[a_best]
    n_a_new = n_a + 1
    mdp_tree.qs[a_best] = update_q_value_estimate(
        mdp_tree.qs[a_best],
        ctx.u(cumcost_total),
        # first fix:
        # ctx.u(cumcost),
        has_goal,
        ctx.k_g,
        n_a)

    # update counts
    mdp_tree.n += 1
    mdp_tree.n_as[a_best] = n_a_new

    # if value of chosen action is better than current best,
    #   update pi[s] with it
    if s not in pi or mdp_tree.qs[a_best] > mdp_tree.qs[pi[s]]:
        logging.debug(f"update {a_best} as best action for state" +
                      f" {rendering.text_render(ctx.env, s[0])}")
        pi[s] = a_best

    # first fix:
    # return mdp_tree, cumcost, has_goal, n_updates + 1

    return mdp_tree, cumcost_fromnow, has_goal, n_updates + 1


def update_q_value_estimate(q, u_val, has_goal, k_g, n_a):
    k = k_g if has_goal else 0
    return (q * n_a + u_val + k) / (n_a + 1)


def sample_next_node(mdp_tree: tree.Tree, a: Literal, env) -> tree.Tree:
    children = mdp_tree.children

    logging.debug("sample_next_node")
    logging.debug(f"state: {rendering.text_render(env, mdp_tree.s[0])}")
    logging.debug(f"action: {a}")
    logging.debug(f"children: {children.keys()}")
    outcomes_next_node = children[a].values()
    probs_next_nodes = [outcome.prob for outcome in outcomes_next_node]
    next_nodes = [outcome.node for outcome in outcomes_next_node]

    i_next_node = np.random.choice(len(next_nodes), p=probs_next_nodes)

    return next_nodes[i_next_node]


def get_remaining_cost_at_deadend(s: ExtendedState, cost_fn,
                                  actions: frozenset[Literal], depth, horizon):
    cost = cost_fn(s.literals, next(iter(actions)))
    return cost * (horizon - 1 - depth)


def uct_value(a, qs, n, n_a, exploration_constant=SQRT_TWO, normalize=False):
    if n == 0 or n_a == 0:
        return math.inf

    if normalize:
        exploration_constant = exploration_constant * max(qs.values())

    return qs[a] + exploration_constant * math.sqrt(math.log(n) / n_a)


# selects first action from multiple maximal actions in the order
#  they are passed
def select_first_criterion(max_actions):
    if len(max_actions) == 0:
        raise IndexError("Maximal actions array is empty")

    return max_actions[0]


def uct_best_action(mdp_tree,
                    exploration_constant,
                    norm_exploration_constant=False,
                    action_selection_criterion=random.choice):
    actions = np.array(list(sorted(mdp_tree.valid_actions)))
    uct_vals = np.array([
        uct_value(a, mdp_tree.qs, mdp_tree.n, mdp_tree.n_as[a],
                  exploration_constant, norm_exploration_constant)
        for a in actions
    ])
    max_uct_val = np.max(uct_vals)
    max_actions = actions[uct_vals == max_uct_val]

    logging.debug(f"actions: {actions}")
    logging.debug(f"max_actions: {max_actions}")
    return action_selection_criterion(max_actions)


def build_std_cost_fn(goal):

    def std_cost_fn(s: frozenset[Literal], _):
        if check_goal(pddl.from_literals(s), goal):
            return 0

        return 1

    return std_cost_fn
