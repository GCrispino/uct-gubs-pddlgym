import logging
import math
import time

import numpy as np
import pddlgym.core as pddlcore
from pddlgym.inference import check_goal
from pddlgym.structs import Literal

from uct_gubs import context, error, tree, pddl, rendering

SQRT_TWO = math.sqrt(2)


def risk_exp_fn(lamb):
    return lambda cumcost: np.exp(lamb * cumcost)


def simulate_with_uct_gubs(ctx: context.ProblemContext,
                           s: tuple[frozenset[Literal],
                                    float], actions: frozenset, n_steps: int):
    mdp_tree = tree.new_tree(s, 0, actions)
    pi = {}

    cur_tree = mdp_tree
    i = 0
    while True:
        if i == n_steps - 1:
            logging.info("reached maximum number of simulation steps. Exiting")

        s = cur_tree.s
        logging.info("running uct-gubs for " +
                     f"state {rendering.text_render(ctx.env, s[0])}" +
                     f" cost {s[1]}")

        if ctx.check_goal(s[0]):
            logging.info("found goal, exiting")
            break
        if len(cur_tree.valid_actions) == 0:
            logging.info("found deadend, exiting")
            break

        cur_tree, pi, n_updates = uct_gubs(ctx, cur_tree, actions, pi)
        logging.info("finished running uct-gubs for " +
                     f"state {rendering.text_render(ctx.env, s[0])}" +
                     f" cost {s[1]}")
        a_best = pi[s]
        logging.info(f"optimal action at initial state: {a_best}")
        logging.info(f"{mdp_tree.qs}{cur_tree.qs}")
        logging.info(
            f"value of optimal action at current state: {cur_tree.qs[a_best]}")
        cur_tree = sample_next_node(cur_tree, a_best, ctx.cost_fn, ctx.env)

    def pi_func(s):
        logging.debug(f"{len(pi)}")
        logging.debug("states in pi:")
        for s_ in pi:
            logging.debug(f"  {s_}")
        return pi[s]

    return mdp_tree, pi_func, n_updates


def uct_gubs(ctx: context.ProblemContext, mdp_tree: tree.Tree,
             actions: frozenset, pi: dict[tuple[Literal, float], Literal]):
    logging.debug(f"type of mdp_tree: {type(mdp_tree)}")

    logging.info("starting rollouts")

    start = time.perf_counter()
    for i in range(ctx.n_rollouts):
        search(ctx, 0, actions, mdp_tree, pi)
    logging.debug(f"type of mdp_tree: {type(mdp_tree)}")
    stop = time.perf_counter()
    logging.info(f"finished rollouts after {stop - start} seconds")

    # TODO -> count number of updates
    n_updates = 0
    return mdp_tree, pi, n_updates


def search(ctx: context.ProblemContext, depth, actions, mdp_tree, pi):
    s = mdp_tree.s

    # if goal, return optimal value
    if check_goal(pddl.from_literals(s[0]),
                  ctx.env.problems[ctx.problem_index].goal):
        logging.debug("found goal on search")
        return mdp_tree, 0, True

    # == or > ?
    if depth == ctx.horizon - 1:
        logging.debug("reached max horizon")
        return mdp_tree, 0, False

    # if leaf, initialize children
    # TODO -> give option to use rollout policy and return instead of
    #   initializing values and continuing
    if mdp_tree.is_leaf():
        mdp_tree.initialize_children(actions, ctx.cost_fn, ctx.h, ctx.env)
        if not mdp_tree.valid_actions:
            # if there aren't valid actions at the current state,
            #  then it is a deadend and its value is 0
            logging.debug("found deadend on search")

            cost = ctx.cost_fn(s[0], next(iter(actions)))
            future_cost = cost * (ctx.horizon - 1 - depth)
            return mdp_tree, future_cost, False
        # TODO -> if len(valid_actions) == 1, then algorithm can be optimized

    a_best = uct_best_action(mdp_tree, ctx.exploration_constant)
    cost = ctx.cost_fn(s[0], a_best)

    next_node = sample_next_node(mdp_tree, a_best, ctx.cost_fn, ctx.env)

    _, future_cost, has_goal = search(ctx, depth + 1, actions, next_node, pi)

    cumcost = cost + future_cost
    logging.debug(f"n_as: {mdp_tree.n_as}")
    n_a = mdp_tree.n_as[a_best]
    n_a_new = n_a + 1
    mdp_tree.qs[a_best] = update_q_value_estimate(mdp_tree.qs[a_best],
                                                  ctx.u(cumcost), has_goal,
                                                  ctx.k_g, n_a)

    # update counts
    mdp_tree.n += 1
    mdp_tree.n_as[a_best] = n_a_new

    # if value of chosen action is better than current best,
    #   update pi[s] with it
    if s not in pi or mdp_tree.qs[a_best] > mdp_tree.qs[pi[s]]:
        logging.debug(f"update {a_best} as best action for state" +
                      f" {rendering.text_render(ctx.env, s[0])}")
        pi[s] = a_best

    return mdp_tree, cumcost, has_goal


def update_q_value_estimate(q, u_val, has_goal, k_g, n_a):
    k = k_g if has_goal else 0
    return (q * n_a + u_val + k) / (n_a + 1)


def sample_next_node(mdp_tree, a, cost_fn, env):
    children = mdp_tree.children
    next_state = pddlcore.get_successor_state(
        pddl.from_literals(mdp_tree.s[0]), a, env.domain)

    cost = cost_fn(next_state, a)
    logging.debug("sample_next_node")
    logging.debug(f"state: {rendering.text_render(env, mdp_tree.s[0])}")
    logging.debug(f"action: {a}")
    logging.debug(f"children: {children.keys()}")
    next_nodes = [
        child for s_, child in children[a].items()
        if s_ == (next_state.literals, cost)
    ]

    assert len(next_nodes) == 1, error.MATCHING_CHILD_SAMPLING_ERROR

    return next_nodes[0]


# TODO -> see if we can normalize the exploration constant
#   according to the costs of the problem
def uct_value(q, n, n_a, exploration_constant=SQRT_TWO):
    if n == 0 or n_a == 0:
        return math.inf

    return q + exploration_constant * math.sqrt(math.log(n) / n_a)


# selects one action from multiple maximal actions
def select_first_criterion(max_actions):
    if len(max_actions) == 0:
        raise IndexError("Maximal actions array is empty")

    return max_actions[0]


def uct_best_action(mdp_tree,
                    exploration_constant,
                    action_selection_criterion=select_first_criterion):
    actions = np.array(list(sorted(mdp_tree.valid_actions)))
    uct_vals = np.array([
        uct_value(mdp_tree.qs[a], mdp_tree.n, mdp_tree.n_as[a],
                  exploration_constant) for a in actions
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
