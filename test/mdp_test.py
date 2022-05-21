import math
import unittest

import numpy as np
import pddlgym.core as pddlcore
from pddlgym.structs import Literal, Predicate

import uct_gubs.mdp.general as mdp
from uct_gubs import context, pddl, tree
from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")

exploration_constant = math.sqrt(2)
actions = np.array([Literal(Predicate("pred", 1), [val]) for val in range(4)])


# dummy heuristic function
def h(_):
    return 1


def build_dict_from_actions(actions):
    return lambda it: dict(zip(actions, it))


class TestUCTValue(unittest.TestCase):

    def test_uct_value(self):
        q = 1
        n = 10
        n_a = 3

        uct_val = mdp.uct_value(q, n, n_a, exploration_constant)
        assert uct_val == 2.2389740629499464

    def test_uct_value_first_visit_node(self):
        q = 1
        n = 0
        n_a = 0

        uct_val = mdp.uct_value(q, n, n_a, exploration_constant)
        assert uct_val == math.inf

    def test_uct_value_first_visit_action(self):
        q = 1
        n = 10
        n_a = 0

        uct_val = mdp.uct_value(q, n, n_a, exploration_constant)
        assert uct_val == math.inf


class TestUCTBestAction(unittest.TestCase):

    def test_uct_best_action(self):
        dict_from_actions = build_dict_from_actions(actions)
        n = 10
        n_as = dict_from_actions([1, 2, 3, 4])
        qs = dict_from_actions([10, 3, 100, 50])
        mdp_tree = tree.Tree(n=n,
                             s=frozenset(),
                             valid_actions=actions,
                             depth=0,
                             n_as=n_as,
                             qs=qs,
                             children=None)
        best_action = mdp.uct_best_action(mdp_tree, exploration_constant)
        assert best_action == actions[2]

    def test_uct_best_action_large_constant(self):
        dict_from_actions = build_dict_from_actions(actions)
        n = 80
        n_as = dict_from_actions([3, 2, 70, 5])
        qs = dict_from_actions([10, 3, 100, 50])
        mdp_tree = tree.Tree(n=n,
                             s=frozenset(),
                             valid_actions=actions,
                             depth=0,
                             n_as=n_as,
                             qs=qs,
                             children=None)
        best_action = mdp.uct_best_action(mdp_tree, 80)
        assert best_action == actions[3]

    def test_uct_best_action_first_visit(self):
        dict_from_actions = build_dict_from_actions(actions)
        n = 0
        n_as = dict_from_actions([0, 0, 0, 0])
        qs = dict_from_actions([0, 0, 0, 0])

        mdp_tree = tree.Tree(n=n,
                             s=frozenset(),
                             valid_actions=actions,
                             depth=0,
                             n_as=n_as,
                             qs=qs,
                             children=None)
        best_action = mdp.uct_best_action(mdp_tree, exploration_constant)
        assert best_action == actions[0]

    def test_uct_best_action_only_not_visited(self):
        dict_from_actions = build_dict_from_actions(actions)
        n = 10
        n_as = dict_from_actions([2, 0, 3, 5])
        qs = dict_from_actions([3, 0, 100, 50])
        print("n_as:", n_as)
        print("qs:", qs)
        mdp_tree = tree.Tree(n=n,
                             s=frozenset(),
                             valid_actions=actions,
                             depth=0,
                             n_as=n_as,
                             qs=qs,
                             children=None)
        best_action = mdp.uct_best_action(mdp_tree, exploration_constant)

        assert best_action == actions[1]


class TestUpdateQEstimate(unittest.TestCase):

    def test_update_q_estimate(self):
        q = 0.5
        u_val = 0.8
        k_g = 1
        n_a = 10
        q_goal = mdp.update_q_value_estimate(q, u_val, True, k_g, n_a)
        q_nogoal = mdp.update_q_value_estimate(q, u_val, False, k_g, n_a)

        assert round(q_goal, 5) == 0.61818
        assert round(q_nogoal, 5) == 0.52727


class TestSearch(unittest.TestCase):

    def test_sample_next_node(self):

        mdp_tree = tree.new_tree((tireworld_s0.literals, 0), 0,
                                 tireworld_actions)
        mdp_tree.initialize_children(tireworld_actions,
                                     mdp.build_std_cost_fn(tireworld_goal), h,
                                     tireworld_env)
        action_movecar12 = pddl.create_literal("movecar", 1, ["location"],
                                               ["l-1-2"])
        sampled_next_node = mdp.sample_next_node(
            mdp_tree, action_movecar12, mdp.build_std_cost_fn(tireworld_goal),
            tireworld_env)
        assert sampled_next_node.s in mdp_tree.children[action_movecar12]
        assert sampled_next_node == mdp_tree.children[action_movecar12][
            sampled_next_node.s].node

    def test_search(self):
        # instantiate/create domain and tree
        mdp_tree = tree.new_tree((tireworld_s0.literals, 0), 0,
                                 tireworld_actions)

        lamb = -0.1
        k_g = 1
        n_rollouts = 0  # not needed for this test
        horizon = 10
        ctx = context.ProblemContext(tireworld_env, tireworld_s0, 0, h, 0,
                                     mdp.risk_exp_fn(lamb),
                                     mdp.build_std_cost_fn(tireworld_goal),
                                     mdp.SQRT_TWO, k_g, n_rollouts, horizon)
        pi = {}

        # run search
        new_mdp_tree, cumcost, has_goal = mdp.search(ctx, 0, tireworld_actions,
                                                     mdp_tree, pi)

        assert new_mdp_tree == mdp_tree

        # depth should be less than or equal to horizon
        assert new_mdp_tree.depth <= ctx.horizon
        # accumulate cost should be no bigger than H - 1
        assert cumcost <= ctx.horizon - 1

        # assert states in policy dict
        pi_states = set(pi)

        # find set of states in tree
        states_visited = set()

        def add_state_if_visited_callback(node):
            if node.n != 0:
                states_visited.add(node.s)

        new_mdp_tree.traverse(add_state_if_visited_callback)

        assert set(states_visited) == pi_states

        # assert valid action set for each state in the tree is correct
        def get_valid_actions_callback(node):
            valid_actions_dict = pddl.get_valid_actions_and_outcomes(
                node.s[0], tireworld_actions, tireworld_env)

            valid_actions_actual = set(valid_actions_dict)

            valid_actions_expected = set()
            for a in tireworld_actions:
                try:
                    frozenset({
                        s_.literals
                        for s_ in pddlcore.get_successor_states(
                            pddl.from_literals(node.s[0]),
                            a,
                            tireworld_env.domain,
                            raise_error_on_invalid_action=True)
                    })
                except pddlcore.InvalidAction:
                    continue
                valid_actions_expected.add(a)
            assert valid_actions_expected == valid_actions_actual

        new_mdp_tree.traverse(get_valid_actions_callback)

    def test_search_deadend(self):

        # deadend literal
        not_flattire = pddl.create_literal("not-flattire")

        # instantiate/create domain and tree
        mdp_tree = tree.new_tree((frozenset({not_flattire}), 0), 0,
                                 tireworld_actions)

        lamb = -0.1
        k_g = 1
        n_rollouts = 0  # not needed for this test
        horizon = 10
        ctx = context.ProblemContext(tireworld_env, tireworld_s0, 0, h, 0,
                                     mdp.risk_exp_fn(lamb),
                                     mdp.build_std_cost_fn(tireworld_goal),
                                     mdp.SQRT_TWO, k_g, n_rollouts, horizon)
        pi = {}

        # run search
        new_mdp_tree, cumcost, has_goal = mdp.search(ctx, 0, tireworld_actions,
                                                     mdp_tree, pi)

        assert not has_goal
        assert cumcost == horizon - 1

        max_depth = 0

        def find_max_depth_callback(node):
            nonlocal max_depth
            max_depth = max(max_depth, node.depth)

        new_mdp_tree.traverse(find_max_depth_callback)
