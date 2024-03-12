import math

import numpy as np
import pddlgym.core as pddlcore
import pytest
from pddlgym.structs import Literal, Predicate

import uct_gubs.mdp.general as mdp
from uct_gubs import pddl, tree
from uct_gubs.mdp.types import ExtendedState

from test import fixtures
from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")

exploration_constant = math.sqrt(2)
actions = np.array([Literal(Predicate("pred", 1), [val]) for val in range(4)])

ctx = pytest.fixture(fixtures.ctx)
mdp_tree = pytest.fixture(fixtures.tireworld_mdp_tree)


def build_dict_from_actions(actions):
    return lambda it: dict(zip(actions, it))


class TestUCTValue:

    def test_uct_value(self):
        a = pddl.create_literal("a")
        qs = {a: 1}
        n = 10
        n_a = 3

        uct_val = mdp.uct_value(a, qs, n, n_a, exploration_constant)
        assert uct_val == 2.2389740629499464

    def test_uct_value_first_visit_node(self):
        a = pddl.create_literal("a")
        qs = {a: 1}
        n = 0
        n_a = 0

        uct_val = mdp.uct_value(a, qs, n, n_a, exploration_constant)
        assert uct_val == math.inf

    def test_uct_value_first_visit_action(self):
        a = pddl.create_literal("a")
        qs = {a: 1}
        n = 10
        n_a = 0

        uct_val = mdp.uct_value(a, qs, n, n_a, exploration_constant)
        assert uct_val == math.inf

    def test_uct_value_with_normalization(self):
        a = pddl.create_literal("a")
        qs = {a: 2}
        n = 10
        n_a = 3

        uct_val = mdp.uct_value(a,
                                qs,
                                n,
                                n_a,
                                exploration_constant,
                                normalize=True)
        assert uct_val == 4.477948125899893


class TestUCTBestAction:

    def test_uct_best_action(self, mdp_tree):
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
        best_action = mdp.uct_best_action(
            mdp_tree,
            exploration_constant,
            action_selection_criterion=mdp.select_first_criterion)
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
        best_action = mdp.uct_best_action(
            mdp_tree,
            80,
            action_selection_criterion=mdp.select_first_criterion)

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
        best_action = mdp.uct_best_action(
            mdp_tree,
            exploration_constant,
            action_selection_criterion=mdp.select_first_criterion)
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
        best_action = mdp.uct_best_action(
            mdp_tree,
            exploration_constant,
            action_selection_criterion=mdp.select_first_criterion)

        assert best_action == actions[1]


class TestUpdateQEstimate:

    def test_update_q_estimate(self):
        q = 0.5
        u_val = 0.8
        k_g = 1
        n_a = 10
        q_goal = mdp.update_q_value_estimate(q, u_val, True, k_g, n_a)
        q_nogoal = mdp.update_q_value_estimate(q, u_val, False, k_g, n_a)

        assert round(q_goal, 5) == 0.61818
        assert round(q_nogoal, 5) == 0.52727


class TestSearch:

    def test_sample_next_node(self, ctx, mdp_tree):

        mdp_tree.initialize_children(ctx, tireworld_actions)
        action_movecar12 = pddl.create_literal("movecar", 1, ["location"],
                                               ["l-1-2"])
        sampled_next_node = mdp.sample_next_node(mdp_tree, action_movecar12,
                                                 tireworld_env)
        assert sampled_next_node.s in mdp_tree.children[action_movecar12]
        assert sampled_next_node == mdp_tree.children[action_movecar12][
            sampled_next_node.s].node

    def test_search(self, ctx, mdp_tree):
        pi = {}

        # run search
        new_mdp_tree, cumcost, has_goal, n_updates = mdp.search(
            ctx, 0, tireworld_actions, mdp_tree, pi)

        depth = new_mdp_tree.get_depth()
        assert n_updates > 0
        assert n_updates == depth

        assert new_mdp_tree == mdp_tree

        # depth should be less than or equal to horizon
        assert new_mdp_tree.depth <= ctx.horizon
        # accumulate cost should be no bigger than H - 1
        assert cumcost <= ctx.horizon - 1

        maxcost = 0

        def get_max_cost(tree):
            nonlocal maxcost
            maxcost = max(maxcost, tree.s.cumcost)

        new_mdp_tree.traverse(get_max_cost)
        if has_goal:
            # accumulate cost should be no bigger than
            #   max depth *in this domain*
            assert cumcost <= depth
            assert maxcost <= depth
            assert cumcost != 0

        # assert states in policy dict
        pi_states = set(pi)

        # find set of states in tree
        states_visited = set()

        new_mdp_tree.traverse_and_accumulate(
            lambda visited, node: states_visited.add(node.s)
            if node.n != 0 else visited, states_visited)

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

    def test_search_deadend(self, ctx):

        # deadend literal
        deadend = ExtendedState(frozenset(), 0)

        # instantiate/create domain and tree
        mdp_tree = tree.new_tree(deadend, 0, tireworld_actions)

        horizon = 10
        pi = {}

        # run search
        new_mdp_tree, cumcost, has_goal, n_updates = mdp.search(
            ctx, 0, tireworld_actions, mdp_tree, pi)

        assert n_updates == 0
        assert not has_goal
        assert cumcost == horizon - 1

        max_depth = new_mdp_tree.get_depth()
        assert max_depth == 0

    def test_search_goal(self, ctx):
        # literals
        vehicle_at_l_1_1_literal = pddl.create_literal("vehicle-at", 1,
                                                       ["location"], ["l-1-1"])
        vehicle_at_l_2_2_literal = pddl.create_literal("vehicle-at", 1,
                                                       ["location"], ["l-2-2"])
        vehicle_at_l_3_1_literal = pddl.create_literal("vehicle-at", 1,
                                                       ["location"], ["l-3-1"])
        vehicle_at_l_1_3_literal = pddl.create_literal("vehicle-at", 1,
                                                       ["location"], ["l-1-3"])

        # literal sets
        vehicle_at_l_2_2_state_literals = tireworld_s0.literals - frozenset(
            {vehicle_at_l_1_1_literal}) | frozenset({vehicle_at_l_2_2_literal})
        vehicle_at_l_3_1_state_literals = tireworld_s0.literals - frozenset(
            {vehicle_at_l_1_1_literal}) | frozenset({vehicle_at_l_3_1_literal})
        goal_state_literals = tireworld_s0.literals - frozenset(
            {vehicle_at_l_1_1_literal}) | frozenset({vehicle_at_l_1_3_literal})

        # extended states
        goal_state = ExtendedState(goal_state_literals, 4)
        vehicle_at_l_2_2_state = ExtendedState(vehicle_at_l_2_2_state_literals,
                                               3)

        vehicle_at_l_3_1_state = ExtendedState(vehicle_at_l_3_1_state_literals,
                                               2)

        # actions
        dummy_action_a = pddl.create_literal("a")
        dummy_action_b = pddl.create_literal("b")
        actions = frozenset({dummy_action_a, dummy_action_b})

        # state nodes
        vehicle_at_l_3_1_tree = tree.new_tree(vehicle_at_l_3_1_state, 2,
                                              actions)

        vehicle_at_l_2_2_tree = tree.new_tree(vehicle_at_l_2_2_state, 3,
                                              actions)
        goal_tree = tree.new_tree(goal_state, 4, actions)
        goal_tree.qs = {dummy_action_a: 0, dummy_action_b: 0}
        goal_tree.n_as = {dummy_action_a: 0, dummy_action_b: 0}

        vehicle_at_l_2_2_tree.children = {
            dummy_action_a: {
                goal_tree.s: tree.NodeOutcome(prob=1, node=goal_tree)
            }
        }
        vehicle_at_l_2_2_tree.qs = {dummy_action_a: 0, dummy_action_b: 0}
        vehicle_at_l_2_2_tree.n_as = {dummy_action_a: 0, dummy_action_b: 2}

        vehicle_at_l_3_1_tree.children = {
            dummy_action_a: {
                vehicle_at_l_2_2_tree.s:
                tree.NodeOutcome(prob=1, node=vehicle_at_l_2_2_tree)
            },
            dummy_action_b: {  # make action b the same just so it's not empty
                vehicle_at_l_2_2_tree.s:
                tree.NodeOutcome(prob=1, node=vehicle_at_l_2_2_tree)
            }
        }
        vehicle_at_l_3_1_tree.qs = {dummy_action_a: 0, dummy_action_b: 0}
        vehicle_at_l_3_1_tree.n_as = {dummy_action_a: 0, dummy_action_b: 2}

        # run search from state at l-3-1
        _, cumcost, has_goal, n_updates = mdp.search(ctx, 2, actions,
                                                     vehicle_at_l_3_1_tree, {})

        # assertions
        assert has_goal
        assert cumcost == 2
        assert n_updates == 2

        u = ctx.u(vehicle_at_l_3_1_state.cumcost + 2) + ctx.k_g
        assert vehicle_at_l_2_2_tree.n_as == {
            dummy_action_a: 1,
            dummy_action_b: 2
        }
        assert vehicle_at_l_2_2_tree.qs == {
            dummy_action_a: u,
            dummy_action_b: 0
        }
