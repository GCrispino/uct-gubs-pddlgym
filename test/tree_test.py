import pytest

import uct_gubs.pddl as pddl

from test import fixtures
from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")

ctx = pytest.fixture(fixtures.ctx)
mdp_tree = pytest.fixture(fixtures.tireworld_mdp_tree)
mdp_tree_cost_1 = pytest.fixture(fixtures.tireworld_mdp_tree_cost_1)


# Literals
####
action_movecar12 = pddl.create_literal("movecar", 1, ["location"], ["l-1-2"])
action_movecar21 = pddl.create_literal("movecar", 1, ["location"], ["l-2-1"])
vehicleat11 = pddl.create_literal("vehicle-at", 1, ["location"], ["l-1-1"])
vehicleat12 = pddl.create_literal("vehicle-at", 1, ["location"], ["l-1-2"])
vehicleat21 = pddl.create_literal("vehicle-at", 1, ["location"], ["l-2-1"])
not_flattire = pddl.create_literal("not-flattire")
####

# States
####
s_12_notflat = (tireworld_s0.literals - frozenset({vehicleat11})).union(
    frozenset({vehicleat12}))
s_12_flat = s_12_notflat - frozenset({not_flattire})
s_21_notflat = (tireworld_s0.literals - frozenset({vehicleat11})).union(
    frozenset({vehicleat21}))
s_21_flat = s_21_notflat - frozenset({not_flattire})
####


class TestTree:

    def test_initialize_children(self, ctx, mdp_tree):

        expected_actions = frozenset({
            a
            for a in tireworld_actions
            if a in {action_movecar12, action_movecar21}
        })

        mdp_tree.initialize_children(ctx, tireworld_actions)

        assert mdp_tree.valid_actions == expected_actions

        assert mdp_tree.qs == {a: 2 for a in expected_actions}
        assert mdp_tree.n_as == {a: 0 for a in expected_actions}

        children = mdp_tree.children

        # one of the child nodes must have a state of
        #   ((vehicle-at 1-2, not-flattire) cost 1) and the other,
        #   (vehicle-at 2-1, cost 1)
        assert frozenset(children) == expected_actions

        child_movecar12 = children[action_movecar12]
        child_movecar12_states = {
            ext_state: node_outcome.prob
            for ext_state, node_outcome in child_movecar12.items()
        }

        assert child_movecar12_states == ({
            (s_12_flat, 1): 0.5,
            (s_12_notflat, 1): 0.5
        })
        for child_node_outcome in child_movecar12.values():
            assert child_node_outcome.node.depth == mdp_tree.depth + 1

        child_movecar21 = children[action_movecar21]
        child_movecar21_states = frozenset(child_movecar21)
        assert child_movecar21_states == frozenset({(s_21_flat, 1),
                                                    (s_21_notflat, 1)})
        for child_node_outcome in child_movecar21.values():
            assert child_node_outcome.node.depth == mdp_tree.depth + 1

    def test_initialize_children_init_cost_1(self, ctx, mdp_tree_cost_1):

        expected_actions = frozenset({
            a
            for a in tireworld_actions
            if a in {action_movecar12, action_movecar21}
        })

        mdp_tree_cost_1.initialize_children(ctx, tireworld_actions)

        assert mdp_tree_cost_1.valid_actions == expected_actions

        assert mdp_tree_cost_1.qs == {a: 2 for a in expected_actions}
        assert mdp_tree_cost_1.n_as == {a: 0 for a in expected_actions}

        children = mdp_tree_cost_1.children

        # one of the child nodes must have a state of
        #   ((vehicle-at 1-2, not-flattire) cost 2) and the other,
        #   (vehicle-at 2-1, cost 2)
        assert frozenset(children) == expected_actions

        child_movecar12 = children[action_movecar12]
        child_movecar12_states = {
            ext_state: node_outcome.prob
            for ext_state, node_outcome in child_movecar12.items()
        }

        assert child_movecar12_states == ({
            (s_12_flat, 2): 0.5,
            (s_12_notflat, 2): 0.5
        })
        for child_node_outcome in child_movecar12.values():
            assert child_node_outcome.node.depth == mdp_tree_cost_1.depth + 1

        child_movecar21 = children[action_movecar21]
        child_movecar21_states = frozenset(child_movecar21)
        assert child_movecar21_states == frozenset({(s_21_flat, 2),
                                                    (s_21_notflat, 2)})
        for child_node_outcome in child_movecar21.values():
            assert child_node_outcome.node.depth == mdp_tree_cost_1.depth + 1

    def test_size(self, ctx, mdp_tree):
        size = mdp_tree.size()
        assert size == 1

        mdp_tree.initialize_children(ctx, tireworld_actions)
        size = mdp_tree.size()
        assert size == 5
