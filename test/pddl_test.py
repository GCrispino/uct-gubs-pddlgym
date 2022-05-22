import unittest

import uct_gubs.pddl as pddl
from uct_gubs.mdp.types import StateOutcome

from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")


class TestPDDL(unittest.TestCase):

    def test_get_valid_actions_and_outcomes(self):
        movecar12_str = "movecar(l-1-2:location)"
        movecar21_str = "movecar(l-2-1:location)"
        expected_actions_strs = frozenset({movecar12_str, movecar21_str})
        expected_actions = frozenset(
            {a
             for a in tireworld_actions if str(a) in expected_actions_strs})

        move_action_outcomes = pddl.get_valid_actions_and_outcomes(
            tireworld_s0.literals, tireworld_actions, tireworld_env)

        actual_actions = frozenset(move_action_outcomes)
        assert expected_actions == actual_actions

        for a, outcomes in move_action_outcomes.items():
            if str(a) == movecar21_str:
                # this action has two outcomes - flat tire or not
                assert len(outcomes) == 2
                outcomes_list = sorted(list(outcomes))
                vehicle_ats = [
                    pddl.get_literals_by_name(outcome.literals, "vehicle-at")
                    for outcome in outcomes_list
                ]
                probs_vehicle_ats = [outcome.prob for outcome in outcomes_list]

                assert str(vehicle_ats) == (
                    "[frozenset({vehicle-at(l-2-1:location)}), " +
                    "frozenset({vehicle-at(l-2-1:location)})]")
                assert probs_vehicle_ats == [0.5, 0.5]

                not_flattires = [
                    pddl.get_literals_by_name(outcome.literals, "not-flattire")
                    for outcome in outcomes_list
                ]
                probs_not_flattires = [
                    outcome.prob for outcome in outcomes_list
                ]
                assert str(not_flattires
                           ) == "[frozenset(), frozenset({not-flattire()})]"
                assert probs_not_flattires == [0.5, 0.5]

    def test_filter_superfluous_actions_same_state(self):
        a1, a2, a3, a4 = [pddl.create_literal(f"a{i}") for i in range(1, 5)]
        (lit1, lit2, lit3, lit4, lit5,
         lit6) = [pddl.create_literal(f"lit{i}") for i in range(1, 7)]

        s0 = frozenset({lit3, lit5})
        actions_outcomes = {
            a1:
            frozenset({
                StateOutcome(prob=0.8, literals=frozenset({lit1, lit2})),
                StateOutcome(prob=0.2, literals=frozenset({lit1, lit3}))
            }),
            a2:
            frozenset({
                StateOutcome(prob=0.4, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.6, literals=frozenset({lit3, lit5}))
            }),
            a3:
            frozenset({StateOutcome(prob=1, literals=frozenset({lit3,
                                                                lit5}))}),
            a4:
            frozenset({StateOutcome(prob=1, literals=frozenset({lit3,
                                                                lit5}))}),
        }

        expected_reasonable_actions_outcomes = {
            a1:
            frozenset({
                StateOutcome(prob=0.8, literals=frozenset({lit1, lit2})),
                StateOutcome(prob=0.2, literals=frozenset({lit1, lit3}))
            }),
            a2:
            frozenset({
                StateOutcome(prob=0.4, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.6, literals=frozenset({lit3, lit5}))
            }),
        }
        actual_reasonable_actions_outcomes = pddl.filter_superfluous_actions(
            s0, actions_outcomes)

        assert (expected_reasonable_actions_outcomes ==
                actual_reasonable_actions_outcomes)

    def test_filter_superfluous_actions_same_outcomes(self):
        (a1, a2, a3, a4,
         a5) = [pddl.create_literal(f"a{i}") for i in range(1, 6)]
        (lit1, lit2, lit3, lit4, lit5,
         lit6) = [pddl.create_literal(f"lit{i}") for i in range(1, 7)]

        s0 = frozenset({lit3, lit5})
        actions_outcomes = {
            a1:
            frozenset({
                StateOutcome(prob=0.8, literals=frozenset({lit1, lit2})),
                StateOutcome(prob=0.2, literals=frozenset({lit1, lit3}))
            }),
            a2:
            frozenset({
                StateOutcome(prob=0.4, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.6, literals=frozenset({lit3, lit5}))
            }),
            a3:
            frozenset({StateOutcome(prob=1, literals=frozenset({lit3,
                                                                lit5}))}),
            a4:
            frozenset({
                StateOutcome(prob=0.4, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.6, literals=frozenset({lit3, lit5}))
            }),
            a5:
            frozenset({
                StateOutcome(prob=0.3, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.7, literals=frozenset({lit3, lit5}))
            }),
        }

        expected_reasonable_actions_outcomes = {
            a1:
            frozenset({
                StateOutcome(prob=0.8, literals=frozenset({lit1, lit2})),
                StateOutcome(prob=0.2, literals=frozenset({lit1, lit3}))
            }),
            a2:
            frozenset({
                StateOutcome(prob=0.4, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.6, literals=frozenset({lit3, lit5}))
            }),
            a5:
            frozenset({
                StateOutcome(prob=0.3, literals=frozenset({lit1, lit4})),
                StateOutcome(prob=0.7, literals=frozenset({lit3, lit5}))
            }),
        }
        actual_reasonable_actions_outcomes = pddl.filter_superfluous_actions(
            s0, actions_outcomes)

        assert actual_reasonable_actions_outcomes != actions_outcomes
        assert (expected_reasonable_actions_outcomes ==
                actual_reasonable_actions_outcomes)
