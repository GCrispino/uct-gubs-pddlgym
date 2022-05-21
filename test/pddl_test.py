import unittest

import uct_gubs.pddl as pddl
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
