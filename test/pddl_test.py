import unittest

import uct_gubs.pddl as pddl
from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")


class TestPDDL(unittest.TestCase):

    def test_get_valid_actions_successors(self):
        movecar12_str = "movecar(l-1-2:location)"
        movecar21_str = "movecar(l-2-1:location)"
        expected_actions_strs = frozenset({movecar12_str, movecar21_str})
        expected_actions = frozenset(
            {a
             for a in tireworld_actions if str(a) in expected_actions_strs})

        move_action_successors = pddl.get_valid_actions_and_successors(
            tireworld_s0.literals, tireworld_actions, tireworld_env)

        actual_actions = frozenset(move_action_successors)
        assert expected_actions == actual_actions

        for a, succs in move_action_successors.items():
            if str(a) == movecar21_str:
                # this action has a two outcomes - flat tire or not
                assert len(succs) == 2
                succs_list = sorted(list(succs))
                vehicle_ats = [
                    pddl.get_literals_by_name(succ, "vehicle-at")
                    for succ in succs_list
                ]
                assert str(vehicle_ats) == (
                    "[frozenset({vehicle-at(l-2-1:location)}), " +
                    "frozenset({vehicle-at(l-2-1:location)})]")

                not_flattires = [
                    pddl.get_literals_by_name(succ, "not-flattire")
                    for succ in succs_list
                ]
                assert str(not_flattires
                           ) == "[frozenset(), frozenset({not-flattire()})]"
