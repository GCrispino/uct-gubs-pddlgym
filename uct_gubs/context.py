from dataclasses import dataclass
from typing import Callable

from pddlgym.core import PDDLEnv
from pddlgym.inference import check_goal as _check_goal
from pddlgym.structs import Literal

from uct_gubs import pddl


@dataclass
class ProblemContext:
    env: PDDLEnv
    s0: frozenset[Literal]
    problem_index: int
    h_u: Callable[[frozenset[Literal]], float]
    h_p: Callable[[frozenset[Literal]], float]
    action_tiebreaker: Callable[[list[Literal]], Literal]
    init_count: int
    u: Callable[[float], float]
    cost_fn: Callable[[frozenset[Literal], Literal], float]
    exploration_constant: float
    norm_exp_constant: bool
    k_g: float
    n_rollouts: int
    horizon: int

    # TODO -> move to other module
    def check_goal(self, s: frozenset[Literal]):
        return _check_goal(pddl.from_literals(s),
                           self.env.problems[self.problem_index].goal)
