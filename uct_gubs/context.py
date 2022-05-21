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
    h: Callable[[frozenset[Literal]], float]
    init_count: float
    u: Callable[[float], float]
    cost_fn: Callable[[frozenset[Literal], Literal], float]
    exploration_constant: float
    k_g: float
    n_rollouts: int
    horizon: int

    def check_goal(self, s: frozenset[Literal]):
        return _check_goal(pddl.from_literals(s),
                           self.env.problems[self.problem_index].goal)
