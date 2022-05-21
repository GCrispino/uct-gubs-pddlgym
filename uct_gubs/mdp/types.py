from typing import NamedTuple

from pddlgym.structs import Literal


class ExtendedState(NamedTuple):
    literals: frozenset[Literal]
    cumcost: float


class StateOutcome(NamedTuple):
    prob: float
    literals: frozenset[Literal]
