import logging
from dataclasses import dataclass
from typing import Callable

from pddlgym.structs import Literal

from uct_gubs.pddl import get_valid_actions_and_outcomes
from uct_gubs.mdp.types import ExtendedState


@dataclass
class NodeOutcome:
    prob: float
    node: "Tree"


@dataclass
class Tree:
    n: int
    s: ExtendedState
    valid_actions: frozenset[Literal]
    depth: int
    n_as: dict[Literal, int]
    qs: dict[Literal, float]
    children: dict[Literal, dict[ExtendedState, "NodeOutcome"]]

    def is_leaf(self):
        return all((c is None for c in self.children.values()))

    def initialize_children(self, actions, cost_fn, h, env, h_weight=0):
        logging.debug("initializing children")
        valid_actions_outcomes = get_valid_actions_and_outcomes(
            self.s[0], actions, env)

        self.valid_actions = frozenset(valid_actions_outcomes)
        logging.debug("found following valid actions on initialization: " +
                      f"{self.valid_actions}")

        self.n = h_weight
        for a, outcomes in valid_actions_outcomes.items():
            # initialize q-value with heuristic for current state
            # TODO -> incorporate prob heuristic here
            self.qs[a] = h(self.s[0])
            self.n_as[a] = 0

            # get new cumcost for next states
            new_cost = cost_fn(self.s[0], a)

            # initialize each child's subtree
            self.children[a] = {}
            for outcome in outcomes:
                ext_state = ExtendedState(outcome.literals, new_cost)
                self.children[a][ext_state] = NodeOutcome(
                    prob=outcome.prob,
                    node=new_tree(ext_state, self.depth + 1, actions))

    def traverse(self, fn: Callable[["Tree"], None]):
        fn(self)

        if not self.children:
            return

        for child_outcomes_dict in self.children.values():
            for child_outcome in child_outcomes_dict.values():
                child_outcome.node.traverse(fn)


def new_tree(s: ExtendedState, depth, actions) -> Tree:
    return Tree(0, s, actions, depth, n_as={}, qs={}, children={})
