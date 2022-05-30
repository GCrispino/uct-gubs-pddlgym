import logging
from dataclasses import dataclass
from math import inf
from typing import Callable

from pddlgym.structs import Literal

from uct_gubs import pddl
from uct_gubs.context import ProblemContext
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

    def initialize_children(self, ctx: ProblemContext, actions):
        logging.debug("initializing children")
        initial_valid_actions_outcomes = pddl.get_valid_actions_and_outcomes(
            self.s[0], actions, ctx.env)

        # remove redundant/superfluous actions
        valid_actions_outcomes = pddl.filter_superfluous_actions(
            self.s[0], initial_valid_actions_outcomes)

        if (redundant_actions := set(valid_actions_outcomes) -
                set(initial_valid_actions_outcomes)) != set({}):
            logging.debug(f"detected {len(redundant_actions)} actions:" +
                          f" {redundant_actions}")

        self.valid_actions = frozenset(valid_actions_outcomes)
        logging.debug("found following valid actions on initialization: " +
                      f"{self.valid_actions}")

        self.n = ctx.init_count
        for a, outcomes in valid_actions_outcomes.items():
            # initialize q-value with heuristic for current state
            self.qs[a] = ctx.h_u(self.s[0]) + ctx.h_p(self.s[0]) * ctx.k_g
            self.n_as[a] = 0

            # get new cumcost for next states
            new_cost = ctx.cost_fn(self.s[0], a)

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

    def size(self) -> int:
        size = 0

        def count(tree):
            nonlocal size
            size += 1

        self.traverse(count)

        return size

    def best_action(self) -> Literal:
        max_q = -inf
        a_max = None
        for a, q in self.qs.items():
            if q > max_q:
                max_q = q
                a_max = a

        return a_max

    def v(self) -> float:
        return self.qs[self.best_action]


def new_tree(s: ExtendedState, depth, actions) -> Tree:
    return Tree(0, s, actions, depth, n_as={}, qs={}, children={})
