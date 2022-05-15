import logging
from dataclasses import dataclass

from pddlgym.structs import Literal

from uct_gubs.pddl import get_valid_actions_and_successors


@dataclass
class Tree:
    n: int
    s: tuple[frozenset[Literal], float]
    valid_actions: frozenset[Literal]
    depth: int
    n_as: dict[Literal, int]
    qs: dict[Literal, float]
    children: dict[Literal, dict[tuple[frozenset[Literal], float], "Tree"]]

    def is_leaf(self):
        return all((c is None for c in self.children.values()))

    def initialize_children(self, actions, cost_fn, h, env):
        logging.debug("initializing children")
        valid_actions_successors = get_valid_actions_and_successors(
            self.s[0], actions, env)

        self.valid_actions = frozenset(valid_actions_successors)
        logging.debug("found following valid actions on initialization: " +
                      f"{self.valid_actions}")

        for a, succ in valid_actions_successors.items():
            # initialize q-value with heuristic for current state
            self.qs[a] = h(self.s[0])
            self.n_as[a] = 0

            # get new cumcost for next states
            new_cost = cost_fn(self.s[0], a)

            # initialize each child's subtree
            self.children[a] = {}
            for s_ in succ:
                self.children[a][(s_, new_cost)] = new_tree(
                    (s_, new_cost), self.depth + 1, actions)

    def traverse(self, fn):
        fn(self)

        if not self.children:
            return

        for child_dict in self.children.values():
            for child in child_dict.values():
                child.traverse(fn)


def new_tree(s, depth, actions):
    n_as = {}
    qs = {}
    children = {}
    return Tree(0, s, actions, depth, n_as, qs, children)
