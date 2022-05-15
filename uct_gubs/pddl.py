import pddlgym.core as pddlcore
from pddlgym.structs import Literal, Predicate, State, Type


def from_literals(literals):
    empty_set = frozenset()
    return State(literals, empty_set, empty_set)


def get_values_of_literal_by_name(obs, name):
    return [lit.variables for lit in get_literals_by_name(obs, name)]


def get_literals_by_name(s, name):
    return frozenset((lit for lit in s if lit.predicate.name == name))


def create_literal(pred_name: str,
                   arity: int = 0,
                   type_strs: list[str] = None,
                   variables: list[str] = None) -> Literal:
    type_strs = [] if type_strs is None else type_strs
    variables = [] if variables is None else variables
    return Literal(
        Predicate(pred_name, arity,
                  [Type(type_str) for type_str in type_strs]), variables)


def get_valid_actions_and_successors(
        s: frozenset[Literal], actions,
        env) -> dict[Literal, tuple[frozenset[Literal], float]]:
    valid_actions_successors = {}
    for a in actions:
        successors = None
        try:
            successors = frozenset({
                s_.literals
                for s_ in pddlcore.get_successor_states(
                    from_literals(s),
                    a,
                    env.domain,
                    raise_error_on_invalid_action=True)
            })
        except pddlcore.InvalidAction:
            continue

        assert successors is not None
        valid_actions_successors[a] = successors

    return valid_actions_successors
