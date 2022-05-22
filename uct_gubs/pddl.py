import pddlgym.core as pddlcore
from pddlgym.structs import Literal, Predicate, State, Type

from uct_gubs.mdp.types import StateOutcome


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


def get_valid_actions_and_outcomes(
        s: frozenset[Literal], actions,
        env) -> dict[Literal, frozenset[StateOutcome]]:
    valid_actions_successors = {}
    for a in actions:
        successors = None
        try:
            successors = frozenset({
                StateOutcome(prob=prob, literals=s_.literals)
                for s_, prob in pddlcore.get_successor_states(
                    from_literals(s),
                    a,
                    env.domain,
                    return_probs=True,
                    raise_error_on_invalid_action=True).items()
            })
        except pddlcore.InvalidAction:
            continue

        assert successors is not None
        valid_actions_successors[a] = successors

    return valid_actions_successors


def filter_superfluous_actions(
    s: frozenset[Literal], actions_outcomes: dict[Literal,
                                                  frozenset[StateOutcome]]
) -> dict[Literal, frozenset[StateOutcome]]:
    """
        A subset of actions can be ignored from the search if:
          - all actions on it lead to the same state literals,
            hence they can't optimal
          - all actions on it lead to the same
            successor states. Then, one of these actions can
            be arbitrarily considered and the other ones
            can be safely ignored
    """

    # discover actions with equal outcomes
    reversed_outcomes: dict[frozenset[StateOutcome], Literal] = {}
    superfluous_actions = set()
    for a, outcomes in actions_outcomes.items():
        if outcomes in reversed_outcomes:
            # if outcome is equal to one of other action,
            #   then current action can be pruned
            print(f"superfluous! {a}, {reversed_outcomes[outcomes]}")
            superfluous_actions.add(a)
        reversed_outcomes[outcomes] = a

    # actions whose single outcome is the current state
    #   can also be ignored
    cur_state_outcome_set = frozenset({StateOutcome(prob=1, literals=s)})
    superfluous_actions.update(
        set({
            a
            for a in actions_outcomes
            if actions_outcomes[a] == cur_state_outcome_set
        }))

    # reasonable actions are actions that are not superfluous
    reasonable_actions = set(actions_outcomes) - superfluous_actions

    # compute final dict to return
    reasonable_actions_outcomes = {
        a: actions_outcomes[a]
        for a in reasonable_actions
    }

    return reasonable_actions_outcomes
