from itertools import chain
from collections import deque

import numpy as np
from pddlgym.structs import ProbabilisticEffect, State

from uct_gubs import utils


def shortest_path_from_goal(graph, V_i, s0):
    queue = deque([s0])
    d = {s: 0 for s in graph}
    visited = set([s0])

    while len(queue) > 0:
        s = queue.popleft()
        for s_ in graph[s]:
            if s_ not in graph or s_ in visited:
                continue
            if s_ not in visited:
                queue.append(s_)
                visited.add(s_)
                d[s_] = d[s] + 1

    return d


def tireworld_shortest_path(env):
    problem = env.problems[env.env._problem_idx]
    road_predicates = utils.get_literals_that_start_with(
        problem.initial_state, 'road')

    graph = {}
    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest = road_p.variables[1]

        if dest not in graph:
            graph[dest] = set()

        graph[dest].add(origin)
    V_i = {s: i for i, s in enumerate(graph)}
    goal_location = problem.goal.literals[0].variables[0]
    d = shortest_path_from_goal(graph, V_i, goal_location)

    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest = road_p.variables[1]
        if origin not in graph:
            if origin not in d:
                d[origin] = float('inf')
            d[origin] = min(d[origin], d[dest] + 1)

    return d


def river_shortest_path(env):
    problem = env.problems[env.env._problem_idx]
    conn_predicates = utils.get_literals_that_start_with(
        problem.initial_state, 'conn')

    graph = {}
    for conn_p in conn_predicates:
        origin = conn_p.variables[0]
        dest = conn_p.variables[1]

        if dest not in graph:
            graph[dest] = set()
        graph[dest].add(origin)
    V_i = {s: i for i, s in enumerate(graph)}

    goal_lit = problem.goal.literals[0]
    first_variable = goal_lit.variables[0]

    try:
        second_variable = goal_lit.variables[1]
    except IndexError:
        goal_location = first_variable
    else:
        goal_location = (first_variable if first_variable.var_type
                         == 'location' else second_variable)

    d = shortest_path_from_goal(graph, V_i, goal_location)

    for conn_p in conn_predicates:
        origin = conn_p.variables[0]
        dest = conn_p.variables[1]
        if origin not in graph:
            if origin not in d:
                d[origin] = float('inf')
            d[origin] = min(d[origin], d[dest] + 1)

    return d


def river_data(env):
    shortest_path = river_shortest_path(env)
    ny = river_get_ny(env)

    return shortest_path, ny


def tireworld_h_p_data(env):
    problem = env.problems[env.env._problem_idx]
    road_predicates = utils.get_literals_that_start_with(
        problem.initial_state, 'road')

    graph = {}
    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest = road_p.variables[1]

        if origin not in graph:
            graph[origin] = set()
        graph[origin].add(dest)

    return graph


def tireworld_h_p(env, obs, graph):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    has_flattire = not ('not-flattire()' in set(map(str, obs.literals)))

    location = utils.get_values(obs.literals, 'vehicle-at')[0][0]

    spares = set(chain(*utils.get_values(obs.literals, 'spare-in')))

    if has_flattire and location not in spares:
        return 0

    no_spare_in_succ = True
    for succs in graph[location]:
        for succ in succs:
            if succ in spares:
                no_spare_in_succ = False
    if no_spare_in_succ:
        return 0.5

    return 1


def tireworld_h_v(env, obs, lamb, shortest_path):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    has_flattire = not ('not-flattire()' in set(map(str, obs.literals)))

    location = utils.get_values(obs.literals, 'vehicle-at')[0][0]

    spares = set(chain(*utils.get_values(obs.literals, 'spare-in')))

    if has_flattire and location not in spares:
        return 0
    else:
        sp = shortest_path[location]

        return np.exp(lamb * sp)


def river_h_v(env, obs, lamb, data):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    shortest_path, _ = data
    waterfall_locs = set(
        chain(*utils.get_values(obs.literals, 'is-waterfall')))

    location = utils.get_values(obs.literals, 'robot-at')[0][1]

    if location in waterfall_locs:
        return 0

    sp = shortest_path[location]
    return np.exp(lamb * sp)


def river_get_ny(env):
    problem = env.problems[env.env._problem_idx]
    loc_objs = utils.get_objects_by_name(problem.objects, 'location')
    ny = max(map(lambda x: int(x[1:-1].split('-')[1]), loc_objs)) + 1
    return ny


def river_h_p(env, obs, data):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    _, ny = data
    waterfall_locs = set(
        chain(*utils.get_values(obs.literals, 'is-waterfall')))
    river_locs = set(chain(*utils.get_values(obs.literals, 'is-river')))

    location = utils.get_values(obs.literals, 'robot-at')[0][1]
    y_coord = int(location[1:-1].split('-')[1])

    if location in waterfall_locs:
        return 0
    if location not in river_locs:
        return 1

    p = 1 - (0.4**(ny - (y_coord + 1)))
    return p


def expblocks_h_v(env, obs, lamb, data):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    goal_lits = set(obs.goal.literals if type(obs.goal) == State else obs.goal)

    h_empty_lits = utils.get_literals_by_name(obs.literals, 'handempty')
    on_lits = utils.get_literals_by_name(obs.literals, 'on')
    ontable_lits = utils.get_literals_by_name(obs.literals, 'ontable')
    clear_lits = utils.get_literals_by_name(obs.literals, 'clear')
    all_lits = set({*h_empty_lits, *on_lits, *ontable_lits, *clear_lits})

    n_not_matched_goal_literals = len(
        set((lit for lit in goal_lits if lit not in all_lits)))

    return np.exp(n_not_matched_goal_literals * lamb)


def navigation_data(env):
    shortest_path = river_shortest_path(env)
    ny = river_get_ny(env)

    return shortest_path, ny


def navigation_h_v(env, obs, lamb, data):
    obs = obs if type(obs) == State else utils.from_literals(obs)
    shortest_path, _ = data

    locations = utils.navigation_get_locations(obs)
    if locations == []:
        return 0

    sp = shortest_path[locations[0][0]]
    return np.exp(lamb * sp)


def navigation_col_prob(env, col):
    operator = env.domain.operators[f'move-robot-col-{col}']
    prob_effect = operator.effects.literals[1]
    assert isinstance(prob_effect, ProbabilisticEffect)

    return prob_effect.probabilities[0]


def navigation_h_p(env, obs, data):
    obs = obs if type(obs) == State else utils.from_literals(obs)

    prob_locs = set(chain(*utils.get_values(obs.literals, 'is-prob')))

    locations = utils.navigation_get_locations(obs)
    if locations == []:
        return 0

    location = locations[0][0]
    x_coord = int(location[1:-1].split('-')[0])

    if location not in prob_locs:
        return 1

    return navigation_col_prob(env, x_coord)


navigation_is = range(1, 5)
navigation_keys = [f"PDDLEnvNavigation{i}-v0" for i in navigation_is]

value_heuristic_data_functions = {
    "PDDLEnvTireworld-v0": tireworld_shortest_path,
    "PDDLEnvRiver-alt-v0": river_data,
    **{k: navigation_data
       for k in navigation_keys}
}

prob_heuristic_data_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_p_data,
    "PDDLEnvRiver-alt-v0": river_data,
    **{k: navigation_data
       for k in navigation_keys}
}

value_heuristic_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_v,
    "PDDLEnvRiver-alt-v0": river_h_v,
    "PDDLEnvExplodingblocks-v0": expblocks_h_v,
    **{k: navigation_h_v
       for k in navigation_keys}
}

prob_heuristic_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_p,
    "PDDLEnvRiver-alt-v0": river_h_p,
    **{k: navigation_h_p
       for k in navigation_keys}
}


def build_hv(env, lamb):
    data = (None if env.spec.id not in value_heuristic_data_functions else
            value_heuristic_data_functions[env.spec.id](env))

    def h_v(obs):
        if env.spec.id not in value_heuristic_functions:
            return 1
        return value_heuristic_functions[env.spec.id](env, obs, lamb, data)

    return h_v


def build_hp(env):
    data = (None if env.spec.id not in prob_heuristic_data_functions else
            prob_heuristic_data_functions[env.spec.id](env))

    def h_p(obs):
        if env.spec.id not in prob_heuristic_functions:
            return 1
        return prob_heuristic_functions[env.spec.id](env, obs, data)

    return h_p
