from pddlgym.structs import State

from uct_gubs import pddl


def tireworld_get_location(obs):
    obs = obs if type(obs) == State else pddl.from_literals(obs)
    for lit in obs.literals:
        if lit.predicate.name == 'vehicle-at':
            return lit.variables[0]
    return []


def tireworld_text_render(obs):
    obs = obs if type(obs) == State else pddl.from_literals(obs)
    vehicle_location = tireworld_get_location(obs)
    flattire = True
    spare_in_locs = []
    for lit in obs.literals:
        if lit.predicate.name == 'not-flattire':
            flattire = False
        elif lit.predicate.name == 'spare-in':
            spare_in_locs.append(lit.variables[0])
    return f"""
        Vehicle at {vehicle_location}
        Spare tires at {spare_in_locs}
        {"Flat tire" if flattire else ""}
    """


def navigation_get_locations(obs):
    return pddl.get_values_of_literal_by_name(obs.literals, 'robot-at')


def navigation_text_render(obs):
    obs = obs if type(obs) == State else pddl.from_literals(obs)
    qualifiers = []
    locations = navigation_get_locations(obs)

    location = "deadend" if locations == [] else locations[0][0]
    for lit in obs.literals:
        if (lit.predicate.name != 'robot-at' and lit.predicate.name != 'conn'
                and lit.variables[0] == location):
            qualifiers.append(lit.predicate.name)
    return f"""
        Robot at {location}
        {f"Qualifiers: {qualifiers}" if len(qualifiers) > 0 else ""}
    """


def expblocks_text_render(obs):
    clear = []
    ontable = []
    on = []
    holding = None
    destroyed_blocks = []
    table_destroyed = None
    for lit in obs.literals:
        if lit.predicate.name == 'clear':
            clear.append(lit.variables[0])
        elif lit.predicate.name == 'on':
            on.append(lit.variables[:2])
        elif lit.predicate.name == 'ontable':
            ontable.append(lit.variables[0])
        elif lit.predicate.name == 'destroyed':
            destroyed_blocks.append(lit.variables[0])
        elif lit.predicate.name == 'holding':
            holding = lit.variables[0]
        elif lit.predicate.name == 'table-destroyed':
            table_destroyed = True
    return f"""
        {"Table destroyed" if table_destroyed else ""}
        {f"Holding {holding}" if holding else "Hand empty"}
        Clear blocks at {clear}
        Blocks on table: {ontable}
        Destroyed blocks: {destroyed_blocks}
        {on}
    """


def river_alt_get_location(obs):
    location = None

    for lit in obs.literals:
        if lit.predicate.name == 'robot-at':
            location = lit.variables[1]
            break

    return location


def river_alt_text_render(obs):
    obs = obs if type(obs) == State else pddl.from_literals(obs)
    qualifiers = []
    location = river_alt_get_location(obs)
    for lit in obs.literals:
        if (lit.predicate.name != 'robot-at' and lit.predicate.name != 'conn'
                and lit.variables[0] == location):
            qualifiers.append(lit.predicate.name)
    return f"""
        Robot at {location}
        {f"Qualifiers: {qualifiers}" if len(qualifiers) > 0 else ""}
    """


navigation_is = range(1, 11)
navigation_keys = [f"PDDLEnvNavigation{i}-v0" for i in navigation_is]
text_render_env_functions = {
    "PDDLEnvTireworld-v0": tireworld_text_render,
    "PDDLEnvTireworld_mini-v0": tireworld_text_render,
    "PDDLEnvExplodingblocks-v0": expblocks_text_render,
    "PDDLEnvExplodingblocksTest-v0": expblocks_text_render,
    "PDDLEnvRiver-alt-v0": river_alt_text_render,
    **{k: navigation_text_render
       for k in navigation_keys}
}


def text_render(env, obs):
    if env.spec.id not in text_render_env_functions:
        return ""
    return text_render_env_functions[env.spec.id](obs)
