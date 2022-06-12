import logging
import time

import gym
import matplotlib
import numpy as np
from pddlgym.structs import Literal

import uct_gubs.argparsing.plan as argparsing
import uct_gubs.mdp.general as mdp
import uct_gubs.output as output
import uct_gubs.context as context
from uct_gubs.mdp.types import ExtendedState
from uct_gubs.utils import nparr_to_list

# TODO -> set seeds to make runs deterministic

matplotlib.use('TkAgg')

args = argparsing.parse_args()

logging.basicConfig(level=args.logging_level,
                    filename=args.logging_output_file)

env = gym.make(args.env)
problem_index = args.problem_index
env.fix_problem_index(problem_index)
problem = env.problems[problem_index]
goal = problem.goal
prob_objects = frozenset(problem.objects)

obs, _ = env.reset()

actions = frozenset(env.action_space.all_ground_literals(obs,
                                                         valid_only=False))
n_updates = None
C_max = None
keep_cost = False

u = mdp.risk_exp_fn(args.lamb)

h_u = args.h_u_loader[1](env, args.lamb)
h_p = args.h_p_loader[1](env)

ctx = context.ProblemContext(env, obs.literals, problem_index, h_u, h_p,
                             args.h_init_count, u, mdp.build_std_cost_fn(goal),
                             args.exploration_constant,
                             args.norm_exploration_constant, args.k_g,
                             args.n_rollouts, args.horizon)

found_goal_results = np.zeros(args.n_rounds, dtype=bool)
cumcost_results = np.zeros(args.n_rounds)
time_results = np.zeros(args.n_rounds, dtype=float)
n_updates_results = np.zeros(args.n_rounds, dtype=int)
tree_size_results = np.zeros(args.n_rounds, dtype=int)
values_s0 = np.zeros(args.n_rounds)
best_actions_s0 = [""] * args.n_rounds
actions_initial_state_results: dict[Literal, list] = {}
for i in range(args.n_rounds):
    logging.info(f'computing policy for round {i}')
    start = time.perf_counter()
    (mdp_tree, pi_func, found_goal, cumcost, action_initial_state, final_time,
     n_updates) = mdp.run_round(ctx, ExtendedState(obs.literals, 0), actions,
                                args.n_sim_steps)
    found_goal_results[i] = found_goal
    cumcost_results[i] = cumcost
    time_results[i] = final_time
    n_updates_results[i] = n_updates

    tree_size = mdp_tree.size()
    tree_size_results[i] = tree_size
    if action_initial_state not in actions_initial_state_results:
        actions_initial_state_results[action_initial_state] = [0, []]
    actions_initial_state_results[action_initial_state][0] += 1
    actions_initial_state_results[action_initial_state][1].append(
        mdp_tree.qs[action_initial_state])

    # TODO -> get more relevant time as data
    #     - each round time?
    #     - time for planning on first state?
    final_time = time.perf_counter() - start

    a_best = pi_func((obs.literals, 0))
    best_actions_s0[i] = str(a_best)
    values_s0[i] = mdp_tree.qs[a_best]

    logging.info(f"finished round {i} on {final_time} seconds")
    logging.info(f"best action at initial state: {a_best}")
    logging.info(f"qs: {mdp_tree.qs}")
    logging.info(f"value of action {a_best} at initial state" +
                 f": {mdp_tree.qs[a_best]}")
    logging.info(f"number of updates: {n_updates}")
    logging.info(f"final tree size: {tree_size}")

logging.info("finished rounds")
logging.info(f"average probability-to-goal: {found_goal_results.mean()}" +
             f" +- {found_goal_results.std()}")
if found_goal_results.any():
    logging.info(
        f"average cost-to-goal: {cumcost_results[found_goal_results].mean()}" +
        f" +- {cumcost_results[found_goal_results].std()}")
logging.info(f"average round cumcost: {cumcost_results.mean()}" +
             f" +- {cumcost_results.std()}")
logging.info(f"average cpu time: {time_results.mean()}" +
             f" +- {time_results.std()}")
logging.info(f"average number of updates: {n_updates_results.mean()}" +
             f" +- {n_updates_results.std()}")
logging.info(f"average tree size: {tree_size_results.mean()}" +
             f" +- {tree_size_results.std()}")


# compute data for action taken at initial state for each round
logging.info("actions taken at initial state:")

action_initial_state_result_counts = {
    a: res[0]
    for a, res in actions_initial_state_results.items()
}
action_initial_state_result_values = {
    a: f'{np.array(res[1]).mean()} +- {np.array(res[1]).std()}'
    for a, res in actions_initial_state_results.items()
}
logging.info(f"  counts: {action_initial_state_result_counts}")
logging.info(f"  values: {action_initial_state_result_values}")

n_episodes = 500

if args.plot_stats:
    output.plot_stats(env, pi_func, n_episodes)

output_dir = None
# Create folder to save images
if args.render_and_save:
    output_dir = output.get_and_create_output_dir(env, args.output_dir)

if args.simulate:
    _, goal = output.run_episode(pi_func,
                                 env,
                                 n_steps=50,
                                 output_dir=output_dir,
                                 print_history=args.print_sim_history,
                                 keep_cost=keep_cost)

if args.render_and_save:
    serializable_args = {
        **vars(args), "h_u": args.h_u_loader[0],
        "h_p": args.h_p_loader[0]
    }
    del serializable_args["h_u_loader"]
    del serializable_args["h_p_loader"]

    out = output.Output(cpu_times=nparr_to_list(time_results),
                        n_updates=nparr_to_list(n_updates_results),
                        tree_sizes=nparr_to_list(tree_size_results),
                        cumcosts=nparr_to_list(cumcost_results),
                        found_goal=nparr_to_list(found_goal_results),
                        values_s0=nparr_to_list(values_s0),
                        best_actions_s0=best_actions_s0,
                        args=serializable_args)
    out.output_info(output_dir)
