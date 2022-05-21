import logging
import time

import gym
import matplotlib

import uct_gubs.argparsing as argparsing
import uct_gubs.mdp.general as mdp
import uct_gubs.output as output
import uct_gubs.context as context
from uct_gubs.mdp.types import ExtendedState

# TODO -> optimize code based on profiling
# TODO -> adapt code for doing experiments

matplotlib.use('TkAgg')

args = argparsing.parse_args()

print("logging level:", args.logging_level)
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

logging.info('obtaining optimal policy')
start = time.perf_counter()
u = mdp.risk_exp_fn(args.lamb)

h_u = args.h_u_loader(env, args.lamb)
h_p = args.h_p_loader(env)

ctx = context.ProblemContext(env, obs.literals, problem_index, h_u, h_p,
                             args.h_init_count, u, mdp.build_std_cost_fn(goal),
                             args.exploration_constant, args.k_g,
                             args.n_rollouts, args.horizon)
mdp_tree, pi_func, n_updates = mdp.simulate_with_uct_gubs(
    ctx, ExtendedState(obs.literals, 0), actions, args.n_sim_steps)
final_time = time.perf_counter() - start
print("Final updates:", n_updates)

a_best = pi_func((obs.literals, 0))
logging.info(f"best action at initial state: {a_best}")
logging.info(f"qs: {mdp_tree.qs}")
logging.info(f"value of action {a_best} at initial state" +
             f": {mdp_tree.qs[a_best]}")

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
    output.ouptut_info(obs, final_time, n_updates, output_dir, args, mdp_tree)
