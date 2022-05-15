import argparse
import logging
import math

DEFAULT_PROB_INDEX = 0
DEFAULT_HORIZON = 20
DEFAULT_NROLLOUTS = 1000
DEFAULT_N_SIM_STEPS = 10
DEFAULT_EXPLORATION_CONSTANT = math.sqrt(2)
DEFAULT_KG = 1
DEFAULT_LAMBDA = -0.1
DEFAULT_LOGGING_LEVEL = logging.INFO
DEFAULT_LOGGING_FILE = None
DEFAULT_SIMULATE = False
DEFAULT_RENDER_AND_SAVE = False
DEFAULT_PRINT_SIM_HISTORY = False
DEFAULT_PLOT_STATS = False
DEFAULT_OUTPUT_DIR = "./output"


# taken from
#   https://stackoverflow.com/questions/35648071/argparse-map-user-input-to-defined-constant
def argconv(**convs):

    def parse_argument(arg):
        print(arg)
        print(f"convs:{convs}")
        if arg in convs:
            print(f"result: {convs[arg]}")
            return convs[arg]
        else:
            msg = "invalid choice: {!r} (choose from {})"
            choices = ", ".join(sorted(
                repr(choice) for choice in convs.keys()))
            raise argparse.ArgumentTypeError(msg.format(arg, choices))

    return parse_argument


def parse_args():
    parser = argparse.ArgumentParser(
        description='Solve PDDLGym environments under the GUBS criterion')

    parser.add_argument('--env',
                        dest='env',
                        required=True,
                        help="PDDLGym environment to solve")
    parser.add_argument(
        '--problem_index',
        type=int,
        default=DEFAULT_PROB_INDEX,
        dest='problem_index',
        help="Chosen environment's problem index to solve (default: %s)" %
        str(DEFAULT_PROB_INDEX))
    parser.add_argument('--horizon',
                        type=int,
                        default=DEFAULT_HORIZON,
                        dest='horizon',
                        help="Maximum horizon for each rollout (default: %s)" %
                        str(DEFAULT_HORIZON))
    parser.add_argument(
        '--n_rollouts',
        type=int,
        default=DEFAULT_NROLLOUTS,
        dest='n_rollouts',
        help="Number of maximum rollouts to run on UCT-GUBS (default: %s)" %
        str(DEFAULT_NROLLOUTS))
    parser.add_argument(
        '--n_sim_steps',
        type=int,
        default=DEFAULT_N_SIM_STEPS,
        dest='n_sim_steps',
        help=("Number of steps to run on simulation" + " (default: %s)") %
        str(DEFAULT_N_SIM_STEPS))
    parser.add_argument(
        '--exploration_constant',
        type=float,
        default=DEFAULT_EXPLORATION_CONSTANT,
        dest='exploration_constant',
        help="Exploration constant used for UCT equation (default: %s)" %
        str(DEFAULT_EXPLORATION_CONSTANT))
    parser.add_argument('--k_g',
                        dest='k_g',
                        type=float,
                        default=DEFAULT_KG,
                        help="Constant goal utility (default: %s)" %
                        str(DEFAULT_KG))
    parser.add_argument('--lambda',
                        dest='lamb',
                        type=float,
                        default=DEFAULT_LAMBDA,
                        help="Risk factor (default: %s)" % str(DEFAULT_LAMBDA))
    parser.add_argument('--logging_level',
                        type=argconv(debug=logging.DEBUG,
                                     info=logging.INFO,
                                     warning=logging.WARNING,
                                     error=logging.ERROR,
                                     critical=logging.CRITICAL),
                        default=DEFAULT_LOGGING_LEVEL,
                        dest='logging_level',
                        help="Logging level (default: %s)" %
                        str(DEFAULT_NROLLOUTS))
    parser.add_argument('--logging_output_file',
                        type=str,
                        default=DEFAULT_LOGGING_FILE,
                        dest='logging_output_file',
                        help='File to output logs, if set')
    parser.add_argument(
        '--simulate',
        dest='simulate',
        default=DEFAULT_SIMULATE,
        action="store_true",
        help=("Defines whether or not to run a simulation in the problem by" +
              " applying the algorithm's resulting policy (default: %s)" %
              DEFAULT_SIMULATE))
    parser.add_argument(
        '--render_and_save',
        dest='render_and_save',
        default=DEFAULT_RENDER_AND_SAVE,
        action="store_true",
        help=("Defines whether or not to render and save the received" +
              " observations during execution to a file (default: %s)" %
              DEFAULT_RENDER_AND_SAVE))
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help="Simulation's output directory (default: %s)" %
                        DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--print_sim_history',
        dest='print_sim_history',
        action="store_true",
        default=DEFAULT_PRINT_SIM_HISTORY,
        help="Defines whether or not to print chosen actions" +
        "during simulation (default: %s)" % DEFAULT_PRINT_SIM_HISTORY)

    parser.add_argument(
        '--plot_stats',
        dest='plot_stats',
        action="store_true",
        default=DEFAULT_PLOT_STATS,
        help="Defines whether or not to run a series of episodes with " +
        "both a random policy and the policy returned by the algorithm and" +
        " plot stats about these runs (default: %s)" % DEFAULT_PLOT_STATS)

    return parser.parse_args()
