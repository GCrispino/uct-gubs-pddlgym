import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run experiments for the UCT-GUBS algorithm')

    parser.add_argument('--env', dest='env')
    parser.add_argument(
        '--problem_index',
        dest='problem_index',
    )
    parser.add_argument(
        '--horizon',
        dest='horizon',
    )
    parser.add_argument(
        '--n_rollouts',
        dest='n_rollouts',
    )
    parser.add_argument(
        '--n_rounds',
        dest='n_rounds',
    )
    parser.add_argument('--n_sim_steps', dest='n_sim_steps')
    parser.add_argument(
        '--h_u',
        metavar=f"{{{', '.join(['h1', 'shortest_path'])}}}",
        dest='h_u',
    )
    parser.add_argument(
        '--h_p',
        metavar=f"{{{', '.join(['handcrafted', 'h1'])}}}",
        dest='h_p',
    )
    parser.add_argument(
        '--exploration_constant',
        dest='exploration_constant',
    )
    parser.add_argument(
        '--norm_exploration_constant',
        dest='norm_exploration_constant',
    )
    parser.add_argument(
        '--h_init_count',
        dest='h_init_count',
    )
    parser.add_argument(
        '--k_g',
        dest='k_g',
    )
    parser.add_argument(
        '--lambda',
        dest='lamb',
    )

    return parser.parse_args()
