import uct_gubs.context as context
import uct_gubs.heuristics as heuristics
import uct_gubs.mdp.general as mdp
import uct_gubs.tree as tree
from uct_gubs.mdp.types import ExtendedState

from test.utils import get_env_info

(tireworld_env, tireworld_problem, tireworld_s0, tireworld_actions,
 tireworld_goal) = get_env_info("PDDLEnvTireworld-v0")


def ctx():
    lamb = -0.1
    k_g = 1
    n_rollouts = 0
    horizon = 10
    ctx = context.ProblemContext(tireworld_env, tireworld_s0, 0,
                                 heuristics.h_1, heuristics.h_1,
                                 mdp.select_first_criterion, 0,
                                 mdp.risk_exp_fn(lamb),
                                 mdp.build_std_cost_fn(tireworld_goal),
                                 mdp.SQRT_TWO, False, k_g, n_rollouts, horizon)
    return ctx


def tireworld_mdp_tree(ctx):
    mdp_tree = tree.new_tree(ExtendedState(tireworld_s0.literals, 0), 0,
                             tireworld_actions)
    return mdp_tree


def tireworld_mdp_tree_cost_1(ctx):
    mdp_tree = tree.new_tree((tireworld_s0.literals, 1), 0, tireworld_actions)
    return mdp_tree
