import gym
import numpy as np


def get_env_info(env_name, problem_index=0):
    tireworld_env = gym.make(env_name)
    tireworld_env.fix_problem_index(problem_index)
    problem = tireworld_env.problems[problem_index]

    tireworld_s0, _ = tireworld_env.reset()
    tireworld_actions = np.array(
        list(
            sorted(
                tireworld_env.action_space.all_ground_literals(
                    tireworld_s0, valid_only=False))))
    tireworld_goal = problem.goal

    return (tireworld_env, problem, tireworld_s0, tireworld_actions,
            tireworld_goal)
