import logging
import os
from dataclasses import dataclass
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np

from uct_gubs import utils


def run_episode(pi,
                env,
                n_steps=50,
                output_dir=None,
                print_history=False,
                keep_cost=False):
    obs, _ = env.reset()

    _render_and_save = output_dir is not None
    if _render_and_save:
        img = env.render()
        imageio.imsave(os.path.join(output_dir, "frame1.png"), img)

    cum_reward = 0
    for i in range(1, n_steps + 1):
        old_obs = obs
        a = pi(obs.literals) if not keep_cost else pi(obs.literals, i - 1)
        obs, reward, done, _ = env.step(a)
        cum_reward += reward
        if print_history:
            state_text_render = utils.text_render(env, old_obs)
            if state_text_render:
                logging.info(f"State: {state_text_render}")
            logging.info(f" {pi(old_obs)} {reward}")

        if _render_and_save:
            img = env.render()
            imageio.imsave(os.path.join(output_dir, f"frame{i + 1}.png"), img)

        if done:
            break
    return i, cum_reward


def plot_stats(env, pi_func, n_episodes):
    print('running episodes with optimal policy')
    steps1 = []
    rewards1 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(pi_func, env, keep_cost=False)
        steps1.append(n_steps)
        rewards1.append(reward)

    print('running episodes with random policy')
    steps2 = []
    rewards2 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(lambda s: env.action_space.sample(s),
                                      env)
        steps2.append(n_steps)
        rewards2.append(reward)
    rewards2 = np.array(rewards2)

    plt.title('Cumulative reward')
    plt.plot(range(len(rewards1)), np.cumsum(rewards1), label="optimal")
    plt.plot(range(len(rewards1)), np.cumsum(rewards2), label="random")
    plt.legend()

    plt.figure()
    plt.title('Average reward')
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards1) / np.arange(1, n_episodes + 1),
             label="optimal")
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards2) / np.arange(1, n_episodes + 1),
             label="random")
    plt.legend()

    plt.figure()
    plt.title('steps')
    plt.plot(range(len(steps1)), np.cumsum(steps1), label="optimal")
    plt.plot(range(len(steps1)), np.cumsum(steps2), label="random")
    plt.legend()
    plt.show()


@dataclass
class Output:
    cpu_times: list[int]
    tree_sizes: list[int]
    cumcosts: list[float]
    found_goal: list[bool]
    values_s0: list[float]
    best_actions_s0: list[str]
    args: dict

    def output_info(self, output_dir):
        output_filename = str(datetime.time(datetime.now())) + '.json'

        output_file_path = utils.output(output_filename,
                                        vars(self),
                                        output_dir=output_dir)
        if output_file_path:
            logging.info(f"Algorithm result written to {output_file_path}")


def get_and_create_output_dir(env, output_dir):
    output_outdir = output_dir
    domain_name = env.domain.domain_name
    problem_name = domain_name + str(
        env.env._problem_idx) if env.env._problem_index_fixed else None
    output_dir = os.path.join(output_outdir, domain_name, problem_name,
                              f"{str(datetime.now().timestamp())}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    return output_dir
