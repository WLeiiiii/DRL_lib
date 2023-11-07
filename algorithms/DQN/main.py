import argparse
import datetime
import os
import random

import numpy as np
import gymnasium as gym
import torch

from utils.set_seed import SetSeed


class Workspace:
    def __init__(self, args):
        self.args = args
        print(
            "Time:\t{}\nWorkspace:\t{}\nDevice:\t{}\nAlgorithm:\t{}".format(
                self.args.curr_time, self.args.work_dir, self.args.device, self.args.algo
            )
        )
        self.env = gym.make(self.args.env, render_mode='human')
        pass

    def run(self):
        observation, info = self.env.reset(seed=self.args.seed)
        for _ in range(1000):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                observation, info = self.env.reset()
        self.env.close()
        pass

    def eval(self):
        pass

    def test(self):
        pass


def main():
    work_dir = os.getcwd()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--curr_time', type=str, default=current_time)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--work_dir', type=str, default=work_dir)
    parser.add_argument('--algo', type=str, default='DQN')
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--episodes', '-e', type=int, default=10000)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=9)
    args = parser.parse_args()

    with SetSeed(args.seed):
        workspace = Workspace(args)
        workspace.run()
    pass


if __name__ == '__main__':
    main()
