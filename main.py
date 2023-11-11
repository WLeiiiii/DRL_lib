import datetime

import gymnasium as gym

import os

import torch

from utils.agents import Agents
from utils.eval_mode import eval_mode
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
        state_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n
        print("State dimension: {}\nAction dimension: {}".format(state_dim, action_dim))

        self.agent = Agents.get_agent(self.args.algo, state_size=state_dim, action_size=action_dim,
                                      device=self.args.device)
        pass

    def run(self):
        observation, info = self.env.reset(seed=self.args.seed)
        for _ in range(1000):
            terminated = False
            while not terminated:
                with eval_mode(self.agent):
                    action = self.agent.act(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.agent.replay_buffer.store(observation, action, reward, next_observation, terminated)
                self.agent.step()
                observation = next_observation
                if terminated or truncated:
                    observation, info = self.env.reset()
        self.env.close()
        pass

    def eval(self):
        pass

    def test(self):
        pass


def main():
    import argparse
    work_dir = os.getcwd()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--curr_time', type=str, default=current_time)
    parser.add_argument('--device', type=str, default=device)
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
