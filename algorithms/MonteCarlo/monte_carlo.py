from collections import defaultdict

import numpy as np

from algorithms.MonteCarlo.config import MonteCarloConfig


class MonteCarloAgent:
    def __init__(self, state_size, action_size):
        """
        Monte Carlo Agent constructor.
        :param state_size:
        :param action_size:
        """
        self.state_dim = state_size
        self.action_dim = action_size
        self.cfgs = MonteCarloConfig()
        self.Q_table = defaultdict(lambda: np.zeros(action_size))
        self.returns = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.returns_sum = defaultdict(float)

    def act(self, state, epsilon=0.):
        """
        Select an action for the given state using an epsilon-greedy policy.
        :param state:
        :param epsilon:
        :return:
        """
        state = str(state)
        if state in self.Q_table.keys():
            best_action = np.argmax(self.Q_table[state])
            action_probs = np.ones(self.action_dim, dtype=float) * epsilon / self.action_dim
            action_probs[best_action] += (1.0 - epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def update(self, one_episode_transitions):
        """
        Update the Q-table for a given state and action.
        :param one_episode_transitions:
        :return:
        """
        states, actions, rewards = zip(*one_episode_transitions)
        discounts = np.array([self.cfgs.gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            self.returns_sum[state] += sum(rewards[i:] * discounts[:-(1 + i)])
            self.returns_count[state] += 1.0
            self.returns[state] = self.returns_sum[state] / self.returns_count[state]
            self.Q_table[state][actions[i]] = self.returns[state]

    def save(self, path):
        """
        Save the Q-table to a file.
        """
        import dill
        import torch
        torch.save(
            obj=self.Q_table,
            f=path + "MonteCarlo_model.pkl",
            pickle_module=dill
        )
        print("QLearning模型已保存！")

    def load(self, path):
        """
        Load the Q-table from a file.
        """
        import dill
        import torch
        self.Q_table = torch.load(f=path + "MonteCarlo_model.pkl", pickle_module=dill)
        print("Qlearning模型成功加载！")


if __name__ == '__main__':
    import gymnasium

    env = gymnasium.make("CliffWalking-v0", render_mode="human")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = MonteCarloAgent(state_dim, action_dim)
    observation, info = env.reset(seed=0)
    for episode in range(1000):
        terminated = False
        one_ep_transitions = []
        while not terminated:
            action = agent.act(observation, epsilon=0.1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            one_ep_transitions.append((observation, action, reward))
            agent.update(one_ep_transitions)
            observation = next_observation
            if terminated or truncated:
                observation = env.reset()
    env.close()
