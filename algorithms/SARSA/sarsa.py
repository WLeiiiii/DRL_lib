from collections import defaultdict

import dill
import numpy as np
import torch

from algorithms.SARSA.config import SarsaConfig


class SarsaAgent:
    def __init__(self, state_size, action_size):
        """
        Sarsa Agent constructor.

        Parameters:
        - state_size (int): Dimensionality of the state space.
        - action_size (int): Number of possible actions.
        """
        self.state_dim = state_size
        self.action_dim = action_size
        self.cfgs = SarsaConfig()
        # Initialize the Q-table, mapping each state to action values
        self.Q_table = defaultdict(lambda: np.zeros(action_size))

    def act(self, state, epsilon=0.):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        q_predict = self.Q_table[str(state)][action]
        q_target = reward + self.cfgs.gamma * self.Q_table[str(next_state)][next_action] * (1 - done)
        self.Q_table[str(state)][action] += self.cfgs.learning_rate * (q_target - q_predict)

    def save(self, path):
        """
        Save the Q-table to a file.
        """
        torch.save(
            obj=self.Q_table,
            f=path + "Sarsa_model.pkl",
            pickle_module=dill
        )
        print("Sarsa模型已保存！")

    def load(self, path):
        """
        Load the Q-table from a file.
        """
        self.Q_table = torch.load(f=path + "Sarsa_model.pkl", pickle_module=dill)
        print("Sarsa模型成功加载！")
