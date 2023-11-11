from collections import defaultdict

import numpy as np

from algorithms.QLearning.config import QLearningConfig


class QLearningAgent:
    def __init__(self, state_size, action_size):
        """
        Q-Learning Agent constructor.

        Parameters:
        - state_size (int): Dimensionality of the state space.
        - action_size (int): Number of possible actions.
        """
        self.state_dim = state_size
        self.action_dim = action_size
        self.cfgs = QLearningConfig()
        # Initialize the Q-table, mapping each state to action values
        self.Q_table = defaultdict(lambda: np.zeros(action_size))

    def act(self, state, epsilon=0.):
        """
        Select an action for the given state using an epsilon-greedy policy.

        Parameters:
        - state (array-like): The current state representation.
        - epsilon (float): The probability of selecting a random action.

        Returns:
        - action (int): The action chosen.
        """
        # With probability epsilon, select a random action
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            # Otherwise, select the action with the highest value from the Q-table
            action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, terminal):
        """
        Update the Q-table for a given state and action.

        Parameters:
        - state (array-like): The current state.
        - action (int): The action taken.
        - reward (float): The reward received after taking the action.
        - next_state (array-like): The next state.
        - terminal (bool): Whether the next state is terminal.
        """
        # Predicted Q value for the current state and action
        q_predict = self.Q_table[str(state)][action]
        # Q target for current state
        q_target = reward + self.cfgs.gamma * np.max(self.Q_table[str(next_state)]) * (not terminal)
        # Update Q value towards target
        self.Q_table[str(state)][action] += self.cfgs.learning_rate * (q_target - q_predict)

    def save(self, path):
        """
        Save the Q-table to a file.
        """
        import dill
        import torch
        torch.save(
            obj=self.Q_table,
            f=path + "QLearning_model.pkl",
            pickle_module=dill
        )
        print("QLearning模型已保存！")

    def load(self, path):
        """
        Load the Q-table from a file.
        """
        import dill
        import torch
        self.Q_table = torch.load(f=path + "QLearning_model.pkl", pickle_module=dill)
        print("Qlearning模型成功加载！")


if __name__ == '__main__':
    import gymnasium

    env = gymnasium.make("CliffWalking-v0", render_mode="human")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = QLearningAgent(state_dim, action_dim)
    observation, info = env.reset(seed=0)
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.999
    epsilon = eps_start
    for episode in range(1000):
        terminated = False
        while not terminated:
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.update(observation, action, reward, next_observation, terminated)
            observation = next_observation
            if terminated or truncated:
                observation = env.reset()
        epsilon = max(epsilon * eps_decay, eps_end)
    env.close()
