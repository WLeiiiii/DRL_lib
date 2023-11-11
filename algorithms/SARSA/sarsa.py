from collections import defaultdict

import numpy as np

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
        """
        Choose an action based on the current state and policy.

        Parameters:
        - state: The current state of the environment.
        - epsilon (float): The probability of choosing a random action for epsilon-greedy policy.

        Returns:
        - action (int): The action chosen to perform.
        """
        # Best action based on current policy
        best_action = np.argmax(self.Q_table[state])
        # Probability distribution for action selection
        action_probs = np.ones(self.action_dim, dtype=float) * epsilon / self.action_dim
        # Increment the probability of the best action by (1 - epsilon)
        action_probs[best_action] += (1.0 - epsilon)
        # Choose an action according to the probability distribution
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table based on agent experience.

        Parameters:
        - state: The current state from which the action was taken.
        - action: The action taken in the current state.
        - reward: The reward received after taking the action.
        - next_state: The next state reached after taking the action.
        - next_action: The next action the agent is planning to take.
        - done (bool): A flag indicating if the episode has ended.
        """
        # Predicted Q-value for the current state-action pair
        q_predict = self.Q_table[str(state)][action]
        # Target Q-value for the current state-action pair
        q_target = reward + self.cfgs.gamma * self.Q_table[str(next_state)][next_action] * (1 - done)
        # Update the Q-value for the current state-action pair
        self.Q_table[str(state)][action] += self.cfgs.learning_rate * (q_target - q_predict)

    def save(self, path):
        """
        Save the Q-table to a file.
        """
        import dill
        import torch
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
        import dill
        import torch
        self.Q_table = torch.load(f=path + "Sarsa_model.pkl", pickle_module=dill)
        print("Sarsa模型成功加载！")


if __name__ == '__main__':
    import gymnasium

    env = gymnasium.make("CliffWalking-v0", render_mode="human")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = SarsaAgent(state_dim, action_dim)
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
            next_action = agent.act(next_observation, epsilon)
            agent.update(observation, action, reward, next_observation, next_action, terminated)
            observation = next_observation
            if terminated or truncated:
                observation = env.reset()
        epsilon = max(epsilon * eps_decay, eps_end)
    env.close()
