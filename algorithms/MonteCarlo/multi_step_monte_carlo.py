import numpy as np

from algorithms.MonteCarlo.config import MonteCarloConfig
from algorithms.MonteCarlo.single_step_monte_carlo import MonteCarloAgent


class MultiMonteCarloAgent(MonteCarloAgent):
    def __init__(self, state_size, action_size):
        """
        Multi-step Monte Carlo Agent constructor.
        :param state_size:
        :param action_size:
        """
        super().__init__(state_size, action_size)

    def update(self, one_episode_transitions, n_steps=3):
        """
        Update the Q-table for a given state and action.
        :param one_episode_transitions:
        :return:
        """
        states, actions, rewards = zip(*one_episode_transitions)
        T = len(states)
        discounts = np.array([self.cfgs.gamma ** i for i in range(len(rewards) + 1)])
        for t in range(T):
            # Calculate the n-step return. If there are fewer than n steps remaining, use what's available.
            tau = min(t + n_steps, T)
            G = sum(rewards[t:tau] * discounts[:tau - t])
            if tau < T:
                G += (self.cfgs.gamma ** n_steps) * self.Q_table[str(states[tau])][actions[tau]]

            # Update the Q-table
            state = str(states[t])
            self.returns_sum[(state, actions[t])] += G
            self.returns_count[(state, actions[t])] += 1.0
            self.returns[(state, actions[t])] = self.returns_sum[(state, actions[t])] / self.returns_count[
                (state, actions[t])]
            self.Q_table[state][actions[t]] = self.returns[(state, actions[t])]


if __name__ == '__main__':
    import gymnasium

    env = gymnasium.make("CliffWalking-v0", render_mode="human")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = MultiMonteCarloAgent(state_dim, action_dim)
    observation, info = env.reset(seed=0)
    for episode in range(1000):
        terminated = False
        one_ep_transitions = []
        while not terminated:
            action = agent.act(observation, epsilon=0.1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            one_ep_transitions.append((observation, action, reward))
            agent.update(one_ep_transitions, n_steps=3)
            observation = next_observation
            if terminated or truncated:
                observation = env.reset()
    env.close()
