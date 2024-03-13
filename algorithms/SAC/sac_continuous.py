import numpy as np
import torch

import torch.nn.functional as F

from algorithms.SAC.models import PolicyNetContinuous, QValueNetContinuous
from algorithms.SAC.replaybuffer import ReplayBuffer
from algorithms.algo_utils import soft_update


class SACContinuousAgent:
    learning_rate_actor = 3e-4
    learning_rate_critic = 3e-3
    learning_rate_alpha = 3e-4
    gamma = 0.99
    tau = 5e-3
    buffer_size = int(1e6)
    batch_size = 256
    update_frequency = 5

    def __init__(self, state_dim, action_dim, action_bound, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.target_entropy = -self.action_dim
        print("State dim: ", self.state_dim)
        print("Action dim: ", self.action_dim)

        self.actor = PolicyNetContinuous(self.state_dim, self.action_dim, self.action_bound).to(device)
        self.critic_1 = QValueNetContinuous(self.state_dim, self.action_dim).to(device)
        self.critic_2 = QValueNetContinuous(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(self.state_dim, self.action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(self.state_dim, self.action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.learning_rate_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.learning_rate_critic)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate_alpha)

        self.replay_buffer = ReplayBuffer(self.action_dim, self.buffer_size, device)

        self.training = True
        self.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

        self.time_step = 0
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic_1.train(training)
        self.critic_2.train(training)
        pass

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action, _ = self.actor(state)
        return action.detach().cpu().numpy()

    def step(self, logger=None, step=None):
        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            if self.replay_buffer.__len__() >= self.batch_size:
                self.update(logger, step)
        pass

    def update(self, logger=None, step=None):
        states, actions, rewards, next_states, terminates = self.replay_buffer.sample(self.batch_size)

        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob

        next_q1_value = self.target_critic_1(next_states, next_actions)
        next_q2_value = self.target_critic_2(next_states, next_actions)

        next_value = torch.min(next_q1_value, next_q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - terminates)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(q1, td_target.detach())
        critic_2_loss = F.mse_loss(q2, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        new_q1_value = self.critic_1(states, new_actions)
        new_q2_value = self.critic_2(states, new_actions)

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(new_q1_value, new_q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        soft_update(self.critic_1, self.target_critic_1, self.tau)
        soft_update(self.critic_2, self.target_critic_2, self.tau)
        print("update!")
        pass


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("Pendulum-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACContinuousAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.high[0],
        device
    )
    observation, info = env.reset(seed=0)
    for _ in range(1000):
        terminated = False
        while not terminated:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.store(observation, action, reward, next_observation, terminated)
            agent.step()
            observation = next_observation
            if terminated or truncated:
                observation, info = env.reset()
    env.close()
    pass
