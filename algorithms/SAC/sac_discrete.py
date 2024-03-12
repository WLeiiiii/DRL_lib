import os

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.SAC.models import PolicyNet, ValueNet
from algorithms.SAC.replaybuffer import ReplayBuffer
from algorithms.algo_utils import soft_update


class SACAgent:
    learning_rate_actor = 5e-6
    learning_rate_critic = 1e-3
    learning_rate_alpha = 1e-4
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e6)
    batch_size = 256
    update_frequency = 5
    target_entropy = -2

    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim[0]
        self.action_dim = action_dim
        self.device = device
        print("State dim: ", self.state_dim)
        print("Action dim: ", self.action_dim)

        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic_1 = ValueNet(self.state_dim, self.action_dim).to(device)
        self.critic_2 = ValueNet(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = ValueNet(self.state_dim, self.action_dim).to(device)
        self.target_critic_2 = ValueNet(self.state_dim, self.action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.learning_rate_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.learning_rate_critic)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
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
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def step(self, logger=None, step=None):
        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            # If enough samples in memory, learn
            if self.replay_buffer.__len__() >= self.batch_size:
                self.update(logger, step)
        pass

    def update(self, logger=None, step=None):
        states, actions, rewards, next_states, terminates = self.replay_buffer.sample(self.batch_size)

        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)

        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)

        min_q_value = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_q_value + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - terminates)

        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        # 梯度反传
        critic_1_loss.backward()
        critic_2_loss.backward()
        # 梯度更新
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_q_value = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        soft_update(self.critic_1, self.target_critic_1, self.tau)
        soft_update(self.critic_2, self.target_critic_2, self.tau)
        pass

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(path, "critic_1.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(path, "critic_2.pth"))
        pass

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic_1.load_state_dict(torch.load(os.path.join(path, "critic_1.pth")))
        self.critic_2.load_state_dict(torch.load(os.path.join(path, "critic_2.pth")))
        pass


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env.observation_space.shape, env.action_space.n, device)
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
