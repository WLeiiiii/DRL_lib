#!/usr/bin/env python
# coding=utf-8
"""
Author: Wang lei
Email: wl120964102@gmail.com
"""
import os
import gymnasium
import torch

from algorithms.DQN.config import DQNConfig
from algorithms.DQN.models import QNetworkCNN
from algorithms.DQN.replaybuffer import ReplayBufferCNN
from algorithms.algo_utils import soft_update


class DQNAgentCNN:
    def __init__(self, state_size, action_size, device):
        self.state_dim = state_size
        self.action_dim = action_size
        self.device = device
        self.cfgs = DQNConfig()  # Configuration parameters for the DQN agent

        # Local network is the Q-Network that will be trained
        self.local_net = QNetworkCNN(self.state_dim, self.action_dim).to(self.device)
        # Target network is used to compute the stable target Q-values
        self.target_net = QNetworkCNN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.cfgs.learning_rate)

        self.replay_buffer = ReplayBufferCNN(state_size, self.cfgs.buffer_size, self.device)

        # Set both networks to training mode
        self.train()
        self.target_net.train()
        self.time_step = 0

    def train(self, training=True):
        # Set the local network to training mode
        self.training = training
        self.local_net.train(training)

    def step(self, logger=None, step=None):
        # Increment time step and check if it's time to update the network
        self.time_step = (self.time_step + 1) % self.cfgs.update_frequency
        if self.time_step == 0:
            # If there are enough experiences in the buffer, perform learning
            if self.replay_buffer.__len__() >= self.cfgs.batch_size:
                self.learn(logger, step)

    def act(self, state, epsilon=0.):
        # Convert the state to a PyTorch tensor with the correct shape
        # Assuming state is a numpy array of shape (H, W, C)
        state = torch.from_numpy(state).float().to(self.device)  # Convert to a torch tensor
        state = state.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        state = state.unsqueeze(0)  # Add a batch dimension (N, C, H, W), where N=1

        q_value = self.local_net(state)
        if torch.rand(1) > epsilon:
            return q_value.argmax().item()
        else:
            return torch.randint(0, self.action_dim, (1,)).item()

    def learn(self, logger=None, step=None):
        # Sample a batch of experiences and perform learning
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfgs.batch_size)

        if self.cfgs.double:
            # Double DQN
            next_actions = self.local_net(next_states).argmax(dim=1, keepdim=True).detach()
            next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        else:
            next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(-1)
        q_target = rewards + self.cfgs.gamma * next_q_values * (1 - dones)
        q = self.local_net(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        soft_update(self.local_net, self.target_net, self.cfgs.tau)

        if logger:
            logger.log("train/batch_reward", rewards.mean(), step)
            logger.log("train/loss", loss, step)

    def save(self, path):
        torch.save(self.local_net.state_dict(), os.path.join(path, "q_cnn_network.pth"))
        pass

    def load(self, path):
        self.local_net.load_state_dict(torch.load(os.path.join(path, "q_cnn_network.pth")))
        self.target_net.load_state_dict(self.local_net.state_dict())
        pass


if __name__ == "__main__":
    env = gymnasium.make("ALE/Pong-v5", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(env.observation_space.shape, env.action_space.n)
    agent = DQNAgentCNN(env.observation_space.shape, env.action_space.n, device)
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
