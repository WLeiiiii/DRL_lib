#!/usr/bin/env python
# coding=utf-8
"""
Author: Wang lei
Email: wl120964102@gmail.com
"""
import os

import torch

from algorithms.DQN.config import DQNConfig
from algorithms.DQN.models import QNetworkMLP
from algorithms.DQN.replaybuffer import ReplayBuffer
from algorithms.algo_utils import soft_update


class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim[0]
        self.action_dim = action_dim
        self.device = device
        self.cfgs = DQNConfig()

        self.local_net = QNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
        self.target_net = QNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.cfgs.learning_rate)

        self.replay_buffer = ReplayBuffer(state_dim, self.action_dim, self.cfgs.buffer_size, self.device)

        self.train()
        self.target_net.train()
        self.time_step = 0
        pass

    def train(self, training=True):
        self.training = training
        self.local_net.train(training)
        pass

    def step(self, logger=None, step=None):
        self.time_step = (self.time_step + 1) % self.cfgs.update_frequency
        if self.time_step == 0:
            # If enough samples in memory, learn
            if self.replay_buffer.__len__() >= self.cfgs.batch_size:
                self.learn(logger, step)
        pass

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        q_value = self.local_net(state)
        if torch.rand(1) > epsilon:
            return q_value.argmax().item()
        else:
            return torch.randint(0, self.action_dim, (1,)).item()

    def learn(self, logger=None, step=None):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfgs.batch_size)
        print(actions.shape)
        if self.cfgs.double:
            next_actions = self.local_net(next_states).argmax(dim=1, keepdim=True).detach()
            next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        else:
            next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        q_target = rewards + self.cfgs.gamma * next_q_values * (1 - dones)
        q = self.local_net(states).gather(1, actions)
        print(q.shape, q_target.shape)

        loss = torch.nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.local_net, self.target_net, self.cfgs.tau)

        if logger:
            logger.log("train/batch_reward", rewards.mean(), step)
            logger.log("train/loss", loss, step)

    def save(self, path):
        torch.save(self.local_net.state_dict(), os.path.join(path, "q_network.pth"))
        pass

    def load(self, path):
        self.local_net.load_state_dict(torch.load(os.path.join(path, "q_network.pth")))
        self.target_net.load_state_dict(self.local_net.state_dict())
        pass
