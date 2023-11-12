#!/usr/bin/env python
# coding=utf-8
"""
Author: Wang lei
Email: wl120964102@gmail.com
"""
import os

import numpy as np
import torch

from algorithms.DQN.config import DQNConfig
from algorithms.DQN.models import DuelQNetworkMLP, QNetworkMLP
from algorithms.DQN.replaybuffer import PrioritizedReplayBuffer
from algorithms.algo_utils import soft_update


class DQNAgentPER:
    def __init__(self, state_size, action_size, device):
        # Initialization of the DQN Agent with prioritized experience replay
        self.state_dim = state_size[0]  # The dimension of the state space
        self.action_dim = action_size  # The dimension of the action space
        self.device = device  # Device (CPU or GPU) to run the computations on
        self.cfgs = DQNConfig()  # Configuration parameters for the DQN agent
        self.cfgs.prior = True
        self.cfgs.__information__()

        # Initializing the networks; dueling architecture if specified in configs
        if self.cfgs.dueling:
            # Dueling DQN uses separate streams for state value and advantage
            self.local_net = DuelQNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
            self.target_net = DuelQNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
        else:
            # Standard DQN uses a single Q-network
            self.local_net = QNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
            self.target_net = QNetworkMLP(self.state_dim, self.action_dim, self.cfgs.hidden_dim).to(self.device)
        # Initializing the target network with the same weights as the local network
        self.target_net.load_state_dict(self.local_net.state_dict())
        # Optimizer for training the local network
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.cfgs.learning_rate)

        # The replay buffer using prioritized experience replay
        self.replay_buffer = PrioritizedReplayBuffer(self.cfgs.buffer_size)

        # Setting both networks to training mode
        self.train()
        self.target_net.train()
        self.time_step = 0  # Counter to keep track of when to update the network

    def train(self, training=True):
        # Set the local network to training or evaluation mode
        self.training = training
        self.local_net.train(training)

    def step(self, logger=None, step=None):
        # Method called at each step of the training
        self.time_step = (self.time_step + 1) % self.cfgs.update_frequency
        if self.time_step == 0:
            # If enough samples are in the replay buffer, perform a learning step
            if self.replay_buffer.__len__() >= self.cfgs.batch_size:
                self.learn(logger, step)

    def act(self, state, epsilon=0.):
        # Select an action according to epsilon-greedy policy
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        q_value = self.local_net(state)
        # Epsilon-greedy action selection
        if torch.rand(1) > epsilon:
            return q_value.argmax().item()  # Exploit
        else:
            return torch.randint(0, self.action_dim, (1,)).item()  # Explore

    def learn(self, logger=None, step=None):
        # Sample a batch from the replay buffer and perform learning
        b_idx, b_memory, ISWeights = self.replay_buffer.sample(self.cfgs.batch_size)
        # Unpack the experiences
        states, actions, rewards, next_states, dones = zip(*b_memory)
        # Convert to torch tensors
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.stack(actions)).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.stack(dones).astype(np.uint8)).float().unsqueeze(-1).to(self.device)
        ISWeights = torch.from_numpy(np.stack(ISWeights)).float().to(self.device)

        # Compute Q values for the next states
        if self.cfgs.double:
            # Double DQN update
            next_actions = self.local_net(next_states).argmax(dim=1, keepdim=True).detach()
            next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        else:
            # Regular DQN update
            next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute the target Q values
        q_target = rewards + self.cfgs.gamma * next_q_values * (1 - dones)
        # Get the current Q values from the local network
        q = self.local_net(states).gather(1, actions)

        # Calculate temporal difference error and update priorities in the buffer
        td_error = abs(q_target - q).detach().squeeze(1).cpu().numpy()
        self.replay_buffer.update(b_idx, td_error)

        # Compute loss, perform backpropagation and update network weights
        loss = (ISWeights * torch.nn.functional.mse_loss(q, q_target, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network towards the local network
        soft_update(self.local_net, self.target_net, self.cfgs.tau)

        # Log training information if a logger is provided
        if logger:
            logger.log("train/batch_reward", rewards.mean(), step)
            logger.log("train/loss", loss, step)

    def save(self, path):
        # Save the local network model to the specified path
        torch.save(self.local_net.state_dict(), os.path.join(path, "q_per_network.pth"))

    def load(self, path):
        # Load the local network model from the specified path
        self.local_net.load_state_dict(torch.load(os.path.join(path, "q_per_network.pth")))
        self.target_net.load_state_dict(self.local_net.state_dict())


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgentPER(env.observation_space.shape, env.action_space.n, device)
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
