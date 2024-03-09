#!/usr/bin/env python
# coding=utf-8
"""
Author: Wang lei
Email: wl120964102@gmail.com
"""
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.PPO.models import Actor, Critic


class PPOAgent:
    """
    ppo
    """
    learning_rate_actor = 5e-6
    learning_rate_critic = 1e-3
    gamma = 0.99
    buffer_size = 1000
    epoch = 10
    counter = 0

    def __init__(self, state_dim, action_dim, max_action, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic = Critic(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        self.replay_buffer = []

        self.training = True
        self.train()
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        pass

    def step(self, transition, logger=None, step=None):
        self.replay_buffer.append(transition)
        self.counter += 1
        if self.counter % self.buffer_size == 0:
            self.update(logger, step)

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mu, sigma = self.actor(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = torch.clamp(action, self.max_action, -self.max_action)
        return action.cpu().data.numpy().flatten(), action_log_prob.cpu().data.numpy().flatten()
        pass

    def update(self, logger=None, step=None):
        states = torch.tensor(np.array([t.s for t in self.replay_buffer]), dtype=torch.float).to(self.device)
        actions = (torch.tensor(np.array([t.a for t in self.replay_buffer]), dtype=torch.float)
                   .to(self.device).view(-1, 1))
        rewards = (torch.tensor(np.array([t.r for t in self.replay_buffer]), dtype=torch.float)
                   .to(self.device).view(-1, 1))
        next_states = torch.tensor(np.array([t.s_ for t in self.replay_buffer]), dtype=torch.float).to(self.device)
        a_log_ps = torch.tensor(np.array([t.a_log_p for t in self.replay_buffer]), dtype=torch.float).to(
            self.device).view(-1, 1)

        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)

        with torch.no_grad():
            next_states_target = self.critic(next_states)
            td_target = rewards + self.gamma * next_states_target
        td_value = self.critic(states)
        adv = (td_target - td_value).detach()

        for _ in range(self.epoch):
            mu, sigma = self.actor(states)
            dist = torch.distributions.Normal(mu, sigma)
            new_probs = dist.log_prob(actions)
            ratio = torch.exp(new_probs - a_log_ps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv

            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            critic_loss = F.smooth_l1_loss(self.critic(states), td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        self.replay_buffer.clear()
        pass


if __name__ == "__main__":
    import gymnasium

    Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

    env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_action = env.action_space.high[0]
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], max_action, device)
    observation, info = env.reset(seed=0)
    for _ in range(1000):
        terminated = False
        while not terminated:
            assert env.observation_space.contains(observation)
            action, action_log_prob = agent.act(observation)
            assert env.action_space.contains(action)
            # action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.step(Transition(observation, action, action_log_prob, reward, next_observation))
            observation = next_observation
            if terminated or truncated:
                observation, info = env.reset()
    env.close()
    pass
