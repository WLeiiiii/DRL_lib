import os

import torch
from torch import optim
import torch.nn.functional as F

from algorithms.DDPG.models import Actor, Critic
from algorithms.DDPG.replaybuffer import ReplayBuffer
from algorithms.algo_utils import soft_update


class DDPGAgent:
    learning_rate_actor = 5e-6
    learning_rate_critic = 1e-3
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e6)
    batch_size = 256
    update_frequency = 5

    def __init__(self, state_dim, action_dim, max_action, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

        self.replay_buffer = ReplayBuffer(self.action_dim, self.buffer_size, device)

        self.training = True
        self.train()
        self.actor_target.train()
        self.critic_target.train()

        self.time_step = 0
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        pass

    def step(self, logger=None, step=None):
        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            # If enough samples in memory, learn
            if self.replay_buffer.__len__() >= self.batch_size:
                self.update(logger, step)
        pass

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
        pass

    def update(self, logger=None, step=None):
        states, actions, rewards, next_states, terminates = self.replay_buffer.sample(self.batch_size)

        next_actions = self.actor_target(next_states).detach()
        q_target_next = self.critic_target(next_states, next_actions).detach()
        q_target = rewards + (self.gamma * q_target_next) * (1 - terminates)
        q = self.critic(states, actions)

        critic_loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.critic, self.critic_target, self.tau)
        soft_update(self.actor, self.actor_target, self.tau)

        if logger:
            logger.log("train/batch_reward", rewards.mean(), step)
            logger.log("train/actor_loss", actor_loss, step)
            logger.log("train/critic_loss", critic_loss, step)
        pass

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        pass

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
        pass


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_action = env.action_space.high[0]
    agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], max_action, device)
    observation, info = env.reset(seed=0)
    for _ in range(1000):
        terminated = False
        while not terminated:
            action = agent.act(observation)
            # action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.store(observation, action, reward, next_observation, terminated)
            agent.step()
            observation = next_observation
            if terminated or truncated:
                observation, info = env.reset()
    env.close()
    pass
