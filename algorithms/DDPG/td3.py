import os

import torch
from torch import optim
import torch.nn.functional as F

from algorithms.DDPG.models import Actor, Critic
from algorithms.DDPG.replaybuffer import ReplayBuffer
from algorithms.algo_utils import soft_update


class TD3Agent:
    learning_rate_actor = 5e-6
    learning_rate_critic = 1e-3
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e6)
    batch_size = 256
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    update_frequency = 5

    def __init__(self, state_dim, action_dim, max_action, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        # Actor Network
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)

        # Critic Networks
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate_critic)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate_critic)

        self.replay_buffer = ReplayBuffer(self.action_dim, self.buffer_size, device)

        self.training = True
        self.train()
        self.actor_target.train()
        self.critic1_target.train()
        self.critic2_target.train()

        self.time_step = 0

    def train(self, training=True):
        self.training = training
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        pass

    # def step(self, logger=None, step=None):
    #     self.time_step = (self.time_step + 1) % self.update_frequency
    #     if self.time_step == 0 and self.replay_buffer.__len__() >= self.batch_size:
    #         self.update(logger, step)

    def step(self, logger=None, step=None):
        self.time_step += 1
        # Check if it's time to update based on update frequency
        if self.time_step % self.update_frequency == 0 and self.replay_buffer.__len__() >= self.batch_size:
            # Perform an update
            self.update(logger, step, update_actor=(self.time_step % (self.update_frequency * self.policy_freq) == 0))

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, logger=None, step=None, update_actor=True):
        states, actions, rewards, next_states, terminates = self.replay_buffer.sample(self.batch_size)

        # Select action according to policy and add clipped noise
        actions = actions.float()
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

        next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1 = self.critic1_target(next_states, next_actions).detach()
        target_Q2 = self.critic2_target(next_states, next_actions).detach()
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - terminates) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        # Compute critic loss
        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if update_actor:
            # Compute actor losse
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic1, self.critic1_target, self.tau)
            soft_update(self.critic2, self.critic2_target, self.tau)
            soft_update(self.actor, self.actor_target, self.tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(path, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(path, "critic2.pth"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic1.load_state_dict(torch.load(os.path.join(path, "critic1.pth")))
        self.critic2.load_state_dict(torch.load(os.path.join(path, "critic2.pth")))


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_action = env.action_space.high[0]
    agent = TD3Agent(env.observation_space.shape[0], env.action_space.shape[0], max_action, device)
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
