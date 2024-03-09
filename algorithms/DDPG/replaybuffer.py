import random

from collections import deque, namedtuple
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, action_dim, buffer_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_dim
        self.experience = namedtuple(
            "experience",
            field_names=["state", "action", "reward", "next_state", "terminated"]
        )
        self.device = device

    def store(self, state, action, reward, next_state, terminated):
        self.memory.append(self.experience(state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        terminates = torch.from_numpy(
            np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, terminates

    def __len__(self):
        return len(self.memory)
