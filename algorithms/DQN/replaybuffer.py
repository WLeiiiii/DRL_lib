import random
from collections import namedtuple, deque

import numpy as np
import torch

from algorithms.DQN.sumtree import SumTree


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


class ReplayBufferCNN:
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

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        terminates = np.array([e.terminated for e in experiences if e is not None])

        # Check if the stacking resulted in the expected shape
        assert states.shape == (batch_size, 210, 160, 3), f"States shape mismatch: {states.shape}"

        # Now convert to PyTorch tensors
        states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)  # Reshape to (N, C, H, W)
        next_states = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(-1).to(self.device)
        terminates = torch.from_numpy(terminates.astype(np.uint8)).unsqueeze(-1).float().to(self.device)

        return states, actions, rewards, next_states, terminates

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer using a binary SumTree for efficient
    priority-based sampling of experiences.
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        """
        Initialize the buffer.

        Parameters:
            capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def __len__(self):
        return self.tree.total_p()

    def store(self, state, action, reward, next_state, terminated):
        """
        Store a new experience in the buffer.
        """
        transition = (state, action, reward, next_state, terminated)
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])  # Find the max priority
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, transition)  # Set the max priority for new transition

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer based on priority.

        Parameters:
            batch_size (int): The size of the batch to sample.

        Returns:
            b_idx (np.array): Array of indices for the sampled experiences.
            b_memory (np.array): Array of sampled experiences.
            ISWeights (np.array): Array of importance sampling weights for the sampled experiences.
        """
        b_idx = np.empty((batch_size,), dtype=np.int32)
        dtypes = [
            ('state', np.float32, (self.tree.data[0][0].size,)),  # 假设 state_size 是状态数组的大小
            ('action', np.int32),
            ('reward', np.float32),
            ('next_state', np.float32, (self.tree.data[0][0].size,)),  # next_state 的大小应与 state_size 相同
            ('terminated', np.bool_)
        ]
        b_memory = np.empty((batch_size,), dtype=dtypes)
        ISWeights = np.empty((batch_size, 1))
        priority_segment = self.tree.total_p() / batch_size  # Priority segment

        # Increase beta each time sample is called, until it reaches 1
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # Calculate the max importance sampling weight
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)

            sampling_prob = priority / self.tree.total_p()
            ISWeights[i, 0] = np.power(sampling_prob / (min_prob + 1e-6), -self.beta)
            b_idx[i] = idx
            b_memory[i] = data
        return b_idx, b_memory, ISWeights

    def update(self, tree_idx, abs_errors):
        """
        Update the priorities of experiences after learning.

        Parameters:
            tree_idx (np.array): Array of indices for the experiences to update.
            abs_errors (np.array): Array of updated absolute errors of the experiences.
        """
        abs_errors += self.epsilon  # Add epsilon to avoid zero probability
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)  # Clip errors
        priorities = np.power(clipped_errors, self.alpha)  # Convert errors to priorities
        for idx, priority in zip(tree_idx, priorities):
            self.tree.update(idx, priority)  # Update the tree with new priorities
