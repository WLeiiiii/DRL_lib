import random
from collections import namedtuple, deque

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
        pass

    def store(self, state, action, reward, next_state, terminated):
        self.memory.append(self.experience(state, action, reward, next_state, terminated))
        pass

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
        pass

    def store(self, state, action, reward, next_state, terminated):
        self.memory.append(self.experience(state, action, reward, next_state, terminated))
        pass

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
        self.tree = SumTree(capacity)

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
        b_memory = np.empty((batch_size, self.tree.data[0].size))
        ISWeights = np.empty((batch_size, 1))
        priority_segment = self.tree.total_p / batch_size  # Priority segment

        # Increase beta each time sample is called, until it reaches 1
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # Calculate the max importance sampling weight
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            sampling_prob = priority / self.tree.total_p
            ISWeights[i, 0] = np.power(sampling_prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
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


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.n_entries = 0

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1  # 新添加的行：增加存储的经验数量

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
