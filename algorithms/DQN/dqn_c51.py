import math
import os

import torch

from algorithms.DQN.config import C51Config
from algorithms.DQN.models import C51Net
from algorithms.DQN.replaybuffer import ReplayBufferCNN
from algorithms.algo_utils import soft_update


class C51Agent:
    def __init__(self, state_size, action_size, device):
        """
        Initialize a Categorical DQN agent (also known as C51).

        :param state_size: Dimensionality of the state space.
        :param action_size: Number of actions in the action space.
        :param device: The device (CPU or GPU) on which to perform computations.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.cfgs = C51Config()  # Instance of a configuration class with hyperparameters for the agent.
        self.cfgs.hidden_dim = 512
        self.cfgs.__information__()

        # Neural networks for the local and target Q-values.
        # The local network is updated every step, while the target network is updated less frequently
        # for stability.
        self.local_net = C51Net(state_size, action_size, self.cfgs.num_atoms).to(device)
        self.target_net = C51Net(state_size, action_size, self.cfgs.num_atoms).to(device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        # Optimizer for applying gradients to the parameters of the local network.
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.cfgs.learning_rate)

        # A replay buffer for storing experience tuples.
        self.replay_buffer = ReplayBufferCNN(state_size, self.cfgs.buffer_size, self.device)

        # Initialize the networks in training mode.
        self.train()
        self.target_net.train()
        self.time_step = 0  # Counter for the time steps to determine when to update the network.

    def train(self, training=True):
        """
        Set the local network to training mode.

        :param training: Boolean flag for whether the network should be in training mode.
        """
        self.training = training
        self.local_net.train(training)

    def step(self, logger=None, step=None):
        """
        Increment the time step and update the network at the defined frequency.

        :param logger: Optional logger to record training statistics.
        :param step: The current training step.
        """
        self.time_step = (self.time_step + 1) % self.cfgs.update_frequency
        if self.time_step == 0 and self.replay_buffer.__len__() >= self.cfgs.batch_size:
            self.learn(logger, step)

    def act(self, state, epsilon=0.):
        """
        Choose an action based on the current state and an epsilon-greedy policy.

        :param state: The current state representation.
        :param epsilon: The probability of choosing a random action.
        :return: The chosen action.
        """
        # Convert the state to a PyTorch tensor and adjust dimensions to match the network's expectations.
        state = torch.from_numpy(state).float().to(self.device)
        state = state.permute(2, 0, 1)
        state = state.unsqueeze(0)

        # Epsilon-greedy action selection.
        if torch.rand(1) > epsilon:
            # If exploiting, select the action with the highest Q-value.
            q_value_dist = self.local_net(state)
            q_value = torch.sum(
                q_value_dist * torch.linspace(self.cfgs.v_min, self.cfgs.v_max, self.cfgs.num_atoms).view(1, 1, -1).to(
                    self.device), dim=2)
            return q_value.argmax().item()
        else:
            # If exploring, select a random action.
            return torch.randint(0, self.action_size, (1,)).item()

    def learn(self, logger=None, step=None):
        """
        Sample a batch of experiences from the replay buffer and perform learning.

        :param logger: Optional logger to record training statistics.
        :param step: The current training step.
        """
        # Sample a batch of experiences from the replay buffer.
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
            self.cfgs.batch_size)

        # Compute Q-value distributions for next states and select best actions based on the maximum Q-value.
        q_target_dist = self.target_net(next_state_batch).detach()
        q_target = torch.sum(
            q_target_dist * torch.linspace(self.cfgs.v_min, self.cfgs.v_max, self.cfgs.num_atoms).view(1, 1, -1).to(
                self.device), dim=2)
        next_actions = q_target.argmax(dim=1, keepdim=True).detach()
        q_target_next_dist = q_target_dist.gather(1, next_actions.unsqueeze(2).expand(-1, -1, self.cfgs.num_atoms))
        q_target_next_dist = q_target_next_dist.squeeze(1)

        # Project the target distribution onto the support of the current state's Z-distribution.
        projected_dist = self.project_distribution(q_target_next_dist, reward_batch, done_batch)

        # Compute Q-value distributions for current states and actions.
        q_current_dist = self.local_net(state_batch)
        q_dist = q_current_dist.gather(1, action_batch.unsqueeze(2).expand(-1, -1, self.cfgs.num_atoms))
        q_dist = q_dist.squeeze(1)

        # Compute the cross-entropy loss between the projected distribution and the current distribution.
        loss = torch.sum(projected_dist * (-torch.log(q_dist + 1e-8)), dim=1).mean()

        # Perform gradient descent.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Softly update the target network.
        soft_update(self.local_net, self.target_net, self.cfgs.tau)

        # Log training information if a logger is provided.
        if logger:
            logger.log("train/batch_reward", reward_batch.mean(), step)
            logger.log("train/loss", loss, step)

    def project_distribution(self, q_target_next_dist, rewards, dones):
        """
        Project the target distribution onto the current state's support using the Categorical algorithm.

        :param q_target_next_dist: The target Q-value distribution for the next state.
        :param rewards: A batch of rewards.
        :param dones: A batch of done flags indicating whether an episode has finished.
        :return: The projected distribution.
        """
        # Initialize the projected distribution as a zero tensor with the same shape as the target distribution.
        projected_dist = torch.zeros_like(q_target_next_dist)

        # Project the target distribution for each batch item individually.
        for i in range(self.cfgs.batch_size):
            if dones[i]:
                # Handle terminal states.
                Tz = min(self.cfgs.v_max, max(self.cfgs.v_min, rewards[i]))
                b = (Tz - self.cfgs.v_min) / self.cfgs.delta_z
                l = int(b)
                u = int(math.ceil(b))
                projected_dist[i, l] += (u - b)
                projected_dist[i, u] += (b - l)
            else:
                # Handle non-terminal states.
                for j in range(self.cfgs.num_atoms):
                    Tz = min(self.cfgs.v_max,
                             max(self.cfgs.v_min, rewards[i] + self.cfgs.gamma * self.cfgs.z_values[j]))
                    b = (Tz - self.cfgs.v_min) / self.cfgs.delta_z
                    l = int(b)
                    u = int(math.ceil(b))
                    projected_dist[i, l] += q_target_next_dist[i, j] * (u - b).item()
                    projected_dist[i, u] += q_target_next_dist[i, j] * (b - l).item()

        return projected_dist

    def save(self, path):
        """
        Save the local network model to the specified path.

        :param path: The directory path to save the model.
        """
        torch.save(self.local_net.state_dict(), os.path.join(path, "c51_network.pth"))

    def load(self, path):
        """
        Load the local network model from the specified path.

        :param path: The directory path from which to load the model.
        """
        self.local_net.load_state_dict(torch.load(os.path.join(path, "c51_network.pth")))
        self.target_net.load_state_dict(self.local_net.state_dict())  # Ensure the target network is identical


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("BreakoutNoFrameskip-v4", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = C51Agent(env.observation_space.shape, env.action_space.n, device)
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
