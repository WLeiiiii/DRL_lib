import math

import torch
from torch import nn


class NoisyLinear(nn.Module):
    # NoisyLinear module for Noisy Networks, implements factorised Gaussian noise
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super().__init__()
        self.input_dim = input_dim  # Number of input features
        self.output_dim = output_dim  # Number of output features
        self.std_init = std_init  # Standard deviation for initializing the noise parameters

        # Parameters for the mean of weights
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        # Parameters for the standard deviation of weights
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        # Noise buffer for the weights that gets re-sampled on each forward pass
        self.register_buffer("weight_epsilon", torch.FloatTensor(output_dim, input_dim))

        # Parameters for the mean of biases
        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        # Parameters for the standard deviation of biases
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        # Noise buffer for the biases that gets re-sampled on each forward pass
        self.register_buffer("bias_epsilon", torch.FloatTensor(output_dim))

        self.reset_parameters()  # Initialize parameters
        self.reset_noise()  # Initialize noise

    def reset_parameters(self):
        # Initialize the parameters for the mean of weights and biases
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        # Generate new noise for the weights and biases
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        # Outer product of input and output noise is applied to weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # Bias noise is simply the scaled noise for the output features
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))

    def _scale_noise(self, size):
        # Helper function to generate noise scaled by the square root of its absolute value
        x = torch.randn(size)
        x = x.sign().mul_(x.abs().sqrt())
        return x

    def forward(self, x):
        # Forward pass through the layer with noise applied if training, otherwise use mean
        if self.training:
            # Use the noisy weights and biases during training
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                        self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # Use the mean weights and biases during evaluation
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)


class NoisyQNetworkMLP(nn.Module):
    # A Q-Network that uses NoisyLinear layers for action selection in noisy DQN
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # First fully connected layer
        self.noisy_fc2 = NoisyLinear(hidden_dim, hidden_dim)  # Second noisy layer
        self.noisy_fc3 = NoisyLinear(hidden_dim, action_dim)  # Third noisy layer that outputs action values

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.noisy_fc2(x))
        return self.noisy_fc3(x)

    def reset_noise(self):
        # Reset the noise in the NoisyLinear layers
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()
