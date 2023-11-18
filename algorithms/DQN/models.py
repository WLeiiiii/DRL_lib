import torch
from torch import nn, autograd


class QNetworkMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QNetworkCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Assuming input_dim is in the form of (H, W, C), we reorder it to (C, H, W)
        channel_dim = input_dim[2]
        height_dim = input_dim[0]
        width_dim = input_dim[1]
        self.input_dim = (channel_dim, height_dim, width_dim)
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def feature_size(self):
        # Use the reordered self.input_dim to create a dummy input for size calculation
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x / 255.0)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class DuelQNetworkMLP(nn.Module):
    # Dueling network MLP
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value stream layers
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        # Advantage stream layers
        self.advantage_fc = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = torch.relu(self.value_fc(x))
        value = self.value(value)

        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        # Combine value and advantage streams to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return value + advantage - advantage.mean()


class DuelQNetworkCNN(nn.Module):
    # Dueling network CNN
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Assuming input_dim is in the form of (H, W, C), we reorder it to (C, H, W)
        channel_dim = input_dim[2]
        height_dim = input_dim[0]
        width_dim = input_dim[1]
        self.input_dim = (channel_dim, height_dim, width_dim)
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the output from the feature extraction layers
        self.feature_output_size = self.feature_size()

        # Value stream layers
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Outputs a single value representing V(s)
        )

        # Advantage stream layers
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)  # Outputs A(s, a) for each action
        )

    def feature_size(self):
        # Use the reordered self.input_dim to create a dummy input for size calculation
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    def forward(self, x):
        # Feature extraction
        x = self.features(x / 255.0)
        x = x.reshape(x.size(0), -1)

        # Value stream
        value = self.value_stream(x)

        # Advantage stream
        advantage = self.advantage_stream(x)

        # Combine value and advantage streams to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return value + advantage - advantage.mean()


class C51Net(nn.Module):
    def __init__(self, input_dim, output_dim, atoms_dim=51):
        super().__init__()
        channel_dim = input_dim[2]
        height_dim = input_dim[0]
        width_dim = input_dim[1]
        self.input_dim = (channel_dim, height_dim, width_dim)
        self.output_dim = output_dim
        self.atoms_dim = atoms_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim * self.atoms_dim)
        )

    def feature_size(self):
        # Use the reordered self.input_dim to create a dummy input for size calculation
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x / 255.0)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(x.size(0), self.output_dim, self.atoms_dim)
        return nn.functional.softmax(x, dim=-1)
