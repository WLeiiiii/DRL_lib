import torch
from torch import nn

import torch.nn.functional as F
from torch.distributions import Normal


class ValueNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(128, action_dim)
        # self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3_mu = nn.Linear(256, action_dim)
        self.fc3_mu.weight.data.normal_(0, 0.1)
        self.fc3_std = nn.Linear(256, action_dim)
        self.fc3_std.weight.data.normal_(0, 0.1)
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.fc3_mu(x)
        std = F.softplus(self.fc3_std(x))

        dist = Normal(mu, std)
        normal_action = dist.sample()
        log_prob = dist.log_prob(normal_action)
        action = torch.tanh(normal_action)
        # 限制动作范围会影响到动作的概率密度函数。
        # 这是因为 tanh 函数的导数在边界点上接近于零，
        # 这可能导致在这些点上计算的概率密度非常小，甚至接近于零。
        # 这会导致梯度消失，从而影响模型的训练效果。
        # 为了解决这个问题，可以使用公式 log(1 - tanh^2(x) + ε)
        # 来重新计算对数概率密度，其中 ε 是一个较小的常数（在这里是 1e-7），
        # 用于避免取对数时的除零错误。这样可以保持对数概率密度的合理值，
        # 并避免梯度消失的问题。
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob
        pass
