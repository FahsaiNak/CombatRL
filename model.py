import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim_lst, fc1_dims=64, fc2_dims=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, sum(action_dim_lst))
        self.action_dim_lst = action_dim_lst

    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        h1, h2 = torch.split(h, self.action_dim_lst, dim=1)
        y1 = torch.softmax(h1, dim=1)
        y2 = torch.softmax(h2, dim=1)
        return y1, y2


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim_lst, fc1_dims=64, fc2_dims=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim+sum(action_dim_lst), fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, state, action):
        state = state.squeeze(0)
        h = F.relu(self.fc1(torch.cat([state, action], axis=1)))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y