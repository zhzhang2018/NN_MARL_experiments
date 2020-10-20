# File for networks that I attempted, for different purposes
import torch
import torch.nn as nn

# Ref: https://github.com/Finspire13/pytorch-policy-gradient-example/blob/master/pg.py
# Implements a network that takes state observations as input, and outputs (discrete) policy/action probabilities.
# Note that PyTorch requires knowing the exact size of each layer...
class PolicyNet(nn.Module):
    def __init__(self, N, ns=2, na=5, hidden=24):
        super(PolicyNet, self).__init__()
        self.N = N # Number of agents

        self.flt = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
        self.fc1 = nn.Linear(ns*self.N, hidden) # Take in flattened input
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, na)  # For DQN, we have to output a value or a probabilty of some sorts for each action

    def forward(self, x): #, y):
        # print(self.flt(x)) # Won't work!?!?
        # print(torch.flatten(x))
#         x = F.relu(self.fc1(torch.cat((torch.flatten(x), y), 0)))
#         x = F.relu(self.fc1(torch.flatten(x)))
        x = torch.relu(self.fc1(self.flt(x)))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Range = [0,1]
        return x

# Implements a net that tries to predict the reward for a state-action pair.
# Should only be able to take in one input, not inputs for all agents
class RewardNet(nn.Module):
    def __init__(self, N, ns=2, na=2, hidden=24):
        super(RewardNet, self).__init__()
        self.N = N # Number of agents

        self.flt1 = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
        self.flt2 = nn.Flatten()
        self.fc1 = nn.Linear(ns*self.N + na, hidden) # Take in flattened input
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)  # For DQN, we have to output a value or a probabilty of some sorts for each action

    def forward(self, x, y):
        # print(self.flt(x)) # Won't work!?!?
        # print(torch.flatten(x))
#         x = F.relu(self.fc1(torch.cat((torch.flatten(x), y), 0)))
#         x = F.relu(self.fc1(torch.flatten(x)))
        x = self.flt1(x)
        y = self.flt2(y)
#         x = torch.flatten(x)
#         y = torch.flatten(y)
#         print(x.shape, y.shape)
#         x = torch.relu(self.fc1( torch.stack( (x, y), dim=1 ).squeeze()))
        x = torch.relu(self.fc1( torch.cat( (x, y), dim=1 ).squeeze()))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.sigmoid(x) # Range = [0,1]
        return x

# Implements a net that tries to predict an action (with a given range, perhaps)
class ActionNet(nn.Module):
    def __init__(self, N, ns=2, na=5, hidden=24, action_range=[-1,1]):
        super(ActionNet, self).__init__()
        self.N = N # Number of agents
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])

        self.flt = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
        self.fc1 = nn.Linear(ns*self.N, hidden) # Take in flattened input
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, na)

    def forward(self, x):
        x = self.flt(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # Range = [-1,1]
        return x * self.range * 0.5 + self.offset

