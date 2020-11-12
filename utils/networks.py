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
        
        self.FTlayers = [self.flt]
        self.PNlayers = [self.fc1,self.fc2,self.fc3]

    def forward(self, x): #, y):
        # print(self.flt(x)) # Won't work!?!?
        # print(torch.flatten(x))
#         x = F.relu(self.fc1(torch.cat((torch.flatten(x), y), 0)))
#         x = F.relu(self.fc1(torch.flatten(x)))
        ### Update: Modified to a more recent version - see below 11/04
#         x = torch.relu(self.fc1(self.flt(x)))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x)) # Range = [0,1]
        for flt in self.FTlayers:
            x = flt(x)
        for fc in self.PNlayers[:-1]:
            x = torch.relu(fc(x))
        x = self.PNlayers[-1](x)
        x = torch.sigmoid(x) # Range = [0,1]
        return x

# Implements a net that tries to predict the reward for a state-action pair.
# Should only be able to take in one input, not inputs for all agents
class RewardNet(nn.Module):
    def __init__(self, N, ns=2, na=2, hidden=24):
        super(RewardNet, self).__init__()
        self.N = N # Number of agents

#         self.flt1 = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
#         self.flt2 = nn.Flatten()
#         self.fc1 = nn.Linear(ns*self.N + na, hidden) # Take in flattened input
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, 1)  # For DQN, we have to output a value or a probabilty of some sorts for each action

#         self.FTlayers = nn.ModuleList([self.flt1,self.flt2])
#         self.RNlayers = nn.ModuleList([self.fc1,self.fc2,self.fc3])
        self.FTlayers = nn.ModuleList([nn.Flatten(), nn.Flatten()])
        self.RNlayers = nn.ModuleList([
            nn.Linear(ns*self.N + na, hidden),
            nn.Linear(hidden, hidden), 
            nn.Linear(hidden, 1)
        ])

    def forward(self, x, y):
#         # print(self.flt(x)) # Won't work!?!?
#         # print(torch.flatten(x))
# #         x = F.relu(self.fc1(torch.cat((torch.flatten(x), y), 0)))
# #         x = F.relu(self.fc1(torch.flatten(x)))
#         x = self.flt1(x)
#         y = self.flt2(y)
# #         x = torch.flatten(x)
# #         y = torch.flatten(y)
# #         print(x.shape, y.shape)
# #         x = torch.relu(self.fc1( torch.stack( (x, y), dim=1 ).squeeze()))
#         x = torch.relu(self.fc1( torch.cat( (x, y), dim=1 ).squeeze()))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         # x = torch.sigmoid(x) # Range = [0,1]
        # Updated 11/04 to suit for more flexible children
        x = self.FTlayers[0](x)
        y = self.FTlayers[1](y)
        x = torch.cat( (x, y), dim=1 ).squeeze()
        for i in range(len(self.RNlayers)-1):
            x = torch.relu(self.RNlayers[i](x))
#         for fc in self.RNlayers[:-1]:
#             x = torch.relu(fc(x))
        x = self.RNlayers[-1](x)
        # x = torch.sigmoid(x) # Range = [0,1]
        return x

# Implements a net that tries to predict the reward for state alone
class RewardStateNet(nn.Module):
    def __init__(self, N, ns=2, hidden=24):
        super(RewardNet, self).__init__()
        self.N = N # Number of agents

        self.FTlayers = nn.ModuleList([nn.Flatten()])
        self.RNlayers = nn.ModuleList([
            nn.Linear(ns*self.N, hidden),
            nn.Linear(hidden, hidden), 
            nn.Linear(hidden, 1)
        ])

    def forward(self, x, y):
        for flt in self.FTlayers:
            x = flt(x)
        for i in range(len(self.RNlayers)-1):
            x = torch.relu(self.RNlayers[i](x))
        x = self.RNlayers[-1](x)
#         x = torch.sigmoid(x) # Range = [0,1] for normalized state values
        return x

# Implements a net that tries to predict an action (with a given range, perhaps)
class ActionNet(nn.Module):
    def __init__(self, N, ns=2, na=5, hidden=24, action_range=[-1,1]):
        super(ActionNet, self).__init__()
        self.N = N # Number of agents
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])

#         self.flt = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
#         self.fc1 = nn.Linear(ns*self.N, hidden) # Take in flattened input
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, na)
        
#         self.FTlayers = [self.flt]
#         self.ANlayers = [self.fc1,self.fc2,self.fc3]
        self.FTlayers = nn.ModuleList([nn.Flatten()])
        self.ANlayers = nn.ModuleList([
            nn.Linear(ns*self.N, hidden),
            nn.Linear(hidden, hidden), 
            nn.Linear(hidden, na)
        ])
        
        # Initialization? 
#         for fc in self.ANlayers:
#             torch.nn.init.kaiming_uniform_(fc.weight,nonlinearity='relu')

    def forward(self, x):
        ### Updated 11/04
#         x = self.flt(x)
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x)) # Range = [-1,1]
#         return x * self.range * 0.5 + self.offset
        for flt in self.FTlayers:
            x = flt(x)
        for fc in self.ANlayers:
            x = torch.tanh(fc(x))
#             x = torch.relu(fc(x)) # Could it solve the gradient problem?
        return x * self.range * 0.5 + self.offset


# Implements a net where actor and critic shares the first layer.
# Here, the critic network doesn't depend on action no more. 
class ActorCriticNet(nn.Module):
    def __init__(self, N, ns=2, na=5, hidden=24, action_range=[-1,1]):
        super(ActionNet, self).__init__()
        self.N = N # Number of agents
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])

        self.FTlayers = nn.ModuleList([nn.Flatten()])
        self.shared_layers = nn.ModuleList([nn.Linear(ns*self.N, hidden)])
        self.ANlayers = nn.ModuleList([
            nn.Linear(hidden, hidden), 
            nn.Linear(hidden, na)
        ])
        self.RNlayers = nn.ModuleList([
            nn.Linear(hidden, hidden), 
            nn.Linear(hidden, 1)
        ])
        
    def forward(self, x):
        for flt in self.FTlayers:
            x = flt(x)
        for sh in self.shared_layers:
            x = sh(x)
        x_a = x
        x_c = x
        for fc in self.ANlayers:
            x_a = torch.relu(fc(x_a))
        for fc in self.RNlayers:
            x_c = torch.relu(fc(x_c))
        return (x_a * self.range * 0.5 + self.offset), x_c


# Implements a net that tries to predict an energy function.
# Input: (B, N, ns) or (B, ns, N)
# Output: (B, 1)
class EnergyNet(nn.Module):
    def __init__(self, N, ns=2, hidden=24):
        super(EnergyNet, self).__init__()
        self.N = N # Number of agents
        self.flt = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
        self.fc1 = nn.Linear(ns*self.N, hidden) # Take in flattened input
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        
        self.FTlayers = [self.flt]
        self.ENlayers = [self.fc1,self.fc2,self.fc3]

    def forward(self, x):
        ### Updated 11/04
#         x = self.flt(x)
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x)) 
#         return x
        for flt in self.FTlayers:
            x = flt(x)
        for fc in self.ENlayers:
            x = torch.tanh(fc(x))
        return x

# Implements the transfer learning functionality for neural nets
class ActionNetTF(ActionNet):
    def __init__(self, N, prevN, path, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 tf_hidden=24):
        super(ActionNetTF, self).__init__(prevN, ns, na, hidden, action_range)
        self.N = N # Number of agents

        # Load layers
        self.load_state_dict(torch.load(path))
        
        # Freeze loaded layers
        for fc in self.ANlayers:
            fc.weight.requires_grad = False
            fc.bias.requires_grad = False
        
        # Create additional layers to suit the size difference
        # self.tf1 = nn.Linear(ns*N, tf_hidden)
        # self.tf2 = nn.Linear(tf_hidden, ns*prevN)
        # self.ANlayers = [self.tf1, self.tf2] + self.ANlayers
        # Here's hoping that this would train...
        self.ANlayers.insert(0,nn.Linear(ns*N, tf_hidden))
        self.ANlayers.insert(1,nn.Linear(tf_hidden, ns*prevN))
        
        
class RewardNetTF(RewardNet):
    def __init__(self, N, prevN, path, ns=2, na=5, hidden=24, tf_hidden=24):
        super(RewardNetTF, self).__init__(prevN, ns, na, hidden)
        
        self.N = N # Number of agents
        
        # Load layers
        self.load_state_dict(torch.load(path))

        # Freeze loaded layers
        for fc in self.RNlayers:
            fc.weight.requires_grad = False
            fc.bias.requires_grad = False
        
        # Create additional layers to suit the size difference
        # self.tf1 = nn.Linear(ns*N+na, tf_hidden)
        # self.tf2 = nn.Linear(tf_hidden, ns*prevN + na)
        # self.RNlayers = [self.tf1, self.tf2] + self.RNlayers
        self.RNlayers.insert(0,nn.Linear(ns*N+na, tf_hidden))
        self.RNlayers.insert(1,nn.Linear(tf_hidden, ns*prevN + na))
