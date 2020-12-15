# File for networks that I attempted, for different purposes
import torch
import torch.nn as nn

NO_RAND = 0
UNIF_RAND = 1
GAUSS_RAND = 2

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
    def __init__(self, N, ns=2, na=2, hidden=24, leaky=0.01):
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
            nn.BatchNorm1d(num_features = hidden),
            nn.LeakyReLU(negative_slope=leaky), # nn.Tanh(), # nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.BatchNorm1d(num_features = hidden),
            nn.LeakyReLU(negative_slope=leaky), # nn.Tanh(), # nn.ReLU(),
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
#         print(x.shape, y.shape)
        x = torch.cat( (x, y), dim=1 )#.squeeze() # Shouldd I have canclled the squeeze? With it, it may break down when batch size = 1
#         print(x.shape, y.shape)
        for i in range(len(self.RNlayers)):#-1):
            x = self.RNlayers[i](x)
#             x = torch.relu(self.RNlayers[i](x))
#         for fc in self.RNlayers[:-1]:
#             x = torch.relu(fc(x))
#         x = self.RNlayers[-1](x)
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
    def __init__(self, N, ns=2, na=5, hidden=24, action_range=[-1,1], leaky=0.01, rand_mode=NO_RAND):
        super(ActionNet, self).__init__()
        self.N = N # Number of agents
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])
        self.rand_mode = rand_mode
        self.na = na
        
        if self.rand_mode == GAUSS_RAND or self.rand_mode == UNIF_RAND:
            self.na *= 2

#         self.flt = nn.Flatten() # Turns 2D input observation into 1D, so we can use linear layers
#         self.fc1 = nn.Linear(ns*self.N, hidden) # Take in flattened input
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, na)
        
#         self.FTlayers = [self.flt]
#         self.ANlayers = [self.fc1,self.fc2,self.fc3]
        self.FTlayers = nn.ModuleList([nn.Flatten()])
        self.ANlayers = nn.ModuleList([
            nn.Linear(ns*self.N, hidden),
            nn.Tanh(), # nn.LeakyReLU(negative_slope=leaky), # # nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(), # nn.LeakyReLU(negative_slope=leaky), # # nn.ReLU(),
            nn.Linear(hidden, self.na), 
            nn.Tanh()
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
            x = fc(x)
#             x = torch.tanh(fc(x))
#             x = torch.relu(fc(x)) # Could it solve the gradient problem?
        if self.rand_mode == NO_RAND or self.rand_mode == UNIF_RAND:
            return x * self.range * 0.5 + self.offset
        elif self.rand_mode == GAUSS_RAND:
            x_mean = x[:self.na] * self.range * 0.5 + self.offset # Don't mess up the stdev
            # Optionally, you can smooth out stdev to make sure it's non-negative.
            # Ref of method: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            # Ref of usage: https://github.com/zafarali/policy-gradient-methods/blob/f0d83a80ddc772dcad0c851aac9bfd41d436c274/pg_methods/policies.py#L82
            # x[self.na:] = nn.functional.softplus(x[self.na:])
            x_var = x[self.na:]
            return torch.cat((x_mean, x_var))


# Implements a net where actor and critic shares the first layer. (Not used yet)
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
                 tf_hidden=24, leaky=0.01):
        super(ActionNetTF, self).__init__(prevN, ns, na, hidden, action_range, leaky)
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
        self.ANlayers.insert(1,nn.Tanh())
        self.ANlayers.insert(2,nn.Linear(tf_hidden, ns*prevN))
        
#         self.ANlayers.insert(0,nn.Linear(tf_hidden, ns*prevN))
#         self.ANlayers.insert(0,nn.Linear(tf_hidden, tf_hidden))
#         self.ANlayers.insert(0,nn.Linear(tf_hidden, tf_hidden))
#         self.ANlayers.insert(0,nn.Linear(ns*N, tf_hidden))
        
        
class RewardNetTF(RewardNet):
    def __init__(self, N, prevN, path, ns=2, na=5, hidden=24, tf_hidden=24, leaky=0.01):
        super(RewardNetTF, self).__init__(prevN, ns, na, hidden, leaky)
        
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
        self.RNlayers.insert(1,nn.LeakyReLU(leaky))
        self.RNlayers.insert(2,nn.Linear(tf_hidden, ns*prevN + na))
        
#         self.RNlayers.insert(0,nn.Linear(tf_hidden, ns*prevN + na))
#         self.RNlayers.insert(0,nn.Linear(tf_hidden, tf_hidden))
#         self.RNlayers.insert(0,nn.Linear(tf_hidden, tf_hidden))
#         self.RNlayers.insert(0,nn.Linear(ns*N+na, tf_hidden))

# Implement a CNN to learn reward mapping directly
class RewardCNN(nn.Module):
    def __init__(self, N, ns=2, na=2, hidden=4, n_hid=2, in_features=1, leaky=0.01):
        super().__init__()
        self.N = N # Number of agents
        self.hidden = hidden # Number of channels for intermediate layers
        self.na = na
        # Starting input shape is (B, 1, F, N), where F=ns+na: We attach a (na,N) layer below to include current action... would this even work?

        layerlist = []
        layerlist.append(nn.Conv2d(
                in_channels=in_features, out_channels=hidden, kernel_size=(1, 1), stride=(1, 1)
            )) # (B, 1, F, N) --> (B, H, F, N)
        for i in range(n_hid):
            if i == 0:
                # Aggregate neighbor readings in the first layer? 
                layerlist.append(nn.Conv2d(
                    in_channels=hidden, out_channels=hidden, kernel_size=(1, 1), stride=(1, N)
                )) # (B, H, F, N) --> (B, H, F, 1)
            else:
                layerlist.append(nn.Conv2d(
                    in_channels=hidden, out_channels=hidden, kernel_size=(1, 1), stride=(1, 1)
                )) # (B, H, F, N/1) --> (B, H, F, N/1)
            layerlist.append(nn.LeakyReLU(leaky))
        # Aggregate the features and gather all the channels
        layerlist.append(nn.Conv2d(
                in_channels=hidden, out_channels=1, kernel_size=(ns+na, 1), stride=(1, 1)
            )) # (B, H, F, N) --> (B, 1, 1, 1)
        
        self.RCNNlayers = nn.ModuleList(layerlist) 

    def forward(self, x,y):
        # Assumption: y started off as an action (B,na,???), and we need it to become (B,1,na,N)
        y = y.view((-1,1,self.na,1))
#         print(x.shape, y.shape)
        y = y.repeat(1,1,1,self.N)
        x = torch.unsqueeze(x, dim=1) # (B, ns, N) --> (B, 1, ns, N)
#         print(x.shape, y.shape)
        x = torch.cat( (x, y), dim=2 )
        for i in range(len(self.RCNNlayers)):#-1):
            x = self.RCNNlayers[i](x)
#             x = torch.relu(self.RCNNlayers[i](x))
#         x = self.RCNNlayers[-1](x)
        # x = torch.sigmoid(x) # Range = [0,1]
        x = torch.squeeze(x)
        return x

from torchvision.models.resnet import *
# Or use this one: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class RewardResNet(ResNet):
    def __init__(self, N,
        block, layers, num_classes=2, zero_init_residual=False,
        groups=1, width_per_group=64, replace_stride_with_dilation=None, 
        ns=2, na=2, hidden=64, n_hid=2, in_features=1):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.N = N # Number of agents
        self.na = na
        self.inplanes = hidden
        self.hidden = hidden
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1,N),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                    
        self.RRNlayers = nn.ModuleList([self.conv1,self.bn1,self.relu,self.maxpool,
                                         self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool,self.fc]) 

    def forward(self, x,y):
        # Assumption: y started off as an action (B,na,???), and we need it to become (B,1,na,N)
        y = y.view((-1,1,self.na,1))
#         print(x.shape, y.shape)
        y = y.repeat(1,1,1,self.N)
        x = torch.unsqueeze(x, dim=1) # (B, ns, N) --> (B, 1, ns, N)
#         print(x.shape, y.shape)
        x = torch.cat( (x, y), dim=2 )
        x = self._forward_impl(x) # Inherited
        x = torch.squeeze(x)
        return x
    
# Implement a CNN to learn action mapping directly
class ActionCNN(nn.Module):
    def __init__(self, N, ns=2, na=2, hidden=4, n_hid=2, in_features=1, action_range=[-1,1]):
        super().__init__()
        self.N = N # Number of agents
        self.hidden = hidden # Number of channels for intermediate layers
        self.na = na
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])

        layerlist = []
        layerlist.append(nn.Conv2d(
                in_channels=in_features, out_channels=hidden, kernel_size=(1, 1), stride=(1, 1)
            )) # (B, 1, F, N) --> (B, H, F, N)
        for i in range(n_hid):
            if i == 0:
                # Aggregate neighbor readings in the first layer? 
                layerlist.append(nn.Conv2d(
                    in_channels=hidden, out_channels=hidden, kernel_size=(1, 1), stride=(1, N)
                )) # (B, H, F, N) --> (B, H, F, 1)
            else:
                layerlist.append(nn.Conv2d(
                    in_channels=hidden, out_channels=hidden, kernel_size=(1, 1), stride=(1, 1)
                )) # (B, H, F, N/1) --> (B, H, F, N/1)
        # Aggregate the features and gather all the channels
        layerlist.append(nn.Conv2d(
                in_channels=hidden, out_channels=na, kernel_size=(ns, 1), stride=(1, 1)
            )) # (B, H, F, N) --> (B, na, 1, 1)
        
        self.ACNNlayers = nn.ModuleList(layerlist) 

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # (B, F, N) --> (B, 1, F, N)
#         print(x.shape)
        for i in range(len(self.ACNNlayers)):
            x = torch.tanh(self.ACNNlayers[i](x))
#             print(x.shape)
#         x = self.ACNNlayers[-1](x)
        # x = torch.sigmoid(x) # Range = [0,1]
        x = x.view((-1, self.na)) # --> (B, na)
        return x * self.range * 0.5 + self.offset
