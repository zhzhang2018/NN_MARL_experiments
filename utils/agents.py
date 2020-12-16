from utils.networks import *
from utils.ReplayMemory import *
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import os
from datetime import datetime

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 10
n_actions = 5
episode_durations = []

batch_size = 32
learning_rate = 0.01 # Default value for RMSprop
gamma = 0.99

class BaseAgent:
    def __init__(self, device, N):
        self.device = device
        self.N = N
        self.needsExpert = False
        self.name = 'BaseAgent'
        self.losses = []
        self.lossesA = []
        self.centralized = False
        self.centralizedA = False
    
    def select_action(self, state, **kwargs):
        pass
    
    def optimize_model(self, batch, **kwargs):
        pass
    
    def save_model(self, suffix="", agent_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if len(suffix) <= 0:
            suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if agent_path is None:
            agent_path = "models/{}_{}".format(self.name, suffix)
        print('Saving model to {}'.format(agent_path))
        torch.save(self.net.state_dict(), agent_path)

    def load_model(self, agent_path):
        print('Loading model from {}'.format(agent_path))
        if agent_path is not None:
            self.net.load_state_dict(torch.load(agent_path))#.to(self.device)
    
    def set_train(self, train):
        if train:
            self.net.train()
        else:
            self.net.eval()

class DqnAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, learning_rate=0.01):
        super().__init__(device, N)
        self.policy_net = PolicyNet(N, ns, na, hidden)
        self.net = self.policy_net
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.name = 'DQNAgent'
        
    # Throws a coin to decide whether to randomly sample or to choose according to reward.
    # Coin prob will change over time.
    # This method should be called for each individual agent.
    def select_action(self, state, **kwargs):
        steps_done = kwargs.get('steps_done', 0)
        rand = kwargs.get('rand', True)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold or (not rand):
            with torch.no_grad():
                # t.max(1) will return largest column value of each row (sample?).
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state.view(1,-1,self.N)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action))
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
                                state_batch.view(BATCH_SIZE, -1, self.N)
                            ).gather(
                                1, action_batch.view(-1,1)
                            ) # gather() Gathers values along axis=1, indexed by action_batch.

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

        self.policy_net.eval()
        next_state_values[non_final_mask] = self.policy_net(
                                        non_final_next_states.view(BATCH_SIZE, 2, -1)
                                    ).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        self.policy_net.train()

        # Compute Huber loss
        loss = torch.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
# Agent for following consensus protocol
class LearnerAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], learning_rate=0.01, centralized=False, 
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = self.centralized
        if centralized:
            # If centralized, then ns should be no larger than env.ns, and the user should be responsible for passing in the right value
            if load_path is None:
                self.net = ActionNet(N, ns, na*N, hidden, action_range)
            else:
                self.net = ActionNetTF(N, prevN, load_path, ns, na*N, hidden, action_range)
        else:
            if load_path is None:
                self.net = ActionNet(N, ns, na, hidden, action_range)
            else:
                self.net = ActionNetTF(N, prevN, load_path, ns, na, hidden, action_range)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer) # Not used for LA yet
        self.needsExpert = True
        self.name = 'LearnerAgent'
        
        self.ns = ns  # Probably forcifully remove all the rest of the parameters to avoid shape mismatch
        self.na = na
        
    # Picks an action based on given state
    def select_action(self, state, **kwargs):
#         print(self.net(state.view(1,-1,self.N)), self.net(state.view(1,-1,self.N)).shape)
        with torch.no_grad():
            if self.centralized:
                return self.net(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy().reshape((self.N,-1))
            else:
                return self.net(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy() # Expected size: (B=1, na, N) -> (na,N)?
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # Should I squeeze?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train() 
        pred_action = self.net(state_batch.view(B, -1, self.N)[:,:self.ns,:])
        if self.centralized:
            pred_action = pred_action.view(-1,self.N,self.na) 
        # Input shape should be (B,no,N) and output be (B,na),
        # which is then reshaped from (B,na*N) into (B,N,na) if centralized
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(action_batch, pred_action)
#         print(loss)
        loss.backward()
#         print("4th Last layer gradients after backward: ", torch.mean(self.net.ANlayers[3].weight.grad))
#         print("3rd Last layer gradients after backward: ", torch.mean(self.net.ANlayers[2].weight.grad))
#         print("2nd Last layer gradients after backward: ", torch.mean(self.net.ANlayers[1].weight.grad))
#         print("Last     layer gradients after backward: ", torch.mean(self.net.ANlayers[0].weight.grad))
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
        # Check sizes - attach real size after those lines.
#         print(state_batch.shape)
#         print(action_batch.shape, pred_action.shape)
#         print(reward_batch.shape)
class LearnerCNNAgent(BaseAgent):
    def __init__(self, device, N, ns=4, na=2, hidden=24, n_hid=2, in_features=1, action_range=[-1,1], learning_rate=0.01, centralized=False, 
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = self.centralized
        if centralized:
            if load_path is None:
                self.net = ActionCNN(N, ns, na*N, hidden, n_hid, in_features, action_range)
            else:
                self.net = ActionNetTF(N, prevN, load_path, ns, na*N, hidden, action_range)
        else:
            if load_path is None:
                self.net = ActionCNN(N, ns, na, hidden, n_hid, in_features, action_range)
            else:
                self.net = ActionNetTF(N, prevN, load_path, ns, na, hidden, action_range)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer) # Not used yet
        self.needsExpert = True
        self.name = 'LearnerCNNAgent'
        
        self.ns = ns
        self.na = na
        
    # Picks an action based on given state
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            if self.centralized:
                return self.net(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy().reshape((self.N,-1))
            else:
                return self.net(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy() 
                # Expected size: (B=1, na, N) -> (B=1, na, N) -> (na,N)?
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train() 
        pred_action = self.net(state_batch.view(B, -1, self.N)[:,:self.ns,:]) # Input shape should be (B,no,N) and output be (B,na)
        if self.centralized:
            pred_action = pred_action.view(-1,self.N,self.na) 
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(action_batch, pred_action)
#         print("ACNN loss: ", loss)
        loss.backward()
#         print("4th Last layer gradients after backward: ", torch.mean(self.net.ACNNlayers[3].weight.grad))
#         print("3rd Last layer gradients after backward: ", torch.mean(self.net.ACNNlayers[2].weight.grad))
#         print("2nd Last layer gradients after backward: ", torch.mean(self.net.ACNNlayers[1].weight.grad))
#         print("The Last layer gradients after backward: ", torch.mean(self.net.ACNNlayers[0].weight.grad))
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
    
# Agent for leaning reward
class RewardAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, learning_rate=0.01, centralized=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = self.centralized
        if centralized:
            if load_path is None:
                self.net = RewardNet(N, ns, na*N, hidden)
            else:
                self.net = RewardNetTF(N, prevN, load_path, ns, na*N, hidden)
        else:
            if load_path is None:
                self.net = RewardNet(N, ns, na, hidden)
            else:
                self.net = RewardNetTF(N, prevN, load_path, ns, na, hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)#RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        # Othe rchoices ffor scheduler: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # Scheduler.step() should be called in the training method.
        self.na = na
        self.name = 'RewardAgent'
    
    # Idk how to implement this... randomly sample a bunch of possible actions and then pick the best one?
    def select_action(self, state, **kwargs):
        num_sample = kwargs.get('num_sample', 50)
        if self.centralized:
            # If centralized, technically we would need to multiply the potential action sample amount by 2^N.
            ### TODO: Implement a better method for finding the optimal action in this case.
            num_sample *= self.N*2
        action_space = kwargs.get('action_space', [-1,1])
        with torch.no_grad():
#             actions = torch.from_numpy( np.random.rand(num_sample, self.na, self.N) )
            if self.centralized:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.N, self.na).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
            else:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.na).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
#             print(state.expand(num_sample, -1, -1).shape, actions.shape)
#             print(state)
            rewards = self.net(state.expand(num_sample, -1, -1), actions).view(-1)
#             print(rewards.shape, rewards.max(0))
            bestind = rewards.max(0)[1]
#             print(bestind, actions)
            return actions[bestind.detach()].numpy()#detach()
    
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch[0]))
#         print(B, len(batch[0]), len(batch), len(batch.action))
        ### TODO: BatchNorm is known to introduce issue when batch size is 1. Find a better way to solve this, instead
        # of using the simple method below.
        if B <= 1:
            print("I didn't learn anything!")
            return
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train()
#         print("State batch shape = ", state_batch.view(B, -1, self.N).shape, "; action shape = ", action_batch.view(B, -1, 1).shape)
        pred_reward = self.net(
            state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1)
        ) # Shape is (B,ns,N) and (B,na) for input and (B, ) for output??
#         print("Reward batch shape = ", reward_batch.shape, "; prediction shape = ", pred_reward.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
#         print(loss)
#         print("First layer gradients before backward: ", self.net.RNlayers[-1].weight.grad)
#         print("Last layer gradients before backward: ", self.net.RNlayers[0].weight.grad)
        loss.backward()
#         print("First    layer gradients after backward: ", torch.mean(self.net.RNlayers[-1].weight.grad))
#         print(self.net.RNlayers[-1].weight.grad)
#         print("6th Last layer gradients after backward: ", torch.mean(self.net.RNlayers[5].weight.grad))
#         print("5th Last layer gradients after backward: ", torch.mean(self.net.RNlayers[4].weight.grad))
#         print("4th Last layer gradients after backward: ", torch.mean(self.net.RNlayers[3].weight.grad))
#         print("3rd Last layer gradients after backward: ", torch.mean(self.net.RNlayers[2].weight.grad))
#         print("2nd Last layer gradients after backward: ", torch.mean(self.net.RNlayers[1].weight.grad))
#         print("Last     layer gradients after backward: ", torch.mean(self.net.RNlayers[0].weight.grad))
#         print(self.net.RNlayers[0].weight.grad)
        # print("Should be the same as: ", self.net.fc1.weight.grad)
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
        
# Agent for leaning reward
class RewardCNNAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, n_hid=2, in_features=1, learning_rate=0.01, centralized=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = self.centralized
        if centralized:
            if load_path is None:
                self.net = RewardCNN(N, ns, na*N, hidden, n_hid, in_features)
            else:
                self.net = RewardNetTF(N, prevN, load_path, ns, na*N, hidden)
        else:
            if load_path is None:
                self.net = RewardCNN(N, ns, na, hidden, n_hid, in_features)
            else:
                self.net = RewardNetTF(N, prevN, load_path, ns, na, hidden)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.na = na
        self.name = 'RewardCNNAgent'
    
    # Idk how to implement this... randomly sample a bunch of possible actions and then pick the best one?
    def select_action(self, state, **kwargs):
        num_sample = kwargs.get('num_sample', 50)
        if self.centralized:
            # If centralized, technically we would need to multiply the potential action sample amount by 2^N.
            ### TODO: Implement a better method for finding the optimal action in this case.
            num_sample *= self.N*2
        action_space = kwargs.get('action_space', [-1,1])
        with torch.no_grad():
            if self.centralized:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.na*self.N).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
            else:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.na).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
            rewards = self.net(state.expand(num_sample, -1, -1), actions).view(-1)
            bestind = rewards.max(0)[1]
#             print(bestind, actions)
            return actions[bestind.detach()].numpy()#detach()
    
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch[0]))
#         print(B, len(batch[0]), len(batch), len(batch.action))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train()
#         print("State batch shape = ", state_batch.view(B, -1, self.N).shape, "; action shape = ", action_batch.view(B, -1, 1).shape)
        pred_reward = self.net(
            state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1)
        ) # Shape is (B,ns,N) and (B,na) for input and (B, ) for output
#         print("Reward batch shape = ", reward_batch.shape, "; prediction shape = ", pred_reward.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        print("RCNN loss: ", loss)
#         print("First layer gradients before backward: ", self.net.RNlayers[-1].weight.grad)
#         print("Last layer gradients before backward: ", self.net.RNlayers[0].weight.grad)
        loss.backward()
#         print("First    layer gradients after backward: ", torch.mean(self.net.RNlayers[-1].weight.grad))
        print("4th Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[3].weight.grad))
        print("3rd Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[2].weight.grad))
        print("2nd Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[1].weight.grad))
        print("The Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[0].weight.grad))
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
        
from torchvision.models.resnet import Bottleneck
# Agent for leaning reward
class RewardRNAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, n_hid=2, in_features=1, learning_rate=0.01, centralized=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = self.centralized
        if centralized:
            if load_path is None:
                self.net = RewardResNet(N, Bottleneck,[2, 2, 2, 2], num_classes=na,
                                        ns=ns, na=na*N, hidden=hidden, n_hid=n_hid, in_features=in_features)
        else:
            if load_path is None:
                self.net = RewardResNet(N, Bottleneck,[2, 2, 2, 2], num_classes=na,
                                        ns=ns, na=na, hidden=hidden, n_hid=n_hid, in_features=in_features)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.na = na
        self.name = 'RewardRNAgent'
    
    # Idk how to implement this... randomly sample a bunch of possible actions and then pick the best one?
    def select_action(self, state, **kwargs):
        num_sample = kwargs.get('num_sample', 50)
        if self.centralized:
            # If centralized, technically we would need to multiply the potential action sample amount by 2^N.
            ### TODO: Implement a better method for finding the optimal action in this case.
            num_sample *= self.N*2
        action_space = kwargs.get('action_space', [-1,1])
        with torch.no_grad():
            if self.centralized:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.na*self.N).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
            else:
                actions = torch.from_numpy( 
                        np.random.rand(num_sample, self.na).astype('float32')
                    ) * (action_space[1]-action_space[0])+action_space[0]
            rewards = self.net(state.expand(num_sample, -1, -1), actions).view(-1)
            bestind = rewards.max(0)[1]
#             print(bestind, actions)
#             print(bestind,actions, actions.shape, rewards, rewards.shape)
            return actions[bestind.detach()].numpy()#detach()
    
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch[0]))
#         print(B, len(batch[0]), len(batch), len(batch.action))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train()
#         print("State batch shape = ", state_batch.view(B, -1, self.N).shape, "; action shape = ", action_batch.view(B, -1, 1).shape)
        pred_reward = self.net(
            state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1)
        ) # Shape is (B,ns,N) and (B,na) for input and (B, ) for output
#         print("Reward batch shape = ", reward_batch.shape, "; prediction shape = ", pred_reward.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        print("RCNN loss: ", loss)
#         print("First layer gradients before backward: ", self.net.RNlayers[-1].weight.grad)
#         print("Last layer gradients before backward: ", self.net.RNlayers[0].weight.grad)
        loss.backward()
#         print("First    layer gradients after backward: ", torch.mean(self.net.RNlayers[-1].weight.grad))
#         print("4th Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[3].weight.grad))
#         print("3rd Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[2].weight.grad))
#         print("2nd Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[1].weight.grad))
#         print("The Last layer gradients after backward: ", torch.mean(self.net.RCNNlayers[0].weight.grad))
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
        
# Agent for leaning action by multiplying it with reward...?
class RewardActionAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], learning_rate=0.01, centralized=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
#         if centralized:
#             pass
#         else:
#             self.net = ActionNet(N, ns, na, hidden, action_range)
        if load_path is None:
            self.net = ActionNet(N, ns, na, hidden, action_range)
        else:
            self.net = ActionNetTF(N, prevN, load_path, ns, na, hidden, action_range)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.name = 'RewardActionAgent'
        
    # Picks an action based on given state
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            return self.net(state.view(1,-1,self.N)).squeeze().detach().numpy()# # Expected size: (B=1, na)?
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        pred_action = self.net(state_batch.view(B, -1, self.N)) # Shape should be (B,na)??

        # Find loss & optimize the model
        self.net.train()
        self.optimizer.zero_grad()
        loss = (pred_action * reward_batch.view(B,-1)).sum()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
        
# Actor-Critic attempt #1
# Properties: Directly estimates action without sampling around. 
#             Directly updates Reward based on state and action without considering the future.
#             Is useless.
class AC1Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rateA=0.01, learning_rateC=0.02, centralized=False, centralizedA=False,
                 prevN=10, load_pathA=None, load_pathC=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = centralizedA
        
        # Load models
        if load_pathA is None:
            self.netA = ActionNet(N, ns, na, hidden, action_range)
        else:
            self.netA = ActionNetTF(N, prevN, load_pathA, ns, na, hidden, action_range)
            
        if load_pathC is None:
            self.netC = RewardNet(N, ns, na, hidden)
        else:
            self.netC = RewardNetTF(N, prevN, load_pathC, ns, na, hidden)
        self.optimizerA = torch.optim.RMSprop(self.netA.parameters(), lr=learning_rateA)
        self.optimizerC = torch.optim.RMSprop(self.netC.parameters(), lr=learning_rateC)
        self.schedulerA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerA)
        self.schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerC)
        self.na = na
        self.name = 'AC1Agent'
        
    # Picks an action based on given state... similar to LearnerAgent that directly outputs an action.
    # In a future AC you could use RewardAgent's action selection instead.
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            return self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the models
        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        self.netA.train() # Actor and action decisions
        pred_action = self.netA(state_batch.view(B, -1, self.N)) # Input shape should be (B,no,N) and output be (B,na)
        self.netC.train() # Critic and value predictions
        pred_reward = self.netC( state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1) )

        lossA = torch.nn.functional.mse_loss(action_batch, pred_action)
        lossC = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        lossA.backward()
        lossC.backward()
        self.optimizerA.step()
        self.optimizerC.step()
        
    # Overwrite original because there are two nets now
    def set_train(self, train):
        if train:
            self.netA.train()
            self.netC.train()
        else:
            self.netA.eval()
            self.netC.eval()

# Actor-Critic attempt #2
# Properties: Chooses action using a net where loss is defined as negative predicted reward from Critic. 
#             Directly updates Reward based on state and action without considering the future.
# The argument "mode" indicates which version of AC2 is implemented. If using any of them, please use non-cumulative rewards.
class AC2Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], add_noise=False, rand_modeA=NO_RAND, 
                 learning_rateA=0.01, learning_rateC=0.02, centralized=False, centralizedA=False, neg_loss_sign=False,
                 prevN=10, load_pathA=None, load_pathC=None, mode=0, gamma=0.98):
        super().__init__(device, N)
        self.noise = add_noise
        self.centralized = centralized
        self.centralizedA = centralizedA
        self.rand_modeA = rand_modeA
        self.action_range = action_range
        
        # Load models for transfer learning. If you just want to use an existing model and see results, consider self.load_models().
        if self.centralized:
            if load_pathA is None:
                self.netA = ActionNet(N, ns, na*N, hidden, action_range, rand_mode=rand_modeA)
            else:
                self.netA = ActionNetTF(N, prevN, load_pathA, ns, na*N, hidden, action_range, rand_mode=rand_modeA)
            
            if load_pathC is None:
                self.netC = RewardNet(N, ns, na*N, hidden)
            else:
                self.netC = RewardNetTF(N, prevN, load_pathC, ns, na*N, hidden)
        else:
            if load_pathA is None:
                self.netA = ActionNet(N, ns, na, hidden, action_range, rand_mode=rand_modeA)
            else:
                self.netA = ActionNetTF(N, prevN, load_pathA, ns, na, hidden, action_range, rand_mode=rand_modeA)

            if load_pathC is None:
                self.netC = RewardNet(N, ns, na, hidden)
            else:
                self.netC = RewardNetTF(N, prevN, load_pathC, ns, na, hidden)
        
        self.optimizerA = torch.optim.RMSprop(self.netA.parameters(), lr=learning_rateA)
        self.optimizerC = torch.optim.RMSprop(self.netC.parameters(), lr=learning_rateC)
        self.schedulerA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerA)
        self.schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerC)
        self.na = na
        self.mode = mode
        self.gamma = gamma
        if neg_loss_sign:
            self.loss_sign = -1
        else:
            self.loss_sign = 1
        self.name = 'AC2Agent'
        
    # Picks an action based on given state... similar to LearnerAgent that directly outputs an action.
    # In a future AC you could use RewardAgent's action selection instead.
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            if self.centralized:
                if self.rand_modeA == NO_RAND:
                    return self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
                elif self.rand_modeA == GAUSS_RAND:
                    # Should I take a sample, or should I just return the mean value?
                    not_use_rand = kwargs.get('rand', False) # Using a reverse logic here to avoid modifying old code
                    if not_use_rand:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy().reshape((self.N,-1))
                        return distrb[:,:self.na]
                    else:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze().view(self.N,-1)
#                         print(distrb, distrb.shape)
                        distrb = torch.distributions.Normal(
                            distrb[:,:self.na],
                            nn.functional.softplus( distrb[:,self.na:] ) 
                        )
                        return torch.clamp( distrb.sample(), 
                                           self.action_range[0], self.action_range[1] ).squeeze().detach().numpy()
            else:
                if self.rand_modeA == NO_RAND:
                    return self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
                elif self.rand_modeA == GAUSS_RAND:
                    # Should I take a sample, or should I just return the mean value?
                    not_use_rand = kwargs.get('rand', False) # Using a reverse logic here to avoid modifying old code
                    if not_use_rand:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
                        return distrb[:self.na]
                    else:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze()
                        distrb = torch.distributions.Normal(
                            distrb[:self.na],
                            nn.functional.softplus( distrb[self.na:] ) 
                        )
                        return torch.clamp( distrb.sample(), 
                                           self.action_range[0], self.action_range[1] ).squeeze().detach().numpy()
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        ### TODO: BatchNorm is known to introduce issue when batch size is 1. Find a better way to solve this, instead
        # of using the simple method below.
        if B <= 1:
            print("I didn't learn anything!")
            return
#         B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss for Critic
        self.netC.train() # Critic and value predictions
        self.optimizerC.zero_grad()
        pred_reward = self.netC( state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1) )
        if self.mode == 1208:
            # Calculate Q_c( s(t+1), a(t+1) )
            next_pred_reward = self.netC( next_state_batch.view(B, -1, self.N), torch.zeros_like(action_batch.view(B, -1, 1)) )
#             lossC = torch.nn.functional.mse_loss(reward_batch.squeeze(), next_pred_reward * self.gamma - pred_reward)
            lossC = torch.nn.functional.mse_loss(reward_batch.unsqueeze(1), next_pred_reward * self.gamma - pred_reward)
#             print(next_pred_reward.shape, pred_reward.shape, reward_batch.shape, lossC.shape)
        elif self.mode == 1209:
            # Calculate Q_c( s(t+1), a(t+1) ).
            next_pred_reward = self.netC( next_state_batch.view(B, -1, self.N), torch.zeros_like(action_batch.view(B, -1, 1)) )
            # Find loss. Not sure if the squeeze() placements below are correct...
            lossC = reward_batch.squeeze() - (next_pred_reward * self.gamma - pred_reward)
            lossC *= pred_reward
            lossC = torch.abs(lossC).mean()
        else:
            lossC = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
#         print("Critic loss: ", lossC)
        lossC.backward()
#         print("Last layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
#         print(self.netC.RNlayers[0].weight.grad)
#         print("Last  layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
#         print("Mid   layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[1].weight.grad))
#         print("Front layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[2].weight.grad))
        self.optimizerC.step()
        
        # Find loss for Actor
        self.netA.train() # Actor and action decisions
        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        
        # Freeze Critic?
        for nfc in self.netC.RNlayers:
            try:
                nfc.weight.requires_grad = False
                nfc.bias.requires_grad = False
            except:
                pass
            # try-catch exists because activation layers don't have such fields
        # Eval critic? 
        self.netC.eval()

        if self.centralized:
            if self.rand_modeA == NO_RAND:
                pred_action = self.netA(state_batch.view(B, -1, self.N)).view(B,self.N,-1) # Input: (B,no,N) and output: (B,N,na) ***(Should I use transpose instead?)***
                if self.noise:
                    stddev = 0.1
                    added_noise = Variable(torch.randn(pred_action.size()) * stddev)
                lossA = -self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).mean() # -??? +???
            elif self.rand_modeA == GAUSS_RAND:
                # Need to create a Gaussian distribution out of the given parameters (make sure stdev>0)... 
                # and have to multiply log of likelihood for the outcome.
                distrb_params = self.netA(state_batch.view(B, -1, self.N)).view(B,self.N,-1) # Shape would be (B,N,na*2)
#                 print(distrb_params.shape, nn.functional.softplus( distrb_params[:,:,self.na:] ) )
                distrb = torch.distributions.Normal(
                    distrb_params[:,:,:self.na],
                    nn.functional.softplus( distrb_params[:,:,self.na:] )
                )
                # Need to keep action within limits
                pred_action = distrb.sample()
                pred_probs = distrb.log_prob(pred_action)
                pred_action = torch.clamp( pred_action, self.action_range[0], self.action_range[1] )
#                 print(self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, self.N, self.na) ).shape, pred_probs.shape)
                lossA = self.loss_sign * self.netC(state_batch.view(B, -1, self.N), pred_action).unsqueeze(2) * pred_probs
                lossA = lossA.mean()
        else:
            if self.rand_modeA == NO_RAND:
                pred_action = self.netA(state_batch.view(B, -1, self.N)) # Input shape should be (B,no,N) and output be (B,na)
                if self.noise:
                    # Add Gaussian noise. https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694
                    # Note that if you need to generate noise within Network class, then you need to use model.training
                    # to see if you're evaluating or training to decide if you want to add the noise to the output.
                    stddev = 0.1
                    added_noise = Variable(torch.randn(pred_action.size()) * stddev)
                lossA = -self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).mean() # -??? or +???
            elif self.rand_modeA == GAUSS_RAND:
                # Need to create a Gaussian distribution out of the given parameters (make sure stdev>0)... 
                # and have to multiply log of likelihood for the outcome.
                distrb_params = self.netA(state_batch.view(B, -1, self.N)) # Shape would be (B,na*2)
    #             pred_action = torch.zeros(B,self.na)
    #             pred_probs = torch.zeros(B)
    #             for i in range(B):
    #                 # ref for implementation: 
    #                 # https://discuss.pytorch.org/t/actor-critic-with-multivariate-normal-network-weights-fail-to-update/74548/2
    #                 # https://pytorch.org/docs/stable/distributions.html#multivariatenormal
    #                 distrb = torch.distributions.multivariate_normal.MultivariateNormal(
    #                 # distrb = torch.distributions.Normal(
    #                     distrb_params[i,:self.na],
    #                     covariance_matrix=torch.diag( nn.functional.softplus( distrb_params[i,self.na:] ) )
    #                 )
    #                 # Need to keep action within limits
    #                 pred_action[i,:] = torch.clamp( distrb.sample(), self.action_range[0], self.action_range[1] )
    #                 pred_probs[i] = distrb.log_prob(pred_action[i,:])

                # https://stackoverflow.com/a/62933292
                distrb = torch.distributions.Normal(
                    distrb_params[:,:self.na],
                    torch.diag( nn.functional.softplus( distrb_params[:,self.na:] ) )
                )
                # Need to keep action within limits
                pred_action = distrb.sample()
    #             print(pred_action)
                pred_probs = distrb.log_prob(pred_action)
                pred_action = torch.clamp( pred_action, self.action_range[0], self.action_range[1] )

                ### !!!!!!! THE CORRECT ACTOR_CRITIC SHOULD USE THE ADVANTAGE, NOT THE REWARD !!!!!!!! FIXING THIS WITH AC3Agent
                ## Problem with the above is that our Critic needs action as part of the input, and we might have
                ## issue accessing the next state's reward (actual or predicted).
                lossA = self.loss_sign * self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)) * pred_probs
                lossA = lossA.mean()

#         lossA = (-self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)) * pred_action).mean()
#         print("Actor loss = reward: ", lossA)
#         print(-self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).detach())

        lossA.backward()

#         print("Last  layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
#         print("Mid   layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[3].weight.grad))
#         print("Front layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[6].weight.grad))
# #         print(self.netC.RNlayers[0].weight.grad)
#         print("Last  layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[0].weight.grad))
#         print("Mid   layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[2].weight.grad))
#         print("Front layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[4].weight.grad))
# #         print(self.netA.ANlayers[0].weight.grad)

        self.optimizerA.step()
    
        self.losses.append(lossC.detach().numpy())
        self.lossesA.append(lossA.detach().numpy())
    
        # UnFreeze Critic?
        for nfc in self.netC.RNlayers:
            try:
                nfc.weight.requires_grad = True
                nfc.bias.requires_grad = True
            except:
                pass
        # UnEval critic? 
        self.netC.train()
        
    # Overwrite original because there are two nets now
    def set_train(self, train):
        if train:
            self.netA.train()
            self.netC.train()
        else:
            self.netA.eval()
            self.netC.eval()
    
    def save_model(self, suffix="", agent_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if len(suffix) <= 0:
            suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if agent_path is None:
            agent_path = "models/{}_{}".format(self.name, suffix)
        print('Saving model to {}'.format(agent_path))
        torch.save(self.netA.state_dict(), agent_path+'_A')
        torch.save(self.netC.state_dict(), agent_path+'_C')

    def load_model(self, agent_path):
        print('Loading model from {}'.format(agent_path))
        if agent_path is not None:
            self.netA.load_state_dict(torch.load(agent_path+'_A'))
            self.netC.load_state_dict(torch.load(agent_path+'_C'))

# Actor-Critic attempt #3
# Properties: Chooses action using negative advantage. 
#             Updates Reward based on state and action assuming values incorporate the future.
# To do this change, we have to use online training instead to provide immediate reward per action. 
# Alternatively, use the next state? to estimate the value???
#    - Another problem with this: The Critic also needs action to output a value. Should I change the network??
# https://github.com/nikhilbarhate99/Actor-Critic-PyTorch/blob/01c833e83006be5762151a29f0719cc9c03c204d/model.py#L33
# http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf
# The argument "mode" is here to implement different update functions. When using them, it might be advised to train with
# normalized but non-cumulative rewards.
class AC3Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], add_noise=False, rand_modeA=NO_RAND, 
                 learning_rateA=0.01, learning_rateC=0.02, centralized=False, centralizedA=False, neg_loss_sign=True,
                 prevN=10, load_pathA=None, load_pathC=None, mode=0, gamma=0.98):
        super().__init__(device, N)
        self.noise = add_noise
        self.centralized = centralized
        self.centralizedA = centralizedA
        self.rand_modeA = rand_modeA
        self.action_range = action_range
        
        # Load models
        if centralized:#A:
            if load_pathA is None:
                self.netA = ActionNet(N, ns, na*N, hidden, action_range, rand_mode=rand_modeA)
            else:
                self.netA = ActionNetTF(N, prevN, load_pathA, ns, na*N, hidden, action_range, rand_mode=rand_modeA)
        else:
            if load_pathA is None:
                self.netA = ActionNet(N, ns, na, hidden, action_range, rand_mode=rand_modeA)
            else:
                self.netA = ActionNetTF(N, prevN, load_pathA, ns, na, hidden, action_range, rand_mode=rand_modeA)

        if centralized:
            if load_pathC is None:
                self.netC = RewardNet(N, ns, na*N, hidden)
            else:
                self.netC = RewardNetTF(N, prevN, load_pathC, ns, na*N, hidden)
        else:
            if load_pathC is None:
                self.netC = RewardNet(N, ns, na, hidden)
            else:
                self.netC = RewardNetTF(N, prevN, load_pathC, ns, na, hidden)
                
        self.optimizerA = torch.optim.RMSprop(self.netA.parameters(), lr=learning_rateA)
        self.optimizerC = torch.optim.RMSprop(self.netC.parameters(), lr=learning_rateC)
        self.schedulerA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerA)
        self.schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerC)
        self.ns = ns
        self.na = na
        self.mode = mode
        self.gamma = gamma
        if neg_loss_sign:
            self.loss_sign = -1
        else:
            self.loss_sign = 1
        self.name = 'AC3Agent'
        
    # Picks an action based on given state... similar to LearnerAgent that directly outputs an action.
    # In a future AC you could use RewardAgent's action selection instead.
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            if self.rand_modeA == NO_RAND:
                if self.centralized:
                    return self.netA(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy().reshape((self.N,-1))
                else:
                    return self.netA(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy()
                # Expected size: (B=1, na, N) -> (na,N)?
            elif self.rand_modeA == GAUSS_RAND:
                # Should I take a sample, or should I just return the mean value?
                not_use_rand = kwargs.get('rand', False) # Using a reverse logic here to avoid modifying old code
                if not_use_rand:
                    if self.centralized:
                        distrb = self.netA(
                            state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().detach().numpy().reshape((self.N,-1))
                        return distrb[:,:self.na]
                    else:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
                        return distrb[:self.na]
                else:
                    if self.centralized:
                        distrb = self.netA(state.view(1,-1,self.N)[:,:self.ns,:]).squeeze().view(self.N,-1)
                        distrb = torch.distributions.Normal(
                            distrb[:,:self.na],
                            nn.functional.softplus( distrb[:,self.na:] ) 
                        )
                        return torch.clamp( distrb.sample(), self.action_range[0], self.action_range[1] ).squeeze().detach().numpy()
                    else:
                        distrb = self.netA(state.view(1,-1,self.N)).squeeze()
                        distrb = torch.distributions.Normal(
                            distrb[:self.na],
                            nn.functional.softplus( distrb[self.na:] ) 
                        )
                        return torch.clamp( distrb.sample(), self.action_range[0], self.action_range[1] ).squeeze().detach().numpy()
#             return self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        ### TODO: BatchNorm is known to introduce issue when batch size is 1. Find a better way to solve this, instead
        # of using the simple method below.
        if B <= 1:
            print("I didn't learn anything!")
            return
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))
        inst_reward_batch = torch.from_numpy(np.asarray(batch.inst_reward).astype('float32'))
        next_state_batch = torch.cat(batch.next_state)

        # Find loss for Critic
        self.netC.train() # Critic and value predictions
        self.optimizerC.zero_grad()
        pred_reward = self.netC( state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1) )
        next_pred_reward = self.netC( next_state_batch.view(B, -1, self.N), torch.zeros_like(action_batch.view(B, -1, 1)) )
        # lossC = torch.nn.functional.mse_loss(reward_batch.unsqueeze(1), next_pred_reward * self.gamma - pred_reward)
        lossC = torch.nn.functional.mse_loss(inst_reward_batch.unsqueeze(1), pred_reward - next_pred_reward * self.gamma)
#         lossC = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
#         print("Critic loss: ", lossC)
        lossC.backward()
#         print("Last layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
#         print(self.netC.RNlayers[0].weight.grad)
        self.optimizerC.step()
        
        # Find loss for Actor
        self.netA.train() # Actor and action decisions
        self.optimizerA.zero_grad()
        
        # Eval critic
        self.netC.eval()

        if self.centralized:
            if self.rand_modeA == NO_RAND:
                pred_action = self.netA(state_batch.view(B, -1, self.N)[:,:self.ns,:]).view(B,self.N,-1) # Input (B,no,N); output (B,N,na) (Should I use transpose instead?)
                if self.centralizedA:
                    pred_action = pred_action.view(B,self.na,self.N) 
                if self.noise:
                    # Add Gaussian noise. 
                    stddev = 0.1
                    added_noise = Variable(torch.randn(pred_action.size()) * stddev)
                # Modify later... Sign??? (default is -, for now?)
                lossA = ( self.netC(next_state_batch.view(B, -1, self.N), pred_action) - inst_reward_batch ).mean()
            elif self.rand_modeA == GAUSS_RAND:
                # TODO: Add centralized differentiations later
                if self.mode != 1204:
                    distrb_params = self.netA(state_batch.view(B, -1, self.N)).view(B,self.N,-1) # Shape would be reshaped from (B,N*na*2) into (B,N,na*2)
                else:
                    distrb_params = self.netA(next_state_batch.view(B, -1, self.N)).view(B,self.N,-1)
                distrb = torch.distributions.Normal(
                    distrb_params[:,:,:self.na],
                    nn.functional.softplus( distrb_params[:,:,self.na:] )
                )
                # Need to keep action within limits
                pred_action = distrb.sample()
                pred_probs = distrb.log_prob(pred_action)
                pred_action = torch.clamp( pred_action, self.action_range[0], self.action_range[1] ) # (B,N,na)

                # self.netC( ) results in shape (B,1). Reward_batch has shape (B,), and needs to be expanded to avoid generating a (128,128) thing.
    #             advantage = self.netC(next_state_batch.view(B, -1, self.N), pred_action) - reward_batch.unsqueeze(1)
                if self.mode == 1208:
                    pred_probs = distrb.log_prob(action_batch)
                    advantage = self.netC(state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() - self.netC(next_state_batch.view(B, -1, self.N), 
                                                                             torch.zeros_like(pred_action)).squeeze()
                elif self.mode == 1205:
                    advantage = self.netC(state_batch.view(B, -1, self.N), pred_action).squeeze() - reward_batch # inst_reward_batch # Idea: This mode just updates by finding advantage using the real cumulative reward
                elif self.mode == 1204:
                    advantage = self.netC(next_state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() + inst_reward_batch - self.netC(state_batch.view(B, -1, self.N), 
                                                                                            action_batch).squeeze()
                elif self.mode == 0:
                    # Idea: The first term is the expected value of the current action.
                    #       The latter two terms sums up to the the (hopefully accurate) expected value of previous action. 
                    advantage = self.netC(state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() - inst_reward_batch - self.netC(
                                                                        next_state_batch.view(B, -1, self.N), 
                                                                        torch.zeros_like(pred_action)
                                                                                                ).squeeze()*self.gamma
                else:
                    # Legacy update rule used by experiments prior to 1208 (at least)
                    advantage = self.netC(next_state_batch.view(B, -1, self.N), pred_action).squeeze() - inst_reward_batch 
                # advantage is going to be a scalar after this, just like the reward and Critic outputs, for centralized ones.
                lossA = self.loss_sign * pred_probs * advantage.unsqueeze(1).unsqueeze(2)
                lossA = lossA.mean()
        elif not self.centralized:
            if self.rand_modeA == NO_RAND:
                pred_action = self.netA(state_batch.view(B, -1, self.N)[:,:self.ns,:]) # Input shape should be (B,no,N) and output be (B,na)
                if self.centralizedA:
                    pred_action = pred_action.view(B,self.na,self.N) 
                if self.noise:
                    # Add Gaussian noise. 
                    stddev = 0.1
                    added_noise = Variable(torch.randn(pred_action.size()) * stddev)
                # Modify later
                lossA = ( self.netC(next_state_batch.view(B, -1, self.N), pred_action) - inst_reward_batch ).mean() # Sign??? (default is -, for now?)
            elif self.rand_modeA == GAUSS_RAND:
                # TODO: Add centralized differentiations later
                if self.mode != 1204:
                    distrb_params = self.netA(state_batch.view(B, -1, self.N)) # Shape would be (B,na*2)
                else:
                    distrb_params = self.netA(next_state_batch.view(B, -1, self.N))
    #             pred_action = torch.zeros(B,self.na)
    #             pred_probs = torch.zeros(B)
                distrb = torch.distributions.Normal(
                    distrb_params[:,:self.na],
                    nn.functional.softplus( distrb_params[:,self.na:] )
    #                 torch.diag( nn.functional.softplus( distrb_params[:,self.na:] ) )
                )
                # Need to keep action within limits
                pred_action = distrb.sample()
                pred_probs = distrb.log_prob(pred_action)
                pred_action = torch.clamp( pred_action, self.action_range[0], self.action_range[1] )

                # self.netC( ) results in shape (B,1). Reward_batch has shape (B,), and needs to be expanded to avoid generating a (128,128) thing.
    #             advantage = self.netC(next_state_batch.view(B, -1, self.N), pred_action) - reward_batch.unsqueeze(1)
                if self.mode == 1208:
                    pred_probs = distrb.log_prob(action_batch)
                    advantage = self.netC(state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() - self.netC(next_state_batch.view(B, -1, self.N), 
                                                                             torch.zeros_like(pred_action)).squeeze()
                elif self.mode == 1205:
                    advantage = self.netC(state_batch.view(B, -1, self.N), pred_action).squeeze() - inst_reward_batch
                elif self.mode == 1204:
                    ### TODO: Problem - are you epxecting reward_batch to be instantaneous, or to be cumulative? The Critic is
                    ### trained on the assumption that it's cumulative, but the second term below assumes it's instantaneous???
                    advantage = self.netC(next_state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() + inst_reward_batch - self.netC(state_batch.view(B, -1, self.N), 
                                                                                            action_batch).squeeze()
                elif self.mode == 0:
                    # Idea: The first term is the expected value of the current action.
                    #       The latter two terms sums up to the the (hopefully accurate) expected value of previous action. 
                    advantage = self.netC(state_batch.view(B, -1, self.N), 
                                          pred_action).squeeze() - inst_reward_batch - self.netC(
                                                                        next_state_batch.view(B, -1, self.N), 
                                                                        torch.zeros_like(pred_action)
                                                                                                ).squeeze()*self.gamma
                else:
                    # Legacy update rule used by experiments prior to 1208 (at least)
                    advantage = self.netC(next_state_batch.view(B, -1, self.N), pred_action).squeeze() - inst_reward_batch #reward_batch
                lossA = self.loss_sign * pred_probs * advantage.unsqueeze(1)
                lossA = lossA.mean()
        else:
#         # Here comes the fun part... the centralized and decentralized Critic would expect differently-shaped inputs...
# #         if self.centralized:
# #             print("Option unsupported! Centralized Critic wants to take in the environment state, but Actor needs judgement based")
# #             # If centralized, then it would want all actions...
# #             if self.centralizedA:
# #                 # Which is convenient in this scenario.
# #                 lossA = ( self.netC(next_state_batch.view(B, -1, self.N), pred_action) - reward_batch ).mean()
# #             else:
# #                 # Where we need to explicitly put things together...
# #                 pass
# # #                 torch.cat()
# #         else:
# #             # If it wants to take things one by one... then it's in luck if Actor isn't centralized.
# #             if self.centralizedA:
# #                 # Do it piece by piece
# #                 pass
# #             else:
# #         #         lossA = ( self.netC(next_state_batch.view(B, -1, self.N), torch.zeros((B, self.na, 1))) - reward_batch ).mean()
# #                 lossA = ( self.netC(next_state_batch.view(B, -1, self.N), pred_action) - reward_batch ).mean()
            pass
#         lossA = (-self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)) * pred_action).mean()
#         print("Actor loss = reward: ", lossA)
#         print(-self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).detach())
        lossA.backward()
#         print("Last  layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
#         print("Mid   layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[1].weight.grad))
#         print("Front layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[2].weight.grad))
#         print(self.netC.RNlayers[0].weight.grad)
#         print("Last  layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[0].weight.grad))
#         print("Mid   layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[1].weight.grad))
#         print("Front layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[2].weight.grad))
#         print(self.netA.ANlayers[0].weight.grad)
        self.optimizerA.step()
    
        self.losses.append(lossC.detach().numpy())
        self.lossesA.append(lossA.detach().numpy())
        
        # UnEval critic
        self.netC.train()
        
    # Overwrite original because there are two nets now
    def set_train(self, train):
        if train:
            self.netA.train()
            self.netC.train()
        else:
            self.netA.eval()
            self.netC.eval()
    
    def save_model(self, suffix="", agent_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if len(suffix) <= 0:
            suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if agent_path is None:
            agent_path = "models/{}_{}".format(self.name, suffix)
        print('Saving model to {}'.format(agent_path))
        torch.save(self.netA.state_dict(), agent_path+'_A')
        torch.save(self.netC.state_dict(), agent_path+'_C')

    def load_model(self, agent_path):
        print('Loading model from {}'.format(agent_path))
        if agent_path is not None:
            self.netA.load_state_dict(torch.load(agent_path+'_A'))
            self.netC.load_state_dict(torch.load(agent_path+'_C'))

# Actor-Critic attempt #4, never finished due to:     
# A) If Actor and Critic don’t share layers, then the gradient that trickles back from Critic’s value prediction to Actor’s first layer would be so small that learning is unobservable. 
# B) If Actor and Critic share layers, then Critic can no longer take in an action, and I can’t find a way to incorporate the Actor output in loss calculation when we use deterministic policy (remember that we can’t use stochastic ones, because action space is continuous).  
# C) One possible way out is to still allow Critic to take action value as input. The necessary modification is now to send both state and action to the first layer; when using Actor, the action input can be set to 0; when using Critic, both are used; during training, we train by updating Critic first, and then clear all gradients and stuff, and then update the Actor; when evaluating the Actor’s action, we would need to pass it into the input, along with the next state from the batch, so that the first shared layer can get some gradient... sounds nasty. Not doing that.  
# Properties: Chooses action using negative advantage. 
#             Updates Reward based on state and action assuming values incorporate the future.
#             Actor and Critic share one layer.
# To do this change, we have to use online training instead to provide immediate reward per action. 
# Alternatively, use the next state? to estimate the value???
# Also, how is the action value supposed to be incorporated into the loss??
class AC4Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rate=0.01, centralized=False, centralizedA=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
        self.centralizedA = centralizedA
        
        # Load models
        if load_path is None:
            self.net = ActorCriticNet(N, ns, na, hidden, action_range)
        else:
            self.net = ActorCriticNet(N, prevN, load_pathA, ns, na, hidden, action_range)
        
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate) # Or separate them?
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.na = na
        self.name = 'AC4Agent'
        
    # Picks an action based on given state... similar to LearnerAgent that directly outputs an action.
    # In a future AC you could use RewardAgent's action selection instead.
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            return self.net(state.view(1,-1,self.N))[0].squeeze().detach().numpy()
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))
        next_state_batch = torch.cat(batch.next_state)

        self.net.train() 
        self.optimizer.zero_grad()
        # Find loss for Critic and Actor
        pred_action, pred_reward = self.net( state_batch.view(B, -1, self.N) )
        lossC = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        # Update C first? And then do zero_grad again? 
#         lossA = ( self.netC(next_state_batch.view(B, -1, self.N)) - reward_batch ).sum()
        lossA = ( self.netC(next_state_batch.view(B, -1, self.N)) - reward_batch ).sum()
        
        # Find loss for Actor
        # Freeze Critic?
        # for nfc in self.net.RNlayers:
        #     nfc.weight.requires_grad = False
        #     nfc.bias.requires_grad = False

#         lossA = -self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).mean()
        lossA = ( self.netC(next_state_batch.view(B, -1, self.N), torch.zeros((B, self.na, 1))) - reward_batch ).mean()
        lossA.backward()
        self.optimizerA.step()
        # UnFreeze Critic?
        # for nfc in self.net.RNlayers:
        #     nfc.weight.requires_grad = False
        #     nfc.bias.requires_grad = False
        
            
# DDPG attempt
# References: https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py, model.py, util.py
# Properties: Chooses action using a net where loss is defined as negative predicted reward from Critic. 
#             Updates Reward based on state and action while [NOT considering the future state (with discounts) for now].
#             Uses two pairs of nets - each pair containing one target network.
# Should this be trained without using rewards that are already cumulative in the future?
class DDPGAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rateA=0.01, learning_rateC=0.02, tau=0.1, centralized=False, centralizedA=False,
                 prevN=10, load_pathA=None, load_pathC=None):
        super().__init__(device, N)
        self.tau = tau
        self.centralized = centralized
        self.centralizedA = centralizedA
        
        # Load models
        if load_pathA is None:
            self.netA = ActionNet(N, ns, na, hidden, action_range)
            self.netAT = ActionNet(N, ns, na, hidden, action_range)
        else:
            self.netA = ActionNetTF(N, prevN, load_pathA, ns, na, hidden, action_range)
            self.netAT = ActionNetTF(N, prevN, load_pathA, ns, na, hidden, action_range)
            
        if load_pathC is None:
            self.netC = RewardNet(N, ns, na, hidden)
            self.netCT = RewardNet(N, ns, na, hidden)
        else:
            self.netC = RewardNetTF(N, prevN, load_pathC, ns, na, hidden)
            self.netCT = RewardNetTF(N, prevN, load_pathC, ns, na, hidden)
            
        # Create hard copies from the reference:
        for target_param, param in zip(self.netAT.parameters(), self.netA.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.netCT.parameters(), self.netC.parameters()):
            target_param.data.copy_(param.data)
#         self.netAT.eval()
#         self.netCT.eval() # Not sure if this would stop it from finding loss values
            
        self.optimizerA = torch.optim.RMSprop(self.netA.parameters(), lr=learning_rateA)
        self.optimizerC = torch.optim.RMSprop(self.netC.parameters(), lr=learning_rateC)
        self.schedulerA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerA)
        self.schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerC)
        self.na = na
        self.name = 'DDPGAgent'
        
    # Picks an action based on given state. Use the non-target Actor for this purpose.
    def select_action(self, state, **kwargs):
        with torch.no_grad():
            return self.netA(state.view(1,-1,self.N)).squeeze().detach().numpy()
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))
        
        # Find target network judgements
        target_next_reward = self.netCT(next_state_batch.view(B, -1, self.N), 
                                        self.netAT(next_state_batch.view(B, -1, self.N)) )
        # Add discounts later...
        target_reward = self.netCT(state_batch.view(B, -1, self.N), 
                                   self.netAT(state_batch.view(B, -1, self.N)) )

        # Find loss for Critic
        self.netC.train() # Critic and value predictions
        self.optimizerC.zero_grad()
        pred_reward = self.netC( state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1) )
        lossC = torch.nn.functional.mse_loss(target_reward.squeeze(), pred_reward.squeeze())
        lossC.backward()
        self.optimizerC.step()
        
        # Find loss for Actor using non-target Critic
        torch.autograd.set_detect_anomaly(True)
        self.netA.train() # Actor and action decisions
        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        state_batch__ = state_batch.view(B, -1, self.N)
        pred_action = self.netA(state_batch__) # Input shape should be (B,no,N) and output be (B,na)
        pred_action__ = pred_action.view(B, -1, 1)
        lossA = -self.netC(state_batch__, pred_action__)
        lossA__ = lossA.mean()
        lossA__.backward()
#         print("Actor loss = reward: ", lossA__)
#         print(-self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).detach())
        print("Last  layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[0].weight.grad))
        print("Mid   layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[1].weight.grad))
        print("Front layer Critic gradients after backward: ", torch.mean(self.netC.RNlayers[2].weight.grad))
#         print(self.netC.RNlayers[0].weight.grad)
        print("Last  layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[0].weight.grad))
        print("Mid   layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[1].weight.grad))
        print("Front layer Actor gradients after backward: ", torch.mean(self.netA.ANlayers[2].weight.grad))
#         print(self.netA.ANlayers[0].weight.grad)
        self.optimizerA.step() # Do this in a later step to avoid
        # issues like this: https://github.com/pytorch/pytorch/issues/39141#issuecomment-636881953
        
        
        self.losses.append(lossC.detach().numpy())
        self.lossesA.append(lossA__.detach().numpy())
        
        # Soft update (copied from reference)
        for target_param, param in zip(self.netAT.parameters(), self.netA.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - self.tau) + param.data * self.tau )
        for target_param, param in zip(self.netCT.parameters(), self.netC.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - self.tau) + param.data * self.tau )
        
    # Overwrite original because there are two nets now
    def set_train(self, train):
        if train:
            self.netA.train()
            self.netC.train()
        else:
            self.netA.eval()
            self.netC.eval()

# Agent for learning a gradient-based method.
# Let's only consider velocity action outputs for now... 
class GradientAgent(BaseAgent):
    def __init__(self, device, N, ns=2, hidden=24, action_range=[-1,1], learning_rate=0.01, centralized=False):
        super().__init__(device, N)
        self.centralized = centralized
        if centralized:
            pass
        else:
            self.net = EnergyNet(N, 2, hidden)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.needsExpert = True
        self.name = 'GradientAgent'
#         self.action_range = action_range
        self.range = action_range[1] - action_range[0]
        self.offset = 0.5*(action_range[0]+action_range[1])
        self.na = 2
    
    def getNextState(self, observed_state, action, dt=0.01):
        # Input: observed_state is a (M,2+,N) array, and action is (M,2). Each agent corresponds to a row.
        # Output: Supposedly, a (M,2,N) array that contains the expected new observed state.
        # Right now we're manually giving the agent a sense of its own dynamics with velocity input (single integrator)...
        # Hopefully it could go automatically soon.
        # Limitation: Doens't know if 0 means "not observed" or "already together". By default we assume the first. 
        # Thus, we first get a M*N matrix that records the neighbor information
        M,_,N = observed_state.shape
        is_neighbor = np.zeros((M,N))
        is_neighbor[ (observed_state[:,0,:]!=0) & (observed_state[:,1,:]!=0) ] = 1
        # Next, find the new relative distances based on the action. Assuming action means velocity, and we use a small dt.
        new_dists = observed_state[:,:2,:] - dt * action.reshape(M,2,1) # Broadcast action (M,2) to (M,2,N)
        # Filter out unobserved states
        new_dists *= is_neighbor.reshape(M,1,N)
        return new_dists
    
    def getEnergy(self, observed_state):
        # This is the expert that finds the energy function for each agent, but shhh don't let the rest know
        # Input: observed_state, expecting shape to be (N,no,N), as per the current environment, and be full of distance norms
        # Output: Probably (N,1), one for each agent. Let's make it only dependent on position for now; use velocity later.
        sum_dists = np.sum(np.linalg.norm(observed_state[:,:2,:], ord=2, axis=1), axis=1)
        return sum_dists
    
    # Picks an action based on given state.
    def select_action(self, state, **kwargs):
        # Input shape: state has shape (ns,N)
        with torch.no_grad():
            # Find gradients in both action space directions
            dt = 0.01
            da = 0.1
            sample_ax1p = self.getNextState(state.view(1,-1,self.N).detach().numpy(), np.array([[da,0]]).astype('float32'), dt) # Now shape: (1,2,N)
            sample_ax1n = self.getNextState(state.view(1,-1,self.N).detach().numpy(), np.array([[-da,0]]).astype('float32'), dt)
            sample_ax1p = self.net(torch.from_numpy(sample_ax1p)).squeeze().detach().numpy()
            sample_ax1n = self.net(torch.from_numpy(sample_ax1n)).squeeze().detach().numpy()
            sample_ax2p = self.net(
                torch.from_numpy(self.getNextState(state.view(1,-1,self.N).detach().numpy(), 
                                np.array([[0,da]]).astype('float32'), dt))).squeeze().detach().numpy()
            sample_ax2n = self.net(
                torch.from_numpy(self.getNextState(state.view(1,-1,self.N).detach().numpy(), 
                                np.array([[0,-da]]).astype('float32'), dt))).squeeze().detach().numpy()
            
            # Calculate approximate gradient
            action_dir = np.array([(sample_ax1p-sample_ax1n), (sample_ax2p-sample_ax2n)])*2/da/dt
#             print(action_dir)
            return np.clip(action_dir, -1,1)
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        
        # Find loss & optimize the model
        self.net.train() 
        pred_energy = self.net(state_batch.view(B, -1, self.N)[:,:2,:]).squeeze() # Input shape should be (B,no,N) and output be (B,1)
        energy = torch.from_numpy(self.getEnergy(state_batch.view(B, -1, self.N).detach().numpy()))
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(energy, pred_energy)
        print(loss)
        loss.backward()
        self.optimizer.step()

