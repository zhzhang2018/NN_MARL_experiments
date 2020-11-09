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
#         if centralized:
#             pass
#         else:
        if load_path is None:
            self.net = ActionNet(N, ns, na, hidden, action_range)
        else:
            self.net = ActionNetTF(N, prevN, load_path, ns, na, hidden, action_range)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.needsExpert = True
        self.name = 'LearnerAgent'
        
    # Picks an action based on given state
    def select_action(self, state, **kwargs):
#         print(self.net(state.view(1,-1,self.N)), self.net(state.view(1,-1,self.N)).shape)
        with torch.no_grad():
            return self.net(state.view(1,-1,self.N)).squeeze().detach().numpy()# # Expected size: (B=1, na, N) -> (na,N)?
    
    # Steps over gradients from memory replay
    def optimize_model(self, batch, **kwargs):
        B = kwargs.get('B', len(batch))
        # This class would assume that the optimal action is stored in batch input
        state_batch = torch.cat(batch.state)
        action_batch = torch.from_numpy(np.asarray(batch.action)) # cat? to device?
        reward_batch = torch.from_numpy(np.asarray(batch.reward).astype('float32'))

        # Find loss & optimize the model
        self.net.train() 
        pred_action = self.net(state_batch.view(B, -1, self.N)) # Input shape should be (B,no,N) and output be (B,na)
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(action_batch, pred_action)
        loss.backward()
        self.optimizer.step()
        
        # Check sizes - attach real size after those lines.
#         print(state_batch.shape)
#         print(action_batch.shape, pred_action.shape)
#         print(reward_batch.shape)

# Agent for leaning reward
class RewardAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, learning_rate=0.01, centralized=False,
                 prevN=10, load_path=None):
        super().__init__(device, N)
        self.centralized = centralized
#         if centralized:
#             pass
#         else:
        if load_path is None:
            self.net = RewardNet(N, ns, na, hidden)
        else:
            self.net = RewardNetTF(N, prevN, load_path, ns, na, hidden)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.na = na
        self.name = 'RewardAgent'
    
    # Idk how to implement this... randomly sample a bunch of possible actions and then pick the best one?
    def select_action(self, state, **kwargs):
        num_sample = kwargs.get('num_sample', 50)
        action_space = kwargs.get('action_space', [-1,1])
        with torch.no_grad():
#             actions = torch.from_numpy( np.random.rand(num_sample, self.na, self.N) )
            actions = torch.from_numpy( 
                    np.random.rand(num_sample, self.na).astype('float32')
                ) * (action_space[1]-action_space[0])+action_space[0]
            rewards = self.net(state.expand(num_sample, -1, -1), actions).view(-1)
#             print(rewards.shape, rewards.max(0))
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
        ) # Shape is (B,ns,N) and (B,na) for input and (B, ) for output??
#         print("Reward batch shape = ", reward_batch.shape, "; prediction shape = ", pred_reward.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        loss.backward()
        self.optimizer.step()
        
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
        
# Actor-Critic attempt #1
# Properties: Directly estimates action without sampling around. 
#             Directly updates Reward based on state and action without considering the future.
#             Is useless.
class AC1Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rateA=0.01, learning_rateC=0.02, centralized=False,
                 prevN=10, load_pathA=None, load_pathC=None):
        super().__init__(device, N)
        self.centralized = centralized
        
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
class AC2Agent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rateA=0.01, learning_rateC=0.02, centralized=False,
                 prevN=10, load_pathA=None, load_pathC=None):
        super().__init__(device, N)
        self.centralized = centralized
        
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
        self.na = na
        self.name = 'AC2Agent'
        
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

        # Find loss for Critic
        self.netC.train() # Critic and value predictions
        self.optimizerC.zero_grad()
        pred_reward = self.netC( state_batch.view(B, -1, self.N), action_batch.view(B, -1, 1) )
        lossC = torch.nn.functional.mse_loss(reward_batch, pred_reward.squeeze())
        lossC.backward()
        self.optimizerC.step()
        
        # Find loss for Actor
        self.netA.train() # Actor and action decisions
        self.optimizerA.zero_grad()
        pred_action = self.netA(state_batch.view(B, -1, self.N)) # Input shape should be (B,no,N) and output be (B,na)
        lossA = -self.netC(state_batch.view(B, -1, self.N), pred_action.view(B, -1, 1)).mean()
        lossA.backward()
        self.optimizerA.step()
        
    # Overwrite original because there are two nets now
    def set_train(self, train):
        if train:
            self.netA.train()
            self.netC.train()
        else:
            self.netA.eval()
            self.netC.eval()

# DDPG attempt
# References: https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py, model.py, util.py
# Properties: Chooses action using a net where loss is defined as negative predicted reward from Critic. 
#             Updates Reward based on state and action while [NOT considering the future state (with discounts) for now].
#             Uses two pairs of nets - each pair containing one target network.
class DDPGAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], 
                 learning_rateA=0.01, learning_rateC=0.02, tau=0.1, centralized=False,
                 prevN=10, load_pathA=None, load_pathC=None):
        super().__init__(device, N)
        self.tau = tau
        self.centralized = centralized
        
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
        self.optimizerA.step() # Do this in a later step to avoid
        # issues like this: https://github.com/pytorch/pytorch/issues/39141#issuecomment-636881953
        
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

