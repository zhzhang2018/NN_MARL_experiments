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

num_episode = 50
num_iteration = 1000
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
        if actor_path is not None:
            self.net.load_state_dict(torch.load(agent_path).to(self.device))

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
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], learning_rate=0.01, centralized=False):
        super().__init__(device, N)
        self.centralized = centralized
        if centralized:
            
        else:
            self.net = ActionNet(N, ns, na, hidden, action_range)
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
        pred_action = self.net(state_batch.view(B, -1, self.N)) # Shape should be (B,na)??
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape)

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(action_batch, pred_action)
        loss.backward()
        self.optimizer.step()
        
        # Check sizes - attach real size after those lines.
        print(state_batch.shape)
        print(action_batch.shape, pred_action.shape)
        print(reward_batch.shape)

# Agent for leaning reward
class RewardAgent(BaseAgent):
    def __init__(self, device, N, ns=2, na=5, hidden=24, learning_rate=0.01, centralized=False):
        super().__init__(device, N)
        self.centralized = centralized
        if centralized:
            
        else:
            self.net = RewardNet(N, ns, na, hidden)
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
    def __init__(self, device, N, ns=2, na=5, hidden=24, action_range=[-1,1], learning_rate=0.01, centralized=False):
        super().__init__(device, N)
        self.centralized = centralized
        if centralized:
            
        else:
            self.net = ActionNet(N, ns, na, hidden, action_range)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.name = 'RewardActionAgent'
        
    # Picks an action based on given state
    def select_action(self, state, **kwargs):
#         print(self.net(state.view(1,-1,self.N)), self.net(state.view(1,-1,self.N)).shape)
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
#         print("Action batch shape = ", action_batch.shape, "; prediction shape = ", pred_action.shape, "; reward shape = ", reward_batch.shape)

        # Find loss & optimize the model
        self.net.train()
        self.optimizer.zero_grad()
        loss = (pred_action * reward_batch.view(B,-1)).sum()
        loss.backward()
        self.optimizer.step()
        
def train(agent, env, num_episode=50, test_interval=25, num_test=20, num_iteration=200, 
          BATCH_SIZE=128, num_sample=50, action_space=[-1,1], debug=True, memory=None, seed=2020):
    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    test_hists = []
    steps = 0
    if memory is None:
        memory = ReplayMemory(10000)
    
    # Values that would be useful
    N = env.N
    # Note that the seed only controls the numpy random, which affects the environment.
    # To affect pytorch, refer to further documentations: https://github.com/pytorch/pytorch/issues/7068
    np.random.seed(seed)
#     torch.manual_seed(seed)
    test_seeds = np.random.randint(0, 5392644, size=(num_episode // test_interval))

    for e in range(num_episode):
        steps = 0
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if debug:
            env.render()

        for t in range(num_iteration):
            agent.net.train()     
            # Try to pick an action, react, and store the resulting behavior in the pool here
            actions = []
            for i in range(N):
                action = agent.select_action(state[i], **{
                    'steps_done':t, 'num_sample':50, 'action_space':action_space
                })
                actions.append(action)
            action = np.array(actions).T # Shape would become (2,N)
#             print(action, actions)
#             action = actions
#             action = np.array(actions)

            next_state, reward, done, _ = env.step(action)
            next_state = Variable(torch.from_numpy(next_state).float()) # The float() probably avoids bug in net.forward()
            action = action.T # Turn shape back to (N,2)

            if agent.needsExpert:
                # If we need to use expert input during training, then we consult it and get the best action for this state
                actions = env.controller()
                action = actions.T # Shape should already be (2,N), so we turn it into (N,2)
            for i in range(N):
                memory.push(state[i], action[i], next_state[i], reward[i])
            state = next_state
            steps += 1

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                agent.optimize_model(batch, **{'B':BATCH_SIZE})
            
            if debug:
                env.render()

            if debug and done:
                print("Took ", t, " steps to converge")
                break
        if debug:
            print("Episode ", e, " finished; t = ", t)
        
        if e % test_interval == 0:
            print("Test result at episode ", e, ": ")
            test_hist = test(agent, env, num_test, num_iteration, num_sample, action_space, seed=test_seeds[int(e/test_interval)])
            test_hists.append(test_hist)
    return test_hists

def test(agent, env, num_test=20, num_iteration=200, num_sample=50, action_space=[-1,1], seed=2020):
    reward_hist_hst = []
    N=env.N
    # To affect pytorch, refer to further documentations: https://github.com/pytorch/pytorch/issues/7068
#     torch.manual_seed(seed)
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 31102528, size=num_test)
    print(env_seeds)
    
    for e in range(num_test):
        steps = 0
        agent.net.eval()
        cum_reward = 0
        reward_hist = []

        np.random.seed(env_seeds[e])
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        env.render()

        for t in range(num_iteration):  
            # Try to pick an action, react, and store the resulting behavior in the pool here
            actions = []
            for i in range(N):
                action = agent.select_action(state[i], **{
                    'steps_done':t, 'rand':False, 'num_sample':50, 'action_space':action_space
                })
                actions.append(action)
            action = np.array(actions).T 

            next_state, reward, done, _ = env.step(action)
            next_state = Variable(torch.from_numpy(next_state).float()) # The float() probably avoids bug in net.forward()
            state = next_state
            cum_reward += sum(reward)
            reward_hist.append(reward)

            if e % 10 == 0:
                env.render()
            steps += 1

            if done:
#                 print("Took ", t, " steps to converge")
                break
        print("Finished test ", e, " with ", t, #" steps, and rewards = ", reward, 
              "; cumulative reward = ", cum_reward)
        reward_hist_hst.append(reward_hist)
    return reward_hist_hst