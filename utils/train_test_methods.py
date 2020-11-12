from utils.networks import *
from utils.ReplayMemory import *
from utils.agents import *
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

# Training modes
UPDATE_PER_ITERATION = 0
UPDATE_PER_EPISODE = 1
FUTURE_REWARD_YES = 0
FUTURE_REWARD_NO = 1
FUTURE_REWARD_YES_NORMALIZE = 2

def train(agent, env, num_episode=50, test_interval=25, num_test=20, num_iteration=200, 
          BATCH_SIZE=128, num_sample=50, action_space=[-1,1], debug=True, memory=None, seed=2020,
          update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_NO, gamma=0.99):
    test_hists = []
    steps = 0
    if memory is None:
        ### UPDate 11/05: Changed memory size based on number of agents
        memory = ReplayMemory(1000 * env.N)
    
    # Values that would be useful
    N = env.N
    # Note that the seed only controls the numpy random, which affects the environment.
    # To affect pytorch, refer to further documentations: https://github.com/pytorch/pytorch/issues/7068
    np.random.seed(seed)
#     torch.manual_seed(seed)
    test_seeds = np.random.randint(0, 5392644, size=int(num_episode // test_interval)+1)

    for e in range(num_episode):
        steps = 0
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if debug:
            env.render()
        # Train History
        state_pool = []
        action_pool = []
        reward_pool = []
        next_state_pool = []

        for t in range(num_iteration):
#             agent.net.train()
            agent.set_train(True)
            # Try to pick an action, react, and store the resulting behavior in the pool here
            actions = []
            for i in range(N):
                action = agent.select_action(state[i], **{
                    'steps_done':t, 'num_sample':50, 'action_space':action_space
                })
                actions.append(action)
            action = np.array(actions).T # Shape would become (2,N)
            # print(action, actions)
            # action = actions
            # action = np.array(actions)

            next_state, reward, done, _ = env.step(action)
            next_state = Variable(torch.from_numpy(next_state).float()) # The float() probably avoids bug in net.forward()
            action = action.T # Turn shape back to (N,2)

            if agent.needsExpert:
                # If we need to use expert input during training, then we consult it and get the best action for this state
                actions = env.controller()
                action = actions.T # Shape should already be (2,N), so we turn it into (N,2)
            
            if reward_mode == FUTURE_REWARD_NO:
                # Push everything directly inside if we don't use future discounts
                for i in range(N):
                    memory.push(state[i], action[i], next_state[i], reward[i])
            else:
                # Store and push them outside the loop
                state_pool.append(state)
                action_pool.append(action)
                reward_pool.append(reward)
                next_state_pool.append(next_state)

            # Update 1028: Moved this training step outside the loop
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                agent.optimize_model(batch, **{'B':BATCH_SIZE})
            elif len(memory) > 0:
                transitions = memory.sample(len(memory))
                batch = Transition(*zip(*transitions))
                agent.optimize_model(batch, **{'B':len(memory)})

            state = next_state
            steps += 1

            if debug:
                env.render()

            if debug and done:
                print("Took ", t, " steps to converge")
                break
        
        if reward_mode == FUTURE_REWARD_YES:
            for j in range(len(reward)):
                if j > 0:
                    reward_pool[-j-1] += gamma * reward_pool[-j]
                for i in range(N):
                    memory.push(state_pool[-j-1][i], action_pool[-j-1][i], 
                                next_state_pool[-j-1][i], reward_pool[-j-1][i])
        elif reward_mode == FUTURE_REWARD_YES_NORMALIZE:
            for j in range(len(reward)):
                if j > 0:
                    reward_pool[-j-1] += gamma * reward_pool[-j]
            reward_pool = torch.tensor(reward_pool)
            reward_pool = (reward_pool - reward_pool.mean()) / reward_pool.std()
            for j in range(len(reward)):
                for i in range(N):
                    memory.push(state_pool[-j-1][i], action_pool[-j-1][i], 
                                next_state_pool[-j-1][i], reward_pool[-j-1][i])

        if debug:
            print("Episode ", e, " finished; t = ", t)
        
        if e % test_interval == 0:
            print("Test result at episode ", e, ": ")
            test_hist = test(agent, env, num_test, num_iteration, num_sample, action_space, 
                             seed=test_seeds[int(e/test_interval)], debug=debug)
            test_hists.append(test_hist)
    env.close()
    return test_hists

def test(agent, env, num_test=20, num_iteration=200, num_sample=50, action_space=[-1,1], seed=2020, debug=True):
    reward_hist_hst = []
    N=env.N
    # To affect pytorch, refer to further documentations: https://github.com/pytorch/pytorch/issues/7068
#     torch.manual_seed(seed)
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 31102528, size=num_test)
    print(env_seeds)
    
    for e in range(num_test):
        steps = 0
#         agent.net.eval()
        agent.set_train(False)
        cum_reward = 0
        reward_hist = []

        np.random.seed(env_seeds[e])
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if debug:
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

            if e % 10 == 0 and debug:
                env.render()
            steps += 1

            if done:
#                 print("Took ", t, " steps to converge")
                break
        print("Finished test ", e, " with ", t, #" steps, and rewards = ", reward, 
              "; cumulative reward = ", cum_reward)
        reward_hist_hst.append(reward_hist)
    env.close()
    return reward_hist_hst