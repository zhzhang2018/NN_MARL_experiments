from utils.networks import *
from utils.ReplayMemory import *
from utils.agents import *
from utils.plotting import plot_test
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
UPDATE_ON_POLICY = 2
FUTURE_REWARD_YES = 0x1
FUTURE_REWARD_NO = 0x0
FUTURE_REWARD_YES_NORMALIZE = 0x3
FUTURE_REWARD_NORMALIZE = 0x2

def train(agent, env, num_episode=50, test_interval=25, num_test=20, num_iteration=200, 
          BATCH_SIZE=128, num_sample=50, action_space=[-1,1], debug=True, memory=None, seed=2020,
          update_mode=UPDATE_PER_ITERATION, reward_mode=FUTURE_REWARD_NO, gamma=0.99, 
          loss_history=[], loss_historyA=[], lr_history=[], lr_historyA=[],
          save_sim_intv=50, save_sim_fnames=[], imdir='screencaps/', useVid=False):
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
        if agent.centralized:
            state = env.state
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if debug:
            env.render()
        # Train History
        state_pool = []
        action_pool = []
        reward_pool = []
        next_state_pool = []
        loss_history.append([])
        loss_historyA.append([])

        for t in range(num_iteration):
#             agent.net.train()
            agent.set_train(True)
            # Try to pick an action, react, and store the resulting behavior in the pool here
            if agent.centralized:
                action = agent.select_action(state, **{
                        'steps_done':t, 'num_sample':50, 'action_space':action_space
                    }).T
            else:
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
            if agent.centralized:
                next_state = env.state
            next_state = Variable(torch.from_numpy(next_state).float()) # The float() probably avoids bug in net.forward()
            action = action.T # Turn shape back to (N,2)

            if agent.needsExpert:
                # If we need to use expert input during training, then we consult it and get the best action for this state
                actions = env.controller()
                action = actions.T # Shape should already be (2,N), so we turn it into (N,2)
            
            if not(agent.centralized):
                if reward_mode & FUTURE_REWARD_YES == 0:
                    # Push everything directly inside if we don't use future discounts
                    for i in range(N):
                        memory.push(state[i], action[i], next_state[i], reward[i])
                else:
                    # Store and push them outside the loop
                    state_pool.append(state)
                    action_pool.append(action)
                    reward_pool.append(reward)
                    next_state_pool.append(next_state)
            else:
                # Centralized training should directly use the real states, instead of observations
                reward = np.sum(reward)
                if reward_mode & FUTURE_REWARD_YES == 0:
                    # Push everything directly inside if we don't use future discounts
                    memory.push(state, action, env.state, reward)
                else:
                    # Store and push them outside the loop
                    state_pool.append(state)
                    action_pool.append(action)
                    reward_pool.append(reward)
                    next_state_pool.append(env.state)

            # Update 1028: Moved this training step outside the loop
            if update_mode == UPDATE_PER_ITERATION:
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    agent.optimize_model(batch, **{'B':BATCH_SIZE})
                elif len(memory) > 0:
                    transitions = memory.sample(len(memory))
                    batch = Transition(*zip(*transitions))
                    agent.optimize_model(batch, **{'B':len(memory)})
                loss_history[-1].append(agent.losses[:])
#                 print(e,t,agent.losses)
                agent.losses=[]
                # Also record scheduler history for learning rate. If the scheduler is a Plateau one, then
                # we can know from the learning rate if we're in a flatter area.
                # https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234/2
                # The scheduler requires the validation loss - can I just use the average training loss instead?
#                 try:
#                     agent.scheduler.step(np.mean(loss_history[-1]))
#                     lr_history.append(agent.optimizer.param_groups[0]['lr'])
#                 except:
#                     agent.schedulerC.step(np.mean(loss_history[-1]))
#                     lr_history.append(agent.optimizerC.param_groups[0]['lr'])
                try:
                    loss_historyA[-1].append(agent.lossesA[:])
                    agent.lossesA=[]
#                     agent.schedulerA.step(np.mean(loss_historyA[-1]))
#                     lr_historyA.append(agent.optimizerA.param_groups[0]['lr'])
                except:
                    pass
            elif update_mode == UPDATE_ON_POLICY:
                # This case would ditch sampling, and just update by the current thing
                print("Not implemented!!!!")
                return None

            state = next_state
            steps += 1

            if debug:
                env.render()

            if debug and done:
                print("Took ", t, " steps to converge")
                break
        
        if reward_mode & FUTURE_REWARD_YES == 1:
            for j in range(len(reward_pool)): ### IT was previously miswritten as "reward". Retard bug that might had effects
                if j > 0:
                    reward_pool[-j-1] += gamma * reward_pool[-j]
            if reward_mode & FUTURE_REWARD_NORMALIZE == 1:
                reward_pool = torch.tensor(reward_pool)
                reward_pool = (reward_pool - reward_pool.mean()) / reward_pool.std()
            
            if agent.centralized:
                for j in range(len(reward_pool)):
                    memory.push(state_pool[-j-1], action_pool[-j-1], 
                                next_state_pool[-j-1], reward_pool[-j-1])
            else:
                for j in range(len(reward_pool)):
                    for i in range(N):
                        memory.push(state_pool[-j-1][i], action_pool[-j-1][i], 
                                    next_state_pool[-j-1][i], reward_pool[-j-1][i])

        if update_mode == UPDATE_PER_EPISODE:
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                agent.optimize_model(batch, **{'B':BATCH_SIZE})
            elif len(memory) > 0:
                transitions = memory.sample(len(memory))
                batch = Transition(*zip(*transitions))
                agent.optimize_model(batch, **{'B':len(memory)})
            loss_history[-1].append(agent.losses[:])
            agent.losses=[]
            # Also record scheduler history for learning rate. If the scheduler is a Plateau one, then
            # we can know from the learning rate if we're in a flatter area.
            # https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234/2
#             try:
#                 agent.scheduler.step(np.mean(loss_history[-1]))
#                 lr_history.append(agent.optimizer.param_groups[0]['lr'])
#             except:
#                 agent.schedulerC.step(np.mean(loss_history[-1]))
#                 lr_history.append(agent.optimizerC.param_groups[0]['lr'])
            try:
                loss_historyA[-1].append(agent.lossesA[:])
                agent.lossesA=[]
#                 agent.schedulerA.step(np.mean(loss_historyA[-1]))
#                 lr_historyA.append(agent.optimizerA.param_groups[0]['lr'])
            except:
                pass
        
        if debug:
            print("Episode ", e, " finished; t = ", t)
        
        if e % test_interval == 0:
            print("Test result at episode ", e, ": ")
            test_hist = test(agent, env, num_test, num_iteration, num_sample, action_space, 
                             seed=test_seeds[int(e/test_interval)], debug=debug)
            test_hists.append(test_hist)
        
        # Save demos of simulation if wanted
        if e % save_sim_intv == (save_sim_intv-1) and e > 0:
            try:
                fnames = [f+'_{0}'.format(e) for f in save_sim_fnames]
                plot_test(agent, env, fnames=fnames,
                    num_iteration=num_iteration, action_space=action_space, imdir=imdir,
                    debug=debug, useVid=useVid)
                for f in fnames:
                    os.system('ffmpeg -y -pattern_type glob -i "'+imdir+f+'*.jpg" '+f+'.gif')
            except:
                print("Failed to save simulation at e={0}".format(e))
    return test_hists
                
def test(agent, env, num_test=20, num_iteration=200, num_sample=50, action_space=[-1,1], seed=2020, debug=True):
    reward_hist_hst = []
    N=env.N
    # To affect pytorch, refer to further documentations: https://github.com/pytorch/pytorch/issues/7068
#     torch.manual_seed(seed)
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 31102528, size=num_test)
#     print(env_seeds)
    with torch.no_grad(): # Added 1112 to reduce memory load
        for e in range(num_test):
            steps = 0
    #         agent.net.eval()
            agent.set_train(False)
            cum_reward = 0
            reward_hist = []

            np.random.seed(env_seeds[e])
            state = env.reset()
            if agent.centralized:
                state = env.state
            state = torch.from_numpy(state).float()
            state = Variable(state)
            if debug:
                env.render()

            for t in range(num_iteration):  
                # Try to pick an action, react, and store the resulting behavior in the pool here
                if agent.centralized:
                    action = agent.select_action(state, **{
                            'steps_done':t, 'rand':False, 'num_sample':50, 'action_space':action_space
                        }).T
                else:
                    actions = []
                    for i in range(N):
                        action = agent.select_action(state[i], **{
                            'steps_done':t, 'rand':False, 'num_sample':50, 'action_space':action_space
                        })
                        actions.append(action)
                    action = np.array(actions).T 

                next_state, reward, done, _ = env.step(action)
                if agent.centralized:
                    next_state = env.state
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
            if debug:
                print("Finished test ", e, " with ", t, #" steps, and rewards = ", reward, 
                  "; cumulative reward = ", cum_reward)
            reward_hist_hst.append(reward_hist)
        env.close()
        return reward_hist_hst