# Description: Decentralized, uses Gaussian Actor, uses normalized, instantaneous reward.
# Created to test out different modes of AC2 and AC3. Also saves models mid-training.
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import sys
sys.path.append(".") # https://stackoverflow.com/a/53918952
import os

from utils.ReplayMemory import * 
from utils.networks import *
from utils.agents import *
from utils.plotting import *
from utils.train_test_methods import *
from utils.params import *

# Ask for input configurations
taskname = input("Enter task name that would be used to name all outputs: ")
failename = taskname
N = input("Number of agents: ")
N = int(N)
mode = input("Mode number: ")
mode = int(mode)
uses_boundary = input("Uses boundary? y/n")
uses_boundary = (uses_boundary == 'y')
# complete_graph = input("Always keep complete graph? y/n")

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

input_type = U_VELOCITY
observe_type = O_VELOCITY
observe_action = O_ACTION
reward_mode=ALL_REWARD

num_episode=750#500
test_interval=10#0
num_test=10#50
num_iteration=200
BATCH_SIZE=256#64#128
save_sim_intv=50
debug=False
num_sample=50
seed=22222
hidden=32
action_space=[-1,1]

# rand_mode = NO_RAND
rand_mode = GAUSS_RAND

N_listv = [N] # [5,10,20]
if N == 0:
    N_listv = [2,5,10]#,20]
    print("Using default N")
env_listv = []
for N_ in N_listv:
    # # Distance-based reward only, with hard penalty on touching the boundary.
    # # Control group that doesn't give surviving reward, and instead stops immediately.
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # # Drop-dead immediately on touching the boundary
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
    #              boundary_policy=DEAD_ON_TOUCH, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # # Hard penalty on boundary, coupled with positive convergence reward. This means it never stops on consensus.
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD, 
    #              finish_reward_policy=REWARD_IF_CONSENSUS, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # log reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, #o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.log(np.abs(x)+1)),
#                  dist_reward_func=lambda x : (np.sqrt(x) * 2),
#                  dist_reward_func=lambda x : (np.abs(x) + 2) * (np.abs(x) + 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # square root reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, #o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.sqrt(x) * 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # shifted quadratic reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, #o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.abs(x) + 2) * (np.abs(x) + 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # Soft penalty with consensus rewards. I don't expect it to successfully discover
    # achieving consensus would bring reward, though.
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS, 
                 uses_boundary=uses_boundary
        ).unwrapped
    )
    # Comparison group that uses dist and actuation rewards, with hard penalty
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=(DIST_REWARD|ACT_REWARD), 
                 uses_boundary=uses_boundary
        ).unwrapped
    )
    env_listv[-1].mov_w = 100

if not uses_boundary:
    taskname += '_fixed0'
else:
    taskname += '_bounded'
taskname += '_m{0}_'.format(mode)
labels = ['log_reward', # 'hard_bound_zero_posReward', 
          'sqrt_reward', # 'dead_bound_zero_posReward', 
          'squareshift_reward', # 'hard_bound_cumu_posReward', 
          'soft_bound_cumu_posReward', 
          'hard_bound_zero_posReward_v_penalty']*len(N_listv)
labels = [taskname+labels[i]+'_N{0}'.format(env_.N) for i,env_ in enumerate(env_listv)]

AC2_listv = []
for i,env_ in enumerate(env_listv):
    AC2_listv.append(
        AC2Agent(device, env_.N, env_.nf, env_.na, hidden, rand_modeA=rand_mode,
                 learning_rateA=0.01, learning_rateC=0.02, mode=mode)
    )

AC2_histv = []
AC2_lossv = []
memory_lane = []
for i,env_ in enumerate(env_listv):
    # AC2_listv[i].optimizerA.learning_rate = 0.1
    # AC2_listv[i].optimizerC.learning_rate = 0.1
    AC2_lossv.append([])
    # Not using memory: 
    # AC2_histv.append(
    #     train(AC2_listv[i], env_, 
    #           num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
    #           BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug,
    #           update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_YES|FUTURE_REWARD_NORMALIZE, 
    #           loss_history=AC2_lossv[i],
    #           save_sim_intv=save_sim_intv, save_sim_fnames=[labels[i]], 
    #           imdir='screencaps/', save_intm_models=True)
    # )
    # Using memory:
    h1,h2 = train(AC2_listv[i], env_, 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug,
              update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_YES|FUTURE_REWARD_NORMALIZE, 
              loss_history=AC2_lossv[i],
              save_sim_intv=save_sim_intv, save_sim_fnames=[labels[i]], 
              imdir='screencaps/', save_intm_models=True, return_memory=True)
    AC2_histv.append(h1)
    memory_lane.append(h2)
    print("Finished training env with {0} agents for AC".format(env_.N))

AC2_test_histv = []

# Plot performance histories
skip = 1
for i in range(len(env_listv)):
    plot_reward_hist([AC2_histv[i][::skip]], test_interval*skip, 
                 [labels[i]], # ['AC2_N{0}'.format(env_list[i].N)], 
                 log=False, num_iteration=num_iteration, 
                 N_list=[env_listv[i].N], # ([1 for env_ in env_list]), 
                 bar=True, fname='plots/'+labels[i])
# Plot loss history
skip=1
plot_loss_hist(hists=[h[::skip] for h in AC2_lossv], hist_names=labels, 
               log=False, num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/'+taskname+'_Critic_loss')
# Save models
# The Agent object would assume there's a subfolder named "models/".
for i,lab in enumerate(labels):
    AC2_listv[i].save_model(taskname+'_'+lab)
    # If using memory:
    torch.save( Transition(*zip(*memory_lane[i].memory)), taskname+'_'+lab+'_memory' )

print("Finished running "+taskname)
print("Trying to generate screenshots now...")

print("Trying to run with the reversed uses_boundary input")
# uses_boundary = not uses_boundary
env_listv = []
for N_ in N_listv:
    # # Distance-based reward only, with hard penalty on touching the boundary.
    # # Control group that doesn't give surviving reward, and instead stops immediately.
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # # Drop-dead immediately on touching the boundary
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
    #              boundary_policy=DEAD_ON_TOUCH, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # # Hard penalty on boundary, coupled with positive convergence reward. This means it never stops on consensus.
    # env_listv.append(
    #     gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
    #           input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD, 
    #              finish_reward_policy=REWARD_IF_CONSENSUS, 
    #              uses_boundary=uses_boundary
    #     ).unwrapped
    # )
    # log reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.log(np.abs(x)+1)),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # square root reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.sqrt(x) * 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # shifted quadratic reward
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
                 uses_boundary=uses_boundary,
                 dist_reward_func=lambda x : (np.abs(x) + 2) * (np.abs(x) + 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # Soft penalty with consensus rewards. I don't expect it to successfully discover
    # achieving consensus would bring reward, though.
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS, 
                 uses_boundary=uses_boundary
        ).unwrapped
    )
    # Comparison group that uses dist and actuation rewards, with hard penalty
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=(DIST_REWARD|ACT_REWARD), 
                 uses_boundary=uses_boundary
        ).unwrapped
    )
    env_listv[-1].mov_w = 100
taskname = failename+'_K_'
# taskname = failename
if not uses_boundary:
    taskname += '_fixed0'
else:
    taskname += '_bounded'
taskname += '_m{0}_'.format(mode)
labels = ['log_reward', # 'hard_bound_zero_posReward', 
          'sqrt_reward', # 'dead_bound_zero_posReward', 
          'squareshift_reward', # 'hard_bound_cumu_posReward', 
          'soft_bound_cumu_posReward', 
          'hard_bound_zero_posReward_v_penalty']*len(N_listv)
labels = [taskname+labels[i]+'_N{0}'.format(env_.N) for i,env_ in enumerate(env_listv)]

AC2_listv = []
for i,env_ in enumerate(env_listv):
    AC2_listv.append(
        AC2Agent(device, env_.N, env_.nf, env_.na, hidden, rand_modeA=rand_mode,
                 learning_rateA=0.01, learning_rateC=0.02, mode=mode)
    )

AC2_histv = []
AC2_lossv = []
for i,env_ in enumerate(env_listv):
    # AC2_listv[i].optimizerA.learning_rate = 0.1
    # AC2_listv[i].optimizerC.learning_rate = 0.1
    AC2_lossv.append([])
    # Not using memory: 
    # AC2_histv.append(
    #     train(AC2_listv[i], env_, 
    #           num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
    #           BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug,
    #           update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_YES|FUTURE_REWARD_NORMALIZE, 
    #           loss_history=AC2_lossv[i],
    #           save_sim_intv=save_sim_intv, save_sim_fnames=[labels[i]], 
    #           imdir='screencaps/', save_intm_models=True)
    # )
    # Using memory:
    h1,h2 = train(AC2_listv[i], env_, 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug,
              update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_YES|FUTURE_REWARD_NORMALIZE, 
              loss_history=AC2_lossv[i],
              save_sim_intv=save_sim_intv, save_sim_fnames=[labels[i]], 
              imdir='screencaps/', save_intm_models=True, return_memory=True)
    AC2_histv.append(h1)
    memory_lane.append(h2)
    print("Finished training env with {0} agents for AC".format(env_.N))

AC2_test_histv = []

# Plot performance histories
skip = 1
for i in range(len(env_listv)):
    plot_reward_hist([AC2_histv[i][::skip]], test_interval*skip, 
                 [labels[i]], # ['AC2_N{0}'.format(env_list[i].N)], 
                 log=False, num_iteration=num_iteration, 
                 N_list=[env_listv[i].N], # ([1 for env_ in env_list]), 
                 bar=True, fname='plots/'+labels[i])
# Plot loss history
skip=1
plot_loss_hist(hists=[h[::skip] for h in AC2_lossv], hist_names=labels, 
               log=False, num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/'+taskname+'_Critic_loss')
# Save models
# The Agent object would assume there's a subfolder named "models/".
for i,lab in enumerate(labels):
    AC2_listv[i].save_model(taskname+'_'+lab)
    # If using memory:
    torch.save( Transition(*zip(*memory_lane[i].memory)), taskname+'_'+lab+'_memory' )

print("Finished running "+taskname)



