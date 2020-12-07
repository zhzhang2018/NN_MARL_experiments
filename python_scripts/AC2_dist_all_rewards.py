# Description: Decentralized, uses Gaussian Actor, uses unnormalized, instantaneous reward.
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
# sys.path.append("..")
sys.path.append(".")
import os
# print (os.getcwd())

# from ..utils.ReplayMemory import * 
# from ..utils.networks import *
# from ..utils.agents import *
# from ..utils.plotting import *
# from ..utils.train_test_methods import *
# from ..utils.params import *

from utils.ReplayMemory import * 
from utils.networks import *
from utils.agents import *
from utils.plotting import *
from utils.train_test_methods import *
from utils.params import *

# Ask for input configurations
taskname = input("Enter task name that would be used to name all outputs: ")
N = input("Number of agents: ")
N = int(N)

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_type = U_VELOCITY
observe_type = O_VELOCITY
observe_action = O_ACTION
reward_mode=ALL_REWARD

num_episode=5#00
test_interval=10#0
num_test=1#0#50
num_iteration=20#0
BATCH_SIZE=64#128
debug=False
num_sample=50
seed=22222
hidden=32

N_listv = [N] # [5,10,20]
env_listv = []
for N_ in N_listv:
    # Distance-based reward only, with hard penalty on touching the boundary.
    # Control group that doesn't give surviving reward, and instead stops immediately.
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD
        ).unwrapped
    )
    # Drop-dead immediately on touching the boundary
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
                 boundary_policy=DEAD_ON_TOUCH
        ).unwrapped
    )
    # Hard penalty on boundary, coupled with positive convergence reward. This means it never stops on consensus.
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD, 
                 finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # Soft penalty with consensus rewards. I don't expect it to successfully discover
    # achieving consensus would bring reward, though.
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, reward_mode=DIST_REWARD,
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
    # Comparison group that uses dist and actuation rewards, with hard penalty
    env_listv.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=(DIST_REWARD|ACT_REWARD)
        ).unwrapped
    )

labels = ['hard_bound_zero_posReward', 
          'dead_bound_zero_posReward', 
          'hard_bound_cumu_posReward', 
          'soft_bound_cumu_posReward', 
          'hard_bound_zero_posReward_v_penalty']*len(N_listv)
labels = [labels[i]+'_N{0}'.format(env_.N) for i,env_ in enumerate(env_listv)]

AC2_listv = []
for i,env_ in enumerate(env_listv):
    AC2_listv.append(
        AC2Agent(device, env_.N, env_.nf, env_.na, hidden, rand_modeA=rand_mode,
                 learning_rateA=0.01, learning_rateC=0.02)
    )

AC2_histv = []
AC2_lossv = []
for i,env_ in enumerate(env_listv):
    # AC2_listv[i].optimizerA.learning_rate = 0.1
    # AC2_listv[i].optimizerC.learning_rate = 0.1
    AC2_lossv.append([])
    AC2_histv.append(
        train(AC2_listv[i], env_, 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug,
#               update_mode=UPDATE_PER_ITERATION, reward_mode=FUTURE_REWARD_YES_NORMALIZE, loss_history=AC2_lossv[i])
              update_mode=UPDATE_PER_EPISODE, reward_mode=FUTURE_REWARD_YES, loss_history=AC2_lossv[i])
    )
    print("Finished training env with {0} agents for AC".format(env_.N))

AC2_test_histv = []
for i,env_ in enumerate(env_listv):
    AC2_test_histv.append(
        plot_test(AC2_listv[i], env_, fnames=['']*num_test,
            num_iteration=num_iteration, action_space=action_space, imdir='plots/',debug=debug)
    )
    print("Finished testnig env with {0} agents for AC".format(env_.N))

# Plot performance histories
skip = 1
for i in range(len(env_listv)):
    plot_reward_hist([AC2_histv[i][::skip]], test_interval*skip, 
                 [labels[i]], # ['AC2_N{0}'.format(env_list[i].N)], 
                 log=False, num_iteration=num_iteration, 
                 N_list=[env_listv[i].N], # ([1 for env_ in env_list]), 
                 bar=True, fname='plots/'+taskname+'_performance_'+labels[i])
# Plot loss history
skip=1
plot_loss_hist(hists=[h[::skip] for h in AC2_lossv], hist_names=labels, 
               log=False, num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/'+taskname+'_Critic_loss')
# Save models
for i,lab in enumerate(labels):
    AC2_listv[i].save_model("models/"+taskname+'_'+lab)

# Maybe save screenshots of real testing
for i,env_ in enumerate(env_listv):
    plot_test(AC2_listv[i], env_, fnames=[taskname+'_'+labels[i]+'_test{0}'.format(j) for j in range(1)],
        num_iteration=100, action_space=action_space, imdir='screencaps/',debug=debug)

print("Finished running "+taskname)



