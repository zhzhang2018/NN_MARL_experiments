%matplotlib inline
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
from utils.retrieve_sim import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

num_episode=20000#500#250#500
test_interval=100#0
num_test=20#10#50
num_iteration=200
BATCH_SIZE=2*128#64#128
save_sim_intv=2000
debug=False
num_sample=50
seed=22222
hidden=32
action_space=[-1,1]

# rand_mode = NO_RAND
rand_mode = GAUSS_RAND

N_list = [3]
env_list = []
for N_ in N_list:
    env_list.append(
        gym.make('ConsensusEnv:CentralizedConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, #o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
#                  uses_boundary=True,
                 uses_boundary=False,
                 dist_reward_func=lambda x : (np.log(np.abs(x)+1)),
#                  dist_reward_func=lambda x : np.sqrt(np.abs(x)),
#                  dist_reward_func=lambda x : (np.abs(x) + 2) * (np.abs(x) + 2), # * (np.abs(x) + 2) * (np.abs(x) + 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS
        ).unwrapped
    )
#     env_list[-1].mov_w = 100

AC2_list = []
for i,N_ in enumerate(N_list):
    AC2_list.append(
        AC2Agent(device, N_, env_list[i].ns, # IMPORTANT!!! use .ns for centralized, and .nf for decentralized
                 env_list[i].na, hidden, rand_modeA=rand_mode, centralized=True,
                 neg_loss_sign=True,#False,
                 learning_rateA=0.01, learning_rateC=0.02, mode=12088)
    )

sim_fnames = ['AC2_centralizedTest_logreward_tanhAC_m0_N{0}'.format(N_) for N_ in N_list]
# sim_fnames = ['AC2_centralizedTest_logreward_tanhAC_leak03A_m0_N{0}'.format(N_) for N_ in N_list]
memory_backup = []
AC2_hist = []
AC2_loss = []
for i,N_ in enumerate(N_list):
    AC2_loss.append([])
    memory_backup.append( ReplayMemory(1000 * env_list[i].N) )
    AC2_hist.append(
        train(AC2_list[i], env_list[i], 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug, memory=memory_backup[-1],
              update_mode=UPDATE_PER_EPISODE, #UPDATE_PER_ITERATION,
              reward_mode=FUTURE_REWARD_NORMALIZE,
#               reward_mode=FUTURE_REWARD_YES,#|FUTURE_REWARD_NORMALIZE, #FUTURE_REWARD_YES_NORMALIZE, 
              loss_history=AC2_loss[i], #reward_mean_var=(torch.Tensor([-69600]), torch.Tensor([46290])),
              save_sim_intv=save_sim_intv, save_sim_fnames=[sim_fnames[i]], 
              imdir='screencaps/', save_intm_models=True, useVid=False)
    )
    print("Finished training env with {0} agents for AC".format(N_))

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
