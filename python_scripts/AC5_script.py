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

input_type = U_VELOCITY
observe_type = O_VELOCITY
observe_action = O_ACTION
reward_mode=ALL_REWARD

num_episode=10000#500#250#500
test_interval=100#0
num_test=20#10#50
num_iteration=200
BATCH_SIZE=2*128#64#128
save_sim_intv=200
debug=False
num_sample=50
seed=22222
hidden=32
action_space=[-1,1]

# rand_mode = NO_RAND
rand_mode = GAUSS_RAND

mode_list = [1204,1205,1208,0,-1204,-1205,-1208]
N_list = [3]*len(mode_list)
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
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS,
                 graph_laplacian_terminate_policy=False#True
        ).unwrapped
    )
#     env_list[-1].mov_w = 100

AC2_list = []
for i,N_ in enumerate(N_list):
    AC2_list.append(
        AC5Agent(device, N_, env_list[i].ns, # IMPORTANT!!! use .ns for centralized, and .nf for decentralized
                 env_list[i].na, hidden, rand_modeA=rand_mode, centralized=True,
                 neg_loss_sign=mode_list[i]<0,
                 learning_rateA=0.01, learning_rateC=0.02, mode=mode_list[i])
    )

sim_fnames = ['AC5_centralized_logreward_Lapeig_difftanhC_leak03A_m{1}_N{0}'.format(
    N_list[i],mode_list[i]) for i in range(len(N_list))]
# sim_fnames = ['AC2_centralizedTest_logreward_tanhAC_leak03A_m0_N{0}'.format(N_) for N_ in N_list]
# memory_backup = []
AC2_hist = []
AC2_loss = []
for i,N_ in enumerate(N_list):
    AC2_loss.append([])
    # memory_backup.append( ReplayMemory(1000 * env_list[i].N) )
    AC2_hist.append(
        train(AC2_list[i], env_list[i], 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug, #memory=memory_backup[-1],
              update_mode=UPDATE_PER_EPISODE, #UPDATE_PER_ITERATION,
              reward_mode=FUTURE_REWARD_NORMALIZE|FUTURE_REWARD_YES,
#               reward_mode=FUTURE_REWARD_YES,#|FUTURE_REWARD_NORMALIZE, #FUTURE_REWARD_YES_NORMALIZE, 
              loss_history=AC2_loss[i], #reward_mean_var=(torch.Tensor([-69600]), torch.Tensor([46290])),
              save_sim_intv=save_sim_intv, save_sim_fnames=[sim_fnames[i]], 
              imdir='screencaps/', save_intm_models=True, useVid=False)
    )
    print("Finished training env with {0} agents for AC".format(N_))

# Plot performance histories
skip = 1
for i in range(len(AC2_hist)):
    plot_reward_hist([AC2_hist[i][::skip]], test_interval*skip, 
                     ['AC5_N{0}_m{1}'.format(N_list[i],mode_list[i])],
                     log=False, num_iteration=num_iteration, N_list=[N_list[i]], bar=True, 
                     fname='plots/'+sim_fnames[i])

# Plot loss history
skip=1
plot_loss_hist(hists=[h[::skip] for h in AC2_loss], 
               hist_names=['AC5_N{0}_m{1}'.format(N_,mode_list[i]) for i,N_ in enumerate(N_list)], log=False, 
               num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/AC5_centralized_logreward_Lapeig_difftanhC_leak03A_Critic')
plot_loss_hist(hists=[h[500::skip] for h in AC2_loss], 
               hist_names=['AC5_N{0}_m{1}'.format(N_,mode_list[i]) for i,N_ in enumerate(N_list)], log=False, 
               num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/AC5_centralized_logreward_Lapeig_difftanhC_leak03A_Critic')



env_list = []
for N_ in N_list:
    env_list.append(
        gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N_, dt=0.1, Delta=0.05, #o_radius=40000,
              input_type=input_type, observe_type=observe_type, observe_action=O_ACTION, 
                 reward_mode=DIST_REWARD,#|ACT_REWARD,
#                  uses_boundary=True,
                 uses_boundary=False,
                 dist_reward_func=lambda x : (np.log(np.abs(x)+1)),
#                  dist_reward_func=lambda x : np.sqrt(np.abs(x)),
#                  dist_reward_func=lambda x : (np.abs(x) + 2) * (np.abs(x) + 2), # * (np.abs(x) + 2) * (np.abs(x) + 2),
                 boundary_policy=SOFT_PENALTY, finish_reward_policy=REWARD_IF_CONSENSUS,
                 graph_laplacian_terminate_policy=False#True
        ).unwrapped
    )
#     env_list[-1].mov_w = 100

AC2_list = []
for i,N_ in enumerate(N_list):
    AC2_list.append(
        AC5Agent(device, N_, env_list[i].ns, # IMPORTANT!!! use .ns for centralized, and .nf for decentralized
                 env_list[i].na, hidden, rand_modeA=rand_mode, centralized=True,
                 neg_loss_sign=mode_list[i]<0,
                 learning_rateA=0.01, learning_rateC=0.02, mode=mode_list[i])
    )

sim_fnames = ['AC5_decentralized_logreward_Lapeig_difftanhC_leak03A_m{1}_N{0}'.format(
    N_list[i],mode_list[i]) for i in range(len(N_list))]
# sim_fnames = ['AC2_centralizedTest_logreward_tanhAC_leak03A_m0_N{0}'.format(N_) for N_ in N_list]
# memory_backup = []
AC2_hist = []
AC2_loss = []
for i,N_ in enumerate(N_list):
    AC2_loss.append([])
    # memory_backup.append( ReplayMemory(1000 * env_list[i].N) )
    AC2_hist.append(
        train(AC2_list[i], env_list[i], 
              num_episode=num_episode, test_interval=test_interval, num_test=num_test, num_iteration=num_iteration, 
              BATCH_SIZE=BATCH_SIZE, num_sample=num_sample, action_space=[-1,1], debug=debug, #memory=memory_backup[-1],
              update_mode=UPDATE_PER_EPISODE, #UPDATE_PER_ITERATION,
              reward_mode=FUTURE_REWARD_NORMALIZE|FUTURE_REWARD_YES,
#               reward_mode=FUTURE_REWARD_YES,#|FUTURE_REWARD_NORMALIZE, #FUTURE_REWARD_YES_NORMALIZE, 
              loss_history=AC2_loss[i], #reward_mean_var=(torch.Tensor([-69600]), torch.Tensor([46290])),
              save_sim_intv=save_sim_intv, save_sim_fnames=[sim_fnames[i]], 
              imdir='screencaps/', save_intm_models=True, useVid=False)
    )
    print("Finished training env with {0} agents for AC".format(N_))

# Plot performance histories
skip = 1
for i in range(len(AC2_hist)):
    plot_reward_hist([AC2_hist[i][::skip]], test_interval*skip, 
                     ['AC5_N{0}_m{1}'.format(N_list[i],mode_list[i])],
                     log=False, num_iteration=num_iteration, N_list=[N_list[i]], bar=True, 
                     fname='plots/'+sim_fnames[i])

# Plot loss history
skip=1
plot_loss_hist(hists=[h[::skip] for h in AC2_loss], 
               hist_names=['AC5_N{0}_m{1}'.format(N_,mode_list[i]) for i,N_ in enumerate(N_list)], log=False, 
               num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/AC5_decentralized_logreward_Lapeig_difftanhC_leak03A_Critic')
plot_loss_hist(hists=[h[500::skip] for h in AC2_loss], 
               hist_names=['AC5_N{0}_m{1}'.format(N_,mode_list[i]) for i,N_ in enumerate(N_list)], log=False, 
               num_iteration=num_iteration, update_mode=UPDATE_PER_ITERATION, bar=False,
               fname='plots/AC5_decentralized_logreward_Lapeig_difftanhC_leak03A_Critic')
