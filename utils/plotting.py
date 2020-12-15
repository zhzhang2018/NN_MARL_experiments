import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from gym import wrappers

# Training modes
UPDATE_PER_ITERATION = 0
UPDATE_PER_EPISODE = 1
UPDATE_ON_POLICY = 2
FUTURE_REWARD_YES = 0
FUTURE_REWARD_NO = 1
FUTURE_REWARD_YES_NORMALIZE = 2

# Run with baseline (e.g. expert)
def run_expert(env, num_iteration=200, seed=2020):
    reward_hist_hst = []
    N=env.N
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 31102528, size=1)
    print(env_seeds)
    
    steps = 0
    cum_reward = 0
    reward_hist = []

    np.random.seed(env_seeds[0])
    state = env.reset()
    state = torch.from_numpy(state).float()
    env.render()

    for t in range(num_iteration):
        action = env.controller()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        cum_reward += sum(reward)
        reward_hist.append(reward)
        steps += 1
        if done:
            break
    print("Finished expert run with ", t, " steps, and cumulative reward = ", cum_reward)
    reward_hist_hst.append(reward_hist)
    return reward_hist_hst

# Randomly give a test
def plot_test(agent, env, fnames=[], num_iteration=100, action_space=[-1,1], imdir='', debug=True, useVid=False):
    reward_hist_hst = []
    N=env.N
    
    if not(useVid):
        try:
            img = env.render(mode="rgb_array")
        except:
            useVid = True # https://stackoverflow.com/a/51183488
            print("Trying to use a video recording (unfinished)")
            env = wrappers.Monitor(env, "/tmp/ConsensusContEnv:ConsensusContEnv-v0")#, video_callable=False)
    
    for e,f in enumerate(fnames):
        steps = 0
#         agent.net.eval()
        agent.set_train(False)
        cum_reward = 0
        reward_hist = []

        state = env.reset()
        if agent.centralized:
            state = env.state
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if debug and not useVid:
            env.render()

        for t in range(num_iteration):  
            # Try to pick an action
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
            try:
                cum_reward += sum(reward)
            except:
                cum_reward += reward
#                 reward_hist.append([reward])
            reward_hist.append(reward)

            if len(f) > 0:
                if useVid:
                    env.render()
                else:
                    img = env.render(mode="rgb_array")
                    plt.imshow(img)
                    plt.savefig(imdir + f + '-{:03d}.jpg'.format(t))
                    plt.clf() # Clear the entire figure, but cla() (clear current axis) might also work
                    # Ref: https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
                    # Ref: https://stackoverflow.com/questions/18829472/why-does-plt-savefig-performance-decrease-when-calling-in-a-loop
            steps += 1

            if done:
                print("Took ", t, " steps to converge")
                break
        print("Finished episode ", e, " with ", t, #" steps, and rewards = ", reward, 
              ";\ncumulative reward = ", cum_reward)
        reward_hist_hst.append(reward_hist)
    env.close()
    return reward_hist_hst

# Plots out reward history data
def plot_reward_hist(reward_hists=[], ep_int=25, hist_names=[], log=True, num_iteration=0, N_list=None, bar=True, fname=''):
    # reward_hist : List of histories of reward histories
    # ep_int : number of episodes / size of intervals between two history lists
    # N_list : List of number of agents, or the denominator to average the total reward upon
    # reward_hist = [ [ [ [agent 1's reward history for 1st run in ep=0],
    #                     [agent 1's reward history for 2nd run in ep=0],... ],
    #                   [ [agent 1's reward history for 1st run in ep=25],
    #                     [agent 1's reward history for 2nd run in ep=25],... ],... ],
    #                 [ same bunch of lists for agent 2 ], ...  ]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12,7))
    fig.suptitle('Top: Mean reward; Bottom: Iteration before done')

    if num_iteration > 0:
        max_num_iteration = num_iteration
    else:
        max_num_iteration = max([max(num_iteration, max([max([len(h) for h in hh]) for hh in hhh])) for hhh in reward_hists])
    
    for i,hhh in enumerate(reward_hists):
        num_ep = np.arange(len(hhh))*ep_int
        if N_list is None:
            N = 1
        else:
            N = N_list[i]
        if log:
            re_avg = [ np.log(-1/np.mean([sum(h)/N for h in hh])) for hh in hhh ] # Mean of cumulative rewards for each ep
        else:
            re_avg = [ np.mean([sum(h)/N for h in hh]) for hh in hhh ] # Mean of cumulative rewards for each ep
        it_avg = [ np.mean([len(h) for h in hh]) for hh in hhh ] # Total number of iteartions for each ep
        max_iter_count = [ [len(h) for h in hh].count(max_num_iteration) for hh in hhh ]
        
        wid = 1 / len(hist_names) * 0.8 * ep_int
        offset = i * wid - 0.4 * ep_int
        if bar:
            ax1.bar(num_ep+offset, re_avg, label=hist_names[i], width=wid)
            ax2.bar(num_ep+offset, it_avg, label=hist_names[i], width=wid)
            ax3.bar(num_ep+offset, max_iter_count, label=hist_names[i], width=wid)
        else:
            ax1.plot(num_ep,re_avg, label=hist_names[i])
            ax2.plot(num_ep,it_avg, label=hist_names[i])
            ax3.plot(num_ep,max_iter_count, label=hist_names[i])
#     ax2.title.set_text('# of episodes trained')
    log_txt = ''
    if log:
        log_txt = 'Log '
    ax1.set_ylabel(log_txt+'Reward history (average)')
    ax2.set_ylabel('# of iterations (average)')
    ax3.set_ylabel('# of runs failed to converge')
#     ax1.set_xlabel('Reward')
    ax3.set_xlabel('# of episodes trained')
    ax2.legend(bbox_to_anchor=(1.05, 1)) # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax1.legend(bbox_to_anchor=(1.05, 1))
    ax3.legend(bbox_to_anchor=(1.05, 1))
    
    if len(fname) > 0:
        plt.savefig(fname+'.png', bbox_inches='tight') # https://stackoverflow.com/a/42303455
#         fig.savefig(fname+'_backup.png')

# Plots out loss history data
def plot_loss_hist(hists=[], hist_names=[], log=True, num_iteration=0, update_mode=UPDATE_PER_ITERATION, bar=True, fname=''):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,7))
    fig.suptitle('Top: Total (average loss per iteartion) per episode; Bottom: Iteration before done')

    if num_iteration > 0:
        max_num_iteration = num_iteration
    else:
        max_num_iteration = max([max(num_iteration, max([len(hh)] for hh in hhh)) for hhh in hists])
    
    for i,hhh in enumerate(hists):
        num_ep = np.arange(len(hhh))
        if log:
            re_avg = [ np.log(-1/sum([np.mean(h) for h in hh])) for hh in hhh ] # Mean of cumulative rewards for each ep
        else:
            # h: List of losses per iteration; hh: List of iterations per episode; hhh: List of episodes per model
            re_avg = [ sum([np.mean(h) for h in hh]) for hh in hhh ]
        iter_count = [ len(hh) for hh in hhh ]
        wid = 1 / len(hists) * 0.8 
        offset = i * wid - 0.4
        if bar:
            ax1.bar(num_ep+offset, re_avg, label=hist_names[i], width=wid)
            if update_mode==UPDATE_PER_ITERATION:
                ax2.bar(num_ep+offset, iter_count, label=hist_names[i], width=wid)
        else:
            ax1.plot(num_ep, re_avg, label=hist_names[i])
            if update_mode==UPDATE_PER_ITERATION:
                ax2.plot(num_ep, iter_count, label=hist_names[i])
    log_txt = ''
    if log:
        log_txt = 'Log '
    ax1.set_ylabel(log_txt+'Loss history (cumulative of iteration average)')
    ax2.set_ylabel('# of iterations (average)')
    ax2.set_xlabel('# of episodes trained')
    ax2.legend(bbox_to_anchor=(1.05, 1)) # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax1.legend(bbox_to_anchor=(1.05, 1))
    
    if len(fname) > 0:
        plt.savefig(fname+'.png', bbox_inches='tight')
#         fig.savefig(fname+'_backup.png')


# plot learning rate history... assuming it's updated per iteration, not per episode. Otherwise you can just
# use the two plotting methods above...
def plot_lr(hists=[], hist_names=[], log=True):

    for i,hhh in enumerate(hists):
        num_ep = np.arange(len(hhh))
        if log:
            plt.plot(num_ep, np.log(hhh), label=hist_names[i])
        else:
            plt.plot(num_ep, hhh, label=hist_names[i])
    log_txt = ''
    if log:
        log_txt = 'Log '
    plt.ylabel(log_txt+'learning rate history')
    plt.xlabel('# of steps updated')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()
    
# Check gradient history
# Ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")






