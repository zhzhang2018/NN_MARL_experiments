import gym
import numpy as np
import time
N = 5
# env = gym.make('ConsensusEnv:ConsensusEnv-v0', N=N).unwrapped

# for i in range(5):
# 	act = np.random.randint(0,5,size=(N,)) # act = np.random.rand(2,N)*2-1
# 	next_state, reward, done, _ = env.step(act)
# 	print(next_state, reward, done)
# 	env.render()
# 	# time.sleep(1)

U_VELOCITY = 1
U_ACCELERATION = 2
O_VELOCITY = 1
O_ACCELERATION = 2
O_ACTION = 1
O_NO_ACTION = 0

DIST_REWARD = 0x1
TIME_REWARD = 0x2
ACT_REWARD = 0x4
ALL_REWARD = DIST_REWARD | TIME_REWARD | ACT_REWARD
# If you want nonlinear featuers, you should be responsible for passing them in during calling.
# Not implemented for now, though.

# Boundary policies
NO_PENALTY = 0
SOFT_PENALTY = 1 # Very light penalty
HARD_PENALTY = 2 # Very expensive penalty
DEAD_ON_TOUCH = 3 # Terminates simulation when one of them is out of bound, together with hard penalty

# Reward for achieving consensus policies
END_ON_CONSENSUS = 0
REWARD_IF_CONSENSUS = 1 # Doesn't stop, but keeps giving positive reward when seen as achieved consensus

env = gym.make('ConsensusEnv:CentralizedConvergeContEnv-v0', N=N, dt=0.1, Delta=0.05,
              input_type=U_ACCELERATION, observe_type=O_VELOCITY, uses_boundary=False,
              boundary_policy=HARD_PENALTY, finish_reward_policy=END_ON_CONSENSUS).unwrapped
# env = gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N, dt=0.1, Delta=0.05,
#               input_type=U_ACCELERATION, observe_type=O_VELOCITY, uses_boundary=False,
#               boundary_policy=HARD_PENALTY, finish_reward_policy=END_ON_CONSENSUS).unwrapped
# env = gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N, dt=0.1, Delta=0.05,
#               input_type=U_VELOCITY, observe_type=O_VELOCITY).unwrapped
env.reset()
for i in range(15):
	act = env.controller()
	# print(act)
	next_state, reward, done, _ = env.step(act)
	print(env.state)
	print(next_state, reward, done)
	# print(reward, done)
	env.render()
	time.sleep(0.5)
# Test driving in diverge
for i in range(15):
	act = env.controller()
	# print(act)
	next_state, reward, done, _ = env.step(-act)
	print(env.state)
	print(next_state, reward, done)
	# print(reward, done)
	env.render()
	time.sleep(0.5)