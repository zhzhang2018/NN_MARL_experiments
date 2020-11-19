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
env = gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N, dt=0.1, Delta=0.05,
              input_type=U_ACCELERATION, observe_type=O_VELOCITY).unwrapped
# env = gym.make('ConsensusEnv:ConsensusContEnv-v0', N=N, dt=0.1, Delta=0.05,
#               input_type=U_VELOCITY, observe_type=O_VELOCITY).unwrapped
env.reset()
for i in range(15):
	act = env.controller()
	# print(act)
	next_state, reward, done, _ = env.step(act)
	# print(next_state, reward, done)
	env.render()
	time.sleep(0.5)
# Test driving in diverge
for i in range(25):
	act = env.controller()
	# print(act)
	next_state, reward, done, _ = env.step(-act)
	# print(next_state, reward, done)
	# print(env.state)
	env.render()
	time.sleep(0.5)