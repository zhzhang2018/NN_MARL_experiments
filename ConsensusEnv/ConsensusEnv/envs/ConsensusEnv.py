# Custom environment for multiagent consensus and similar network control problem
# Reference: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np

class ConsensusEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, N=5, dt=0.1, v=0.5, boundaries=[-1.6,1.6,-1,1], Delta=0.01):
        super(ConsensusEnv, self).__init__()
        
        # Store necessary info for us to simulate the environment successfully
        self.N = N
        self.dt = dt
        self.v = v
        self.Delta = Delta # Criteria of consensus convergence
        self.boundaries = boundaries
        self.worldW = boundaries[1]-boundaries[0]
        self.worldH = boundaries[3]-boundaries[2]
        
        # Define the type and shape of action_space and observation_space. This is similar to Tensorflow.
        # They must be gym.spaces objects. gym.spaces.Box is used for multidimensional continuous spaces with bounds.
        # https://gym.openai.com/docs/#spaces
#         # Here let's just assume the velocity bounds for x and y are both [-1,1].
#         self.action_space = spaces.Box( low=np.array([-1,-1]), 
#                                         high=np.array([1,1]), dtype=np.float32 )
        # Turns out that we probably have to first try discrete actions (up,left,right,down,stay)
        self.action_space = spaces.MultiDiscrete([5 for i in range(N)])
        # How do we implement having N agents, each with an observation range equao to boundaries?
        self.observation_space = spaces.Box( low=np.array([0, boundaries[0], boundaries[2]]), 
                                            high=np.array([N, boundaries[1], boundaries[3]]), dtype=np.float32 )
        
        # Initialize state space
        self.state = np.random.rand(2,N) * np.array([[self.worldW], 
                                                     [self.worldH]]) + np.array([[boundaries[0]], 
                                                                                 [boundaries[2]]])
        # Initialize renderer
        self.viewer = None
    
    def translate_action_to_v(self, action):
        motion = np.zeros((2, self.N))
        for i,a in enumerate(action):
            if a == 0:
                # Up
                motion[:,i] = [0, self.v]
            elif a == 1:
                # Left
                motion[:,i] = [-self.v, 0]
            elif a == 2:
                # Right
                motion[:,i] = [self.v, 0]
            elif a == 3:
                # Down
                motion[:,i] = [0, -self.v]
            else:
                # Stay
                motion[:,i] = [0, 0]
                # But if you want to remove idle state possibility, then do this:
                motion[:,i] = [-self.v, 0]
        return motion

    def is_in_bound(self, state=None):
        # Input is a 2xN array of positions. Returns a list of bools
        if state is None:
            state = self.state
        # (IGNORE this line) not rewards - 0 for staying in bound, negative for being out
        punishmentx = self.boundaries[1]-self.boundaries[0] 
        punishmenty = self.boundaries[3]-self.boundaries[2]
        rewards = np.ones((self.N,))
        for i in range(self.N):
            if self.boundaries[0] >= state[0,i] or self.boundaries[1] <= state[0,i]:
                # rewards[i] -= punishmentx
                rewards[i] = 0
            if self.boundaries[2] >= state[1,i] or self.boundaries[3] <= state[1,i]:
                # rewards[i] -= punishmenty
                rewards[i] = 0
        return rewards
    
    def step(self, action):
        # Execute one time step within the environment. Tasks include:
        # Take in an action; find the change in environment state; calculate the next reward; return the observations.
        # Assuming that action is in the shape (2,N), and contains agents' velocities
        # For now, assume that action is in the shape (N,), belonging to 1 of the 5 possible action steps
        assert self.action_space.contains(action), "Action vector {0} is invalid".format(action)
#         print(action)
#         self.state += self.dt * action
        # self.state += self.dt * self.translate_action_to_v(action)
        temp_state = self.state + self.dt * self.translate_action_to_v(action)
#         print(self.is_in_bound(temp_state) > 0)
        self.state[ :,self.is_in_bound(temp_state) > 0 ] = temp_state[ :,self.is_in_bound(temp_state) > 0 ]
        
        # Find reward via summed total distance for each agent. Maybe we can get a different reward for each agent later.
        rewards = []
        done = True
        # self.state is a 2xN array containing locations.
        # What we want is a diff vector storing all relative positions to return as state - shape (N, 2, N or N-1).
        #     This broadcasting can be done by using shape (N,2,1) tensor minus shape (1,2,N) tensor.
        diff = self.state.T.reshape((self.N,2,1)) - self.state.reshape((1,2,self.N))
#         print(self.state)
#         print(self.state.T.reshape((self.N,2,1)))
#         print(self.state.reshape((1,2,self.N)))
#         print(diff)
        # Then we want a reward as a list, where each element is the sum of distances (or sqrt of dists) from other agents.
        rewards = -np.sum( np.linalg.norm(diff, ord=2, axis=1), axis=1 )
        # print(rewards,self.is_in_bound() )
        rewards += self.is_in_bound()
#         print(rewards)
#         for i in range(self.N):
#             dists = np.linalg.norm(self.state[:,[i]] - self.state, axis=0)
#             rewards.append( -np.sum( dists ) )
#             done = done and ( dists <= self.Delta ).all()
        done = (np.linalg.norm(diff, ord=2, axis=1) <= self.Delta).all()
        # Return: Observation, reward, done or not, ???
        return diff, rewards, done, {}
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.rand(2,self.N) * np.array(
            [[self.boundaries[1]-self.boundaries[0]], 
             [self.boundaries[3]-self.boundaries[2]]]) + np.array([[self.boundaries[0]], [self.boundaries[2]]])
        return self.state.reshape((self.N,2,1)) - self.state.reshape((1,2,self.N))
#         return self.state
    
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen... it could be a simple printout, or a graphics stuff.
        # Ref: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        screen_width = 600
        screen_height = 400
        radius = 5

        scale = min(screen_width/self.worldW, screen_height/self.worldH)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.circletrans = []
            
            for i in range(self.N):
                circ = rendering.make_circle(radius)
                circtrans = rendering.Transform()
                circ.add_attr( 
#                     rendering.Transform(
#                         translation = ((self.state[0,i]-self.boundaries[0]) * scale, 
#                                        (self.state[1,i]-self.boundaries[2]) * scale) )
                    circtrans )
                self.viewer.add_geom(circ)
                self.circletrans.append(circtrans)

        if self.state is None:
            return None
        
        for i in range(self.N):
            self.circletrans[i].set_translation( 
                                     (self.state[0,i]-self.boundaries[0]) * scale, 
                                     (self.state[1,i]-self.boundaries[2]) * scale ) 

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None