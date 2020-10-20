# Custom environment for multiagent consensus and similar network control problem
# Reference: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# This one takes continuous actions, and allows you to specify what are control inputs
import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np

# indicators of what kind of features we're going to use
U_VELOCITY = 1
U_ACCELERATION = 2
O_VELOCITY = 1
O_ACCELERATION = 2
# If you want nonlinear featuers, you should be responsible for passing them in during calling.
# Not implemented for now, though.

class ConsensusContEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, N=5, dt=0.1, v=0.5, v_max=1, boundaries=[-1.6,1.6,-1,1], Delta=0.01, o_radius=0.4,
                 input_type=U_ACCELERATION, observe_type=O_VELOCITY, additional_features=[]):
        super(ConsensusContEnv, self).__init__()
        
        # Store necessary info for us to simulate the environment successfully
        self.N = N
        self.nd = 2 # Number of dimensions for each agent
        self.na = self.nd # Number of actions for each agent
        self.ns = self.nd * 3 # Number of internal states for each agent: x, y, dx, dy, d2x, d2y
        # self.no = Number of observable states for each agent.
        # The layout is going to be: x, y, dx, dy (,d2x, d2y) where the parenthesis one depnds on input_type
        self.no = self.nd * (observe_type+1)
        # And we can attach custom-designed features after it
        self.nf = self.no + len(additional_features)

        self.input_type = input_type
        self.observe_type = observe_type

        self.dt = dt
        self.v = v
        self.v_max = v_max # Max velocity
        self.a_max = 20
        self.Delta = Delta # Criteria of consensus convergence
        self.boundaries = boundaries
        self.worldW = boundaries[1]-boundaries[0]
        self.worldH = boundaries[3]-boundaries[2]
        self.o_radius = min(o_radius, self.worldW, self.worldH) # Receiver radius
        
        # Define the type and shape of action_space and observation_space. 
        # Here let's just assume the velocity bounds for x and y are both [-1,1].
        # (Hint: Even when imposing this constraint, you might still have to clamp down network outputs,
        # because you can't impose boundaries on network predictions...)
        if input_type == U_ACCELERATION:
            self.action_space = spaces.Box( low=-self.a_max, high=self.a_max,
                                            shape=(self.na, self.N), dtype=np.float32 )
        elif input_type == U_VELOCITY:
            self.action_space = spaces.Box( low=-self.v_max, high=self.v_max,
                                            shape=(self.na, self.N), dtype=np.float32 )
        # How do we implement having N agents, each with an observation range equal to boundaries?
        # Not imposing boundary constraints here, because it's implemented later.
        self.observation_space = spaces.Box( low=-np.Inf, high=np.Inf, 
                                             shape=(self.no, self.N), dtype=np.float32 )
        
        # Initialize state space
        self.state = np.random.rand(self.ns, self.N)
        # # Randomize initial positions
        self.state[:2,:] = self.state[:2,:] * self.o_radius * 2 - self.o_radius
        # self.state[:2,:] = self.state[:2,:] * np.array([[self.worldW], 
        #                                                 [self.worldH]]) + np.array([[boundaries[0]], 
        #                                                                             [boundaries[2]]])
        # Randomize initial velocities
        self.state[2:4,:] = self.state[2:4,:] * self.v_max*2 - self.v_max
        # Randomize initial accelerations???
        self.state[4:,:] *= 0

        # Default adjacency matrix
        self.Adj = np.ones((self.N, self.N))
        np.fill_diagonal(self.Adj,0)

        # Initialize renderer
        self.viewer = None

        # Store extra features
        self.additional_features = additional_features
    
    # This is probably not needed for continuous action space.
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

    # Input is a 2xN array of positions. Returns a list of ints as indications.
    def is_in_bound(self, state=None):
        if state is None:
            state = self.state
        rewards = np.ones((self.N,))
        for i in range(self.N):
            if self.boundaries[0] >= state[0,i] or self.boundaries[1] <= state[0,i]:
                rewards[i] = 0
            if self.boundaries[2] >= state[1,i] or self.boundaries[3] <= state[1,i]:
                rewards[i] = 0
        return rewards
    
    # Execute one time step within the environment.
    def step(self, action):
        # Tasks include: Take action; update state; calculate reward; return the observations.
        # Assumes that action is an np array of shape (self.na, self.N).
        # This line should help make sure velocity doesn't explode:
        assert self.action_space.contains(action), "Action vector {0} is invalid".format(action)

        temp_state = self.state
        if self.input_type == U_VELOCITY:
            # Update rule when velocity is input: Change velocity and position, but not acceleration
            temp_state[:2] += self.dt * action
            temp_state[2:4] = action
        elif self.input_type == U_ACCELERATION:
            # Update rule for acceleration input: Change all fields
            temp_state[:2] += self.dt * temp_state[2:4] + 0.5 * self.dt * self.dt * action
            temp_state[2:4] += self.dt * action
            temp_state[4:] = action

        # This line should be able to check if new state exceeds boundary, and restrict them from going out
        # self.state[ :,self.is_in_bound(temp_state) > 0 ] = temp_state[ :,self.is_in_bound(temp_state) > 0 ]

        # Check which agents go out of boudnary
        out_of_bound = self.is_in_bound(temp_state)
        
        # Find reward via summed total distance for each agent. 
        oob_reward = 10 # Deduct this value for out-of-bound agents
        rewards = (out_of_bound - 1) * oob_reward 
        done = True

        # self.state is a NSxN array containing locations.
        # What we do want is a diff vector storing all relative positions to return as state - shape (N, NS, N).
        #     This broadcasting can be done by using shape (N,NS,1) tensor minus shape (1,NS,N) tensor.
        diff = self.state.T.reshape((self.N,self.ns,1)) - self.state.reshape((1,self.ns,self.N))
        diff_norm = np.linalg.norm(diff[:,:2,:], ord=2, axis=1)

        # Check if it's ended / deserves to end by verifying distances between agents
        ### TODO: Proposal to make "done" criterion include near-zero speed and acceleration 
        done = (diff_norm <= self.Delta).all()

        # Modify the state so that each agent is bounded
        self.state = temp_state
        # Bound x and y
        self.state[:2] = np.clip(self.state[:2], np.array([[self.boundaries[0],self.boundaries[2]]]).T,
                                                 np.array([[self.boundaries[1],self.boundaries[3]]]).T)
        # Bound vx and vy
        self.state[2:4] = np.clip(self.state[2:4], -self.v_max, self.v_max)
        # Bound ax and ay
        self.state[4:] = np.clip(self.state[4:], -self.a_max, self.a_max)

        # This section would try to reduce the velocity when it would lead to out-of-bound at next step
        self.state[4:] = np.minimum(self.state[4:], (self.v_max - self.state[2:4]) / self.dt)
        self.state[4:] = np.maximum(self.state[4:], (-self.v_max - self.state[2:4]) / self.dt)
        self.state[2:4] = np.minimum(self.state[2:4], 
            (np.array([[self.boundaries[1],self.boundaries[3]]]).T - self.state[:2]) / self.dt)
        self.state[2:4] = np.maximum(self.state[2:4], 
            (np.array([[self.boundaries[0],self.boundaries[2]]]).T - self.state[:2]) / self.dt)

        # Re-calculate distance
        diff = self.state.T.reshape((self.N,self.ns,1)) - self.state.reshape((1,self.ns,self.N))
        diff_norm = np.linalg.norm(diff[:,:2,:], ord=2, axis=1)

        # Calculate adjacency matrix. Credit: GRASP code
        self.Adj = (diff_norm < self.o_radius).astype(np.float32)
        # print(self.Adj, diff_norm)
        diff = diff * self.Adj.reshape((self.N,1,self.N))

        # Take the rows that can be observed as output
        state_observs = self.attach_nonlin_features(diff[:,:self.no,:])

        # Then we want a reward as a list.
        # If we want it as a tensor, then it's better to offload it to a separate function.
        # rewards = self.find_rewards(diff, rewards)
        rewards -= np.sum(diff_norm, axis=1)

        # Return: Observation, reward, done or not, ???
        return state_observs, rewards, done, {}
    
    def find_rewards(self, diff, rewards=None):
        # Inputs: diff - current state differences
        #         rewards - additional things to add to the final rewards
        if rewards is None:
            rewards = np.zeros((self.N,))

        # Find rewards from the current observable state / current state.
        # The inner call turns diff into a 2D array full of x distances.
        # The outer call sums the distances up for each agent.
        rewards -= np.sum( np.linalg.norm(diff[:,:2,:], ord=2, axis=1), axis=1 )
        return rewards

    def controller(self):
        # Returns supposedly an expert control law. 
        # Output is 2xN action array.
        diff = self.state[:2].T.reshape((self.N,2,1)) - self.state[:2].reshape((1,2,self.N)) # (N,2,N)
        vel_control = self.Adj.reshape((self.N,self.N,1)) * diff.transpose((0,2,1)) # (N,N,2)
        vel_control = -np.sum(vel_control, axis=1).T # (N,N,2) -> (N,2) -> (2,N)
        if self.input_type == U_VELOCITY: 
            return np.clip(vel_control, -self.v_max, self.v_max)
        elif self.input_type == U_ACCELERATION:
            acc_control = (vel_control - self.state[2:4]) / self.dt
            return np.clip(acc_control, -self.a_max, self.a_max)
    
    ### TODO: Attach nonlinear features behind state_observs
    def attach_nonlin_features(self, obsvs):
        # Input shape: (N, no, N); i.e. each agent's entire observation, all stacked together
        return obsvs

    def reset(self):
        # Reset the state of the environment to an initial state

        # Initialize state space
        self.state = np.random.rand(self.ns, self.N)
        # Randomize initial positions
        self.state[:2,:] = self.state[:2,:] * self.o_radius * 2 - self.o_radius
        # self.state[:2,:] = self.state[:2,:] * np.array([[self.worldW], 
        #                                                 [self.worldH]]) + np.array([[self.boundaries[0]], 
        #                                                                             [self.boundaries[2]]])
        # Randomize initial velocities
        self.state[2:4,:] = self.state[2:4,:] * self.v_max*2 - self.v_max
        self.state[4:,:] *= 0

        diff = self.state.T.reshape((self.N,self.ns,1)) - self.state.reshape((1,self.ns,self.N))
        return self.attach_nonlin_features(diff[:,:self.no,:])
    
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