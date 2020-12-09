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

class ConsensusContEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, N=5, dt=0.1, v=0.5, v_max=1, boundaries=[-1.6,1.6,-1,1], Delta=0.02, o_radius=0.4, max_iter=200,
                 input_type=U_ACCELERATION, observe_type=O_VELOCITY, additional_features=[], observe_action=O_NO_ACTION,
                 reward_mode=ALL_REWARD, uses_boundary=True, boundary_policy=HARD_PENALTY, finish_reward_policy=END_ON_CONSENSUS,
                 start_radius=0.5):
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
        self.np = 1 # Number of additional parameters, e.g. time index
        
        self.observe_action = observe_action
        if self.observe_action:
            self.nf = self.no + len(additional_features) + self.na + self.np # self.na for observing neighbr actions
        else:
            self.nf = self.no + len(additional_features) + self.np
        # if self.observe_action == O_ACTION:
        #     self.no = self.nf
        self.max_iter = max_iter
        self.reward_mode = reward_mode

        self.input_type = input_type
        self.observe_type = observe_type
        self.uses_boundary = uses_boundary
        self.boundary_policy = boundary_policy
        self.finish_reward_policy = finish_reward_policy

        self.dt = dt
        self.v = v
        self.v_max = v_max # Max velocity
        self.a_max = 20
        self.Delta = Delta # Criteria of consensus convergence
        self.boundaries = boundaries
        self.worldW = boundaries[1]-boundaries[0]
        self.worldH = boundaries[3]-boundaries[2]
        self.o_radius = o_radius
        self.start_radius = min(o_radius, self.worldW, self.worldH) # Receiver radius
        
        # Parameters (weights) for loss terms
        self.sod_w = 400 # Sum-of-distance weight
        self.nos_w = 0.1 # Number-of-steps weight
        self.nos_base = 1.05 # Number-of-steps base weight, if needed
        self.mov_w = 5 # Magnitude-of-velocity weight
        self.oob_reward = 10 # Deduct this value for out-of-bound agents.
        # Decide the penalty value according to the boundary policy. We'll deal with the DEAD_ON_ARRIVAL policy later.
        if self.boundary_policy == NO_PENALTY:
            self.oob_reward = 0
        elif self.boundary_policy == HARD_PENALTY or self.boundary_policy == DEAD_ON_TOUCH:
            self.oob_reward = 100

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
        self.state[:2,:] = self.state[:2,:] * self.start_radius * 2 - self.start_radius
        # self.state[:2,:] = self.state[:2,:] * self.o_radius * 2 - self.o_radius
        # self.state[:2,:] = self.state[:2,:] * np.array([[self.worldW], 
        #                                                 [self.worldH]]) + np.array([[boundaries[0]], 
        #                                                                             [boundaries[2]]])
        # Randomize initial velocities
        self.state[2:4,:] = self.state[2:4,:] * self.v_max*2 - self.v_max
        # Randomize initial accelerations???
        self.state[4:,:] *= 0

        # Default adjacency matrix
        self.Adj = np.ones((self.N, self.N)) - np.eye(self.N)
        np.fill_diagonal(self.Adj,0)

        # Initialize renderer
        self.viewer = None

        # Store extra features
        self.additional_features = additional_features

        # Checks if the agents have cuddled together long enough
        self.done_count = 0
        self.done_thres = 3#5
        self.done_v_lim = 0.02*self.v_max
        self.done_a_lim = 0.02*self.a_max
        self.step_count = 0
    
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

    # Input is a 2xN array of positions. 
    # Returns a list of ints as indications: 0 means the current position is out-of-bound, and 1 means it's within bounds
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
        rewards = (out_of_bound - 1) * self.oob_reward # shape = (N,)
        # Because out_of_bound is 0 for out-of-bound ones, and 1 for ones staying inside, the (out_of_bound-1) term
        # would be -1 for bad agents and 0 for good agents. Thus, out-of-bound agents get negative reward for being bad.

        done = True

        # self.state is a NSxN array containing locations.
        # What we do want is a diff vector storing all relative positions to return as state - shape (N, NS, N).
        #     This broadcasting can be done by using shape (N,NS,1) tensor minus shape (1,NS,N) tensor.
        ### HOLD UP: Why is this following line using un-updated states???????
        # diff = self.state.T.reshape((self.N,self.ns,1)) - self.state.reshape((1,self.ns,self.N))
        diff = temp_state.T.reshape((self.N,self.ns,1)) - temp_state.reshape((1,self.ns,self.N))
        diff_norm = np.linalg.norm(diff[:,:2,:], ord=2, axis=1)
        diff_norm_unclipped = diff_norm

        # Check if it's ended / deserves to end by verifying distances between agents
        ### TODO: Proposal to make "done" criterion include near-zero speed and acceleration 
        ### HOLD UP: Why are the 2 following lines using un-updated states???????
        # done = (diff_norm <= self.Delta).all() and (np.abs(self.state[2:4]) <= self.done_v_lim).all()
        # done = done and (np.abs(self.state[4:]) <= self.done_a_lim).all()
        done = (diff_norm <= self.Delta).all() and (np.abs(temp_state[2:4]) <= self.done_v_lim).all()
        done = done and (np.abs(temp_state[4:]) <= self.done_a_lim).all()
        if done:
            # Only grant it as "done" if it has lasted long enough
            self.done_count += 1
            done = self.done_count >= self.done_thres
            # Check our ending policy
            if self.finish_reward_policy == END_ON_CONSENSUS:
                pass
            elif self.finish_reward_policy == REWARD_IF_CONSENSUS:
                done = self.step_count >= self.max_iter
                rewards += 10 * self.done_count
        else:
            self.done_count = 0
        # print(done, self.done_count, self.done_thres)

        if self.uses_boundary:
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
        else:
            # Don't need to recalculate distances, but need to shift the agents back to the center if needed.
            # Current method: Fixate the first agent to the center of the environment, and move all other agents'
            # positions based on that. Effectively leader-follower without the leader knowing that it's the leader.
            # Should work as long as nothing in the process (observation, reward, etc) is absolute-position-based.
            offset = self.state[:2,[0]]
            self.state[:2] = self.state[:2] - offset
            # if out_of_bound.all():
            #     # Don't do anything if everything's within bound
            #     pass
            # else:

        # Then we want a reward as a list.
        # If we want it as a tensor, then it's better to offload it to a separate function.
        # Currently using global sum of distances, instead of neighbor-only sums. 
        # One may argue that this might not be suitable for each agent due to local vision, but... otherwise,
        # if they can't see anything, then they'll think they miinimized the punishment!
        # One may opt to use the unclipped differences instead...
        rewards = self.find_rewards(diff_norm_unclipped, action, rewards)
        # rewards = self.find_rewards(diff_norm, action, rewards)

        ### ======= Add some more terms to rewards\
        # rewards -= np.sum(diff_norm, axis=1)

        # Calculate adjacency matrix. Credit: GRASP code
        self.Adj = (diff_norm < self.o_radius).astype(np.float32) - np.eye(self.N)
        # print(self.Adj, diff_norm)
        diff = diff * self.Adj.reshape((self.N,1,self.N))

        # Take the rows that can be observed as output
        state_observs = self.attach_nonlin_features(diff[:,:self.no,:], action)

        # Return: Observation, reward, done or not, ???
        self.step_count += 1

        # Early termination if using strict boundary policy
        if self.boundary_policy == DEAD_ON_TOUCH and (out_of_bound == 0).any():
            done = True
            rewards += (out_of_bound - 1) * oob_reward * 100
        elif self.boundary_policy == HARD_PENALTY and (out_of_bound == 0).all():
            done = True
            rewards += (out_of_bound - 1) * oob_reward * 100
        return state_observs, rewards, done, {}
    
    def find_rewards(self, diff, action, rewards=None):
        # Inputs: diff - current state differences
        #         rewards - additional things to add to the final rewards
        if rewards is None:
            rewards = np.zeros((self.N,))
        # print(rewards)

        sod_w = self.sod_w
        nos_w = self.nos_w
        nos_base = self.nos_base
        mov_w = self.mov_w

        if self.input_type == U_ACCELERATION:
            mov_w *= (self.v_max / self.a_max)

        # Find rewards from the current observable state / current state.
        # The inner call turns diff into a 2D array full of x distances.
        # The outer call sums the distances up for each agent. Sum_of_distances
        # sod = np.sum( np.linalg.norm(diff[:,:2,:], ord=2, axis=1), axis=1 )
        sod = np.sum(diff, axis=1)
        # if self.reward_mode == DIST_REWARD or self.reward_mode == ALL_REWARD:
        if self.reward_mode & DIST_REWARD:
            rewards -= sod * sod_w

        # In addition, we want to restrict agents from breaking the boundaries, and doing other bad things.
        # Boundary punshment is already insiide the provided rewards argument. 

        # We don't need to add punishment to convergence time if using accumulative reward, but still, 
        # we can add some term here. Maybe an exponential one. Number_of_steps
        nos = self.step_count
        # if self.reward_mode == TIME_REWARD or self.reward_mode == ALL_REWARD:
        if self.reward_mode & TIME_REWARD:
            # rewards -= nos * nos_w
            # rewards -= pow(nos_base, nos) * nos_w
            # rewards -= nos*nos*nos_w
            rewards -= nos*nos_w

        # Next, we could constrain the input size, be it velocity or acceleration.
        # If we're using acceleration, it might be better to downscale this thing, because accelerations' values are larger
        mov = np.linalg.norm(action, ord=2, axis=0)
        # if self.reward_mode == ACT_REWARD or self.reward_mode == ALL_REWARD:
        if self.reward_mode & ACT_REWARD:
            rewards -= mov * mov_w
        # print(sod*sod_w)
        # print(mov*mov_w)
        # print(rewards)
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
    def attach_nonlin_features(self, obsvs, action=None):
        # Input shape: obsvs (N, no, N); i.e. each agent's entire observation, all stacked together.
        ### TODO: Attach nonlinear features here.
        # Currently, the output stacks the observed neighbor actions (for the previous time) after the observations.
        # action shape shouuld be (self.na,N), the same as required in filter_neighbor_actions().
        # Then, if there's any additional parameter the agent should know of, then each of them is attached in a separate
        # slice afte the observed actions, each only showing up in the corresponding agent's index.

        # Attach observed action
        if self.observe_action == O_ACTION:
            if action is None:
                action = np.zeros((self.na,self.N))
            obsvs = np.concatenate((obsvs, self.filter_neighbor_actions(action)),axis=1)

        # Attach nonlinearities
        nonlinfeats = np.zeros((self.N,len(self.additional_features), self.N))
        for i,feat in enumerate(self.additional_features):
            try:
                nonlinfeats[:,i,:] = feat(obsvs)
            except:
                pass
        obsvs = np.concatenate((obsvs, nonlinfeats), axis=1)

        # For each observed parameter, make an identity matrix of the right shape, and slap it onto obsvs
        idmatrix = np.eye(self.N).reshape(self.N,1,self.N)
        obsvs = np.concatenate((
                obsvs,
                idmatrix * self.step_count
            ), axis=1)
        return obsvs

    def filter_neighbor_actions(self, action):
        # Input shape: action (self.na,N) is an array recording all agents' actions at some point.
        # Output shape: (N,self.na,N) array where the input is filtered according to adjacency matrix.
        # You can also use this method to obtain broadcasted actioini plan.
        # self.Adj has shape (N,N). Need to add diagonal 1s because agent should be able to see its own actions
        return (self.Adj + np.eye(self.N)).reshape(self.N,1,self.N) * action

    def reset(self):
        # Reset the state of the environment to an initial state
        self.step_count = 0

        # Initialize state space
        self.state = np.random.rand(self.ns, self.N)
        # Randomize initial positions
        self.state[:2,:] = self.state[:2,:] * self.start_radius * 2 - self.start_radius
        # self.state[:2,:] = self.state[:2,:] * self.o_radius * 2 - self.o_radius
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