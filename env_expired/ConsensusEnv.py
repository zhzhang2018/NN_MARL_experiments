# Custom environment for multiagent consensus and similar network control problem
# Reference: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
import gym
from gym import spaces

class ConsensusEnv(gym.Env):
    """Custom Environment that follows gym interface"""
#     metadata = {'render.modes': ['human']}

    def __init__(self, N=5, dt=0.1, boundaries=[-1.6,1.6,-1,1], Delta=0.01):
        super(CustomEnv, self).__init__()
        
        # Store necessary info for us to simulate the environment successfully
        self.N = N
        self.dt = dt
        self.Delta = Delta # Criteria of consensus convergence
        self.boundaries = boundaries
        self.worldW = boundaries[1]-boundaries[0]
        self.worldH = boundaries[3]-boundaries[2]
        
        # Define the type and shape of action_space and observation_space. This is similar to Tensorflow.
        # They must be gym.spaces objects. gym.spaces.Box is used for multidimensional continuous spaces with bounds.
        # https://gym.openai.com/docs/#spaces
        # Here let's just assume the velocity bounds for x and y are both [-1,1].
        self.action_space = spaces.Box( low=np.array([-1,-1]), 
                                        high=np.array([1,1]), dtype=np.float32 )
        # How do we implement having N agents, each with an observation range equao to boundaries?
        self.observation_space = spaces.Box( low=np.array([0, boundaries[0], boundaries[2]]), 
                                            high=np.array([N, boundaries[1], boundaries[3]]), dtype=np.float32 )
        
        # Initialize state space
        self.state = np.random.rand(2,N) * np.array([[self.worldW], 
                                                     [self.worldH]]) + np.array([[boundaries[0]], 
                                                                                 [boundaries[2]]])

    def step(self, action):
        # Execute one time step within the environment. Tasks include:
        # Take in an action; find the change in environment state; calculate the next reward; return the observations.
        # Assuming that action is in the shape (2,N), and contains agents' velocities
        self.state += self.dt * action
        
        # Find reward via summed total distance for each agent. Maybe we can get a different reward for each agent later.
        rewards = 0
        done = True
        for i in range(self.N):
            dists = np.linalg.norm(self.state[:,[i]] - self.state, axis=0)
            rewards -= np.sum( dists )
            done = done and ( dists <= self.Delta ).all()
        
        # Return: Observation, reward, done or not, ???
        rerturn self.state, rewards, done, {}
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.rand(2,N) * np.array([[boundaries[1]-boundaries[0]], 
                                                     [boundaries[3]-boundaries[2]]]) + np.array([[boundaries[0]], 
                                                                                                 [boundaries[2]]])
        return self.state
    
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen... it could be a simple printout, or a graphics stuff.
        screen_width = 600
        screen_height = 400
        radius = 5

        scale = min(screen_width/self.worldW, screen_height/self.worldH)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            for i in range(self.N):
                circ = rendering.make_circle(radius)
                circ.add_attr( 
                    rendering.Transform(
                        translation = ((self.state[0,i]-self.boundaries[0]) * scale, 
                                       (self.state[1,i]-self.boundaries[1]) * scale)
                    ) )
                self.viewer.add_geom(circ)

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    