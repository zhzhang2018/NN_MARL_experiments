import numpy as np
import math

class UAVEnv():
    def __init__(self):
        self.dt = 0.001
        self.n_actions = 3
        self.n_states = 3
        self.alpha = 3
        self.target = [50, 100]
        self.g = 0.5
        self.viewer = None # Added to avoid error in self.close()

    def step(self, action):
        self.state += self.dt*10*action
        return self.state, self.reward(self.state, action), False, {} # Added to follow standard env implementation

    def reward(self, state, action):
        r = -(0.5*sum(action**2) -
              0.5 * state[2]*self.g*(
                  math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(state[0:2], self.target))) # Added to cover math.dist()
              )**(-self.alpha))
        return r

    def reset(self):
        self.state = np.random.rand(self.n_states, 1)

    def render(self):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

