import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'inst_reward')) # Old version doesn't have inst_reward

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition. If memory is full, then replace the first one."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def discard(self, perc=0.5):
        self.memory = self.sample( int(min(max(1-pec,0),1) * len(self.memory)) )
        self.position = len(self.memory)

    def __len__(self):
        return len(self.memory)