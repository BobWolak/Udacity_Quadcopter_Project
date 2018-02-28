import random
from collections import namedtuple

Experience = namedtuple("Experience",
                        field_names=["state","action","reward","next_state","done"])

class Replay_Buffer_Hover:
    
    def __init__(self, size):
        
        self.size = size
        self.memory = []
        self.idx =0
        
        
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx+1)%self.size
       
        
    def sample(self, batch_size):
        self.batch_size=batch_size
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        return len(self.memory)
    
            