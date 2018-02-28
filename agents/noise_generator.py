import math
import numpy as np

class OUNoise:
    
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        self.size=size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state=np.ones(self.size)*self.mu
        self.reset()
    
    def reset(self):
        self.state=self.mu
        
    def sample(self):
        x=self.state
        dx=self.theta*(self.mu-x)+self.sigma*np.random.randn(len(x))
        self.state= x+dx
        return self.state
        