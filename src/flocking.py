'''
Flocking model
'''
import math
import numpy as np
import random


# eq 2.51 of 'Probabilistic Theory of Mean Field Games with Applications' with optimal alphas
class flocking_model():
    
    def __init__(self, N=100, h=1, kappa=1, sigma=0.01, T=100):
        self.N = N
        self.h = h
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
        self.states = self.init_markov()
        self.alphas = np.zeros((self.N, int(self.T/self.h)-1))
        self.Wiener = np.zeros((self.N, int(self.T/self.h)-1))
        self.etas = np.zeros(int(self.T/self.h)-1)

    def eta(self, t):
        val = self.kappa*(math.sqrt(self.N/(self.N-1)))
        val = val*math.tanh(self.kappa*math.sqrt((self.N - 1)/self.N)*(self.T - t))
        self.etas[int(t)] = val
        return val
    
    def next_state(self, x, t):
        xi = np.random.normal(loc=0, scale=1, size=3)
        mean_states_t = np.apply_along_axis(np.mean, 0, self.states[:,t,:])
        next_x = x - self.eta(t)*(1-1/self.N)*(x-mean_states_t)*self.h + self.sigma*xi*math.sqrt(self.h)
        return next_x
    
    def init_markov(self):
        states = np.zeros((self.N, int(self.T/self.h),3))
        states[:,0] = np.random.normal(loc=10, scale=2, size=(self.N,3))
        return states
    
#    def get_alpha(self,i,t):
#        val = -1*(1-1/self.N)*self.eta(self.h*t)*(self.states[i,t]-np.mean(self.states[:,t]))
#        self.alphas[i,t] = val
#        return val
    
    def simulate_mdp(self):
        for t in range(1, int(self.T/self.h)):
            for i in range(self.N):
                #noise = self.sigma*random.gauss(0,1)
                self.states[i,t] = self.next_state(self.states[i,t-1], t-1)
                

# example
flocking_mdp = flocking_model(N=3, h=1, kappa=1, sigma=0.1, T=10)
flocking_mdp.states[:,0,:]
x1 = flocking_mdp.next_state(flocking_mdp.states[0,0,:],0)
flocking_mdp.simulate_mdp()
flocking_mdp.states[:,-1,:]
                
                






    