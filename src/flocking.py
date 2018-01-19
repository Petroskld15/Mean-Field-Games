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
        val = val*math.tanh(self.kappa*math.sqrt((self.N - 1)/self.N)*(self.T - t*self.h))
        self.etas[int(t)] = val
        return val
    
    def next_state(self, x, t):
        xi = np.random.normal(loc=0, scale=1, size=3)
        mean_states_t = np.apply_along_axis(np.mean, 0, self.states[:,t,:])
        next_x = x - self.eta(t)*(1-1/self.N)*(x-mean_states_t)*self.h + self.sigma*xi*math.sqrt(self.h)
        return next_x
    
    def init_markov(self):
        states = np.zeros((self.N, int(self.T/self.h),3))
        states[:,0] = np.random.normal(loc=0, scale=0.3, size=(self.N,3))
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
flocking_mdp = flocking_model(N=50, h=0.05, kappa=1, sigma=0.2, T=100)
#flocking_mdp.states[:,0,:]
x1 = flocking_mdp.next_state(flocking_mdp.states[0,0,:],0)
flocking_mdp.simulate_mdp()
flocking_mdp.states[:,-1,:]


# We plot the final results velocity vectors (coordinates X and Y) for all the birds
import matplotlib.pyplot as plt


X = np.zeros(flocking_mdp.N)
Y = np.zeros(flocking_mdp.N)
U = flocking_mdp.states[:,-1,0]
V = flocking_mdp.states[:,-1,1]
C = np.linspace(0,1,flocking_mdp.N)

plt.figure()
Q = plt.quiver(X,Y,U,V,C, units = 'xy', cmap='jet', width = 0.0005)

# We plot the paths (coordinates X and Y) for all the birds
X_0 = np.zeros(flocking_mdp.states[:,:,0].shape)
Y_0 = np.zeros(flocking_mdp.states[:,:,0].shape)
U = flocking_mdp.states[:,:,0]
V = flocking_mdp.states[:,:,1]
X = X_0
Y = Y_0
for n in range(0, X.shape[0]):
    for t in range(1,X.shape[1]):
        X[n,t] = X[n,t-1]+ flocking_mdp.h*U[n,t-1]
        Y[n,t] = Y[n,t-1]+ flocking_mdp.h*V[n,t-1]
    
plt.figure()
Q = plt.quiver(X,Y,U,V,C, units = 'xy', cmap='jet',width = 0.005)





    