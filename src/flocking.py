# 'Probabilistic Theory of Mean Field Games with Applications' eq 2.51
import math
import numpy as np
import random


N = 100
h = 0.25
kappa = 1
sigma = 0.08
T = 1000


def eta(t):
    val = kappa*(math.sqrt(N/N-1))
    val = val*math.tanh(kappa*math.sqrt((N - 1)/N)*(T - t))
    return val

def next_state(x, alpha):
    next_x = x + alpha*h + sigma*random.gauss(0,1)*math.sqrt(h)
    return(next_x)

def init_markov():
    states = np.zeros((N, int(T/h)))
    states[:,0] = np.random.normal(loc=10,scale=1,size=N)
    return states

def get_alpha(i,t):
    val = -1*(1-1/N)*eta(h*t)*(states[i,t]-np.mean(states[:,t]))
    return val

# simulate MDP
states = init_markov()
for t in range(1,int(T/h)):
    for i in range(N):
        states[i,t] = next_state(states[i,t-1], get_alpha(i,t-1))

states[:,-1] # last states for all the birds
