# 'Probabilistic Theory of Mean Field Games with Applications' eq 2.51
import math
import numpy as np
import random


# class version
class flocking_model():
    
    def __init__(self, N=100, h=1, kappa=1, sigma=0.01, T=100):
        self.N = N
        self.h = h
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
        self.states = self.init_markov()
        self.alphas = np.zeros((self.N, int(self.T/self.h)-1))
        self.etas = np.zeros(int(self.T/self.h)-1)

    def eta(self, t):
        val = self.kappa*(math.sqrt(self.N/(self.N-1)))
        val = val*math.tanh(self.kappa*math.sqrt((self.N - 1)/self.N)*(self.T - t))
        self.etas[int(t)] = val
        return val
    
    def next_state(self, x, alpha):
        next_x = x + alpha*self.h + self.sigma*random.gauss(0,1)*math.sqrt(self.h)
        return next_x
    
    def init_markov(self):
        states = np.zeros((self.N, int(self.T/self.h)))
        states[:,0] = np.random.normal(loc=10, scale=2, size=self.N)
        return states
    
    def get_alpha(self,i,t):
        val = -1*(1-1/self.N)*self.eta(self.h*t)*(self.states[i,t]-np.mean(self.states[:,t]))
        self.alphas[i,t] = val
        return val
    
    def simulate_mdp(self):
        for t in range(1, int(self.T/self.h)):
            for i in range(self.N):
                self.states[i,t] = self.next_state(self.states[i,t-1], self.get_alpha(i, t-1))

flocking_mdp = flocking_model(N=10, h=0.25, kappa=1, sigma=0.1, T=1000)
flocking_mdp.simulate_mdp()
flocking_mdp.states[:,0]
flocking_mdp.states[:,-1]
    
# Dynamic programming


import sympy as sp
from sympy.solvers.solveset import linsolve # we will only need the linear solver, as the quadratic cost becomes linear after doing the partial derivative
import copy

sp.init_printing(use_unicode=True)

n = 3 # number of players/birds
T = 10 # number of steps 
kappa = 1

markov = []
strategies = []
strategies_solution = []
cost = []   # cost at each time point
J = []   # expected cost


def get_avg(t):
    """
    This function returns the average of the speeds of all the birds at time t
    TODO: investigate sympy to simplify it
    """
    l = []
    for pl in range(n):
        l.append(markov[pl][t])
    expr = l[0]
    for i in range(1,n):
        expr = expr+l[i]
    return(1/n*expr)


# initialization of strategies list
# I build 2-dimensional list. Rows are players. Columns are time points
# each cell of the list is a strategy (i.e. alpha)
for player in range(n):
    strategies.append([])
    strategies_solution.append([])
    for t in range(T-1):
        strategies[player].append(sp.symbols('alpha_'+str(player)+"_"+str(t+1)))
        strategies_solution[player].append(0)

# initialization of markov chain
# I build a 2-dimensional list. Rows are players. Columns are time points
# each cell of the list is a state (i.e. speed)
for player in range(n):
    markov.append([sp.symbols('x_'+str(player) + '_1')])

for player in range(n):
    for t in range(1,T):
        markov[player].append(markov[player][-1]+strategies[player][t-1])
markov


# initialization of cost function
# I build a 2-dimensional list. Rows aere players. Columns are time points
# each cell of the list is the cost value
for player in range(n):
    #print('player {}'.format(player))
    cost.append([])
    for t in range(T-1):
        #print(t)
        cost[player].append((kappa**2)/2 * (markov[player][t]-get_avg(t))**2+1/2*(strategies[player][t])**2)


# initialization expected cost
J = copy.deepcopy(cost)
for player in range(n):
    for t in range(T-1,-1,-1):
        try:
            J[player][t-1] = cost[player][t-1] + J[player][t]
        except:
            J[player][t-1] = cost[player][t-1]
          


# DYNAMIC PROGRAMMING
# t = T-1; It is the last step. Therefore alpha_T-1_i = 0 for all the players (easily seen from the expected cost function)
for player in range(n):
    # we update the markov states with the new value of alpha
    markov[player][-1] = markov[player][-1].subs(strategies[player][-1], 0)
    for t in range(T-2, -1, -1):
        for player_j in range(n):
            J[player_j][t] = J[player_j][t].subs(strategies[player][-1], 0)
    

# Recursive step
for t in range(T-2, 0, -1):
    print('Recursive step {}/{}'.format(t, T))
    system_eq = []
    relevant_symbols = []
    for player in range(n):
        system_eq.append(sp.diff(J[player][t-1], strategies[player][t-1]))
        relevant_symbols.append(strategies[player][t-1])
    # we solve system of equations
    sols = linsolve(system_eq, relevant_symbols) # it is a linear system!
    alpha_sols = next(iter(sols))   
    alpha_sols[0].free_symbols # just to check the free variables
    for player in range(n):
        # we replace the solutions in the markov chain for the correpsonding alphas
        for k in range(T, t, -1):
            for player_j in range(n):
                markov[player_j][k-1] = markov[player_j][k-1].subs(strategies[player][t-1], alpha_sols[player])         
        # we replace the solutions in the expected costs for the correpsonding alphas
        for k in range(T-2, 0, -1):
            for player_j in range(n):
                J[player_j][k-1] = J[player_j][k-1].subs(strategies[player][t-1], alpha_sols[player])
                
                
# test with random initial values
import random
initial_states = []

for player in range(n):
    initial_states.append(random.uniform(0, 10))
    
markov_test = copy.deepcopy(markov)

for player in range(n):
    for t in range(T):
        for player_j in range(n):
            markov_test[player_j][t] = markov_test[player_j][t].subs(markov[player][0], initial_states[player])

markov_test[0]
markov_test[1]
markov_test[2]

