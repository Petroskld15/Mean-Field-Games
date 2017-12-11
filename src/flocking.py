"""
This script solves the MFG of the flocking model as presented in Probabilistic Theory of Mean Field Games with Applications Chapter 2.4
I formulate it as an MDP:
    - X_i_t = speed of player i at time t
    - alpha_i_t = change of speed (or acceleration) of player i at time t. This will be the strategy to optimize

X_{t+1}^i = X_t^i + \alpha_t^i (I have not included the Wiener process)
C(X_t^i, \alpha_t^i) = \kappa^2/2 * (X_t^i-\bar{X}_t)^2 + 1/2*(\alpha_t^i)^2
J_k = \sum_{t=k}^T C(X_t^i, \alpha_t^i)
"""

import sympy as sp
from sympy.solvers.solveset import nonlinsolve, linsolve # we will only need the linear solver, as the quadratic cost becomes linear after doing the partial derivative
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

# THE BIRDS SPEEDS CONVERGE!
                

# TODO: compare this with the explicit solution from 
# 'Probabilistic Theory of Mean Field Games with Applications' eq 2.51
                
            





        
        


            

