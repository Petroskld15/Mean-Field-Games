# initialization of the problem
import sympy as sp
from sympy.solvers.solveset import linsolve # we will only need the linear solver, as the quadratic cost becomes linear after doing the partial derivative
import copy

sp.init_printing(use_unicode=True)

n = 3 # number of players/birds
T = 10 # number of steps 
kappa = 1

markov = []
markov2 = []
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
    markov2.append([sp.symbols('x_'+str(player) + '_1')])

for player in range(n):
    for t in range(1,T):
        markov[player].append(markov[player][-1]+strategies[player][t-1])
        markov2[player].append(sp.symbols('x_'+str(player)+'_'+str(t+1)))
markov
markov2


# initialization of cost function
# I build a 2-dimensional list. Rows aere players. Columns are time points
# each cell of the list is the cost value
for player in range(n):
    #print('player {}'.format(player))
    cost.append([])
    for t in range(T):
        #print(t)
        try:
            cost[player].append((kappa**2)/2 * (markov[player][t]-get_avg(t))**2+1/2*(strategies[player][t])**2)
        except:
            cost[player].append((kappa**2)/2 * (markov[player][t]-get_avg(t))**2)


# initialization expected cost
J = copy.deepcopy(cost)
for player in range(n):
    for t in range(T-1,0,-1):
        J[player][t-1] = cost[player][t-1] + J[player][t]

          

#######################
# dynamic programming #
#######################
# recursive step starting from the end
for t in range(T-1, 0, -1):
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
                
        # we add solution to strategies_solution
        for player_j in range(n):
            coef = sp.collect(alpha_sols[player], syms=markov[player_j][0], evaluate=False)[markov[player_j][0]]
            strategies_solution[player][t-1] = strategies_solution[player][t-1] + coef*markov2[player_j][t-1]    
                
# test with random initial values
sigma = 0.01
h = 1
simulation = []

for player in range(n):
    simulation.append([random.gauss(10, 2)])

for t in range(1,T):
    for player in range(n):
        strategy = strategies_solution[player][t-1]
        for player_j in range(n):
            strategy = strategy.subs(markov2[player_j][t-1], simulation[player_j][t-1])
        simulation[player].append(simulation[player][t-1]+strategy*h+sigma*random.gauss(0,1))

# TODO: use alphas from analytical solution, using same noise saved from the analytical solution
# analytical solution
flocking_mdp = flocking_model(N=3, h=1, kappa=1, sigma=0.01, T=10)
flocking_mdp.simulate_mdp()

# DP solution
sigma = 0.01
h = 1
simulation = []
alphas_dp = np.zeros((n, T-1))
for player in range(n):
    simulation.append([flocking_mdp.states[player,0]])

for t in range(1,T):
    for player in range(n):
        strategy = strategies_solution[player][t-1]
        for player_j in range(n):
            strategy = strategy.subs(markov2[player_j][t-1], simulation[player_j][t-1])
        alphas_dp[player, t-1] = strategy
        simulation[player].append(simulation[player][t-1]+strategy*h+flocking_mdp.wind[player, t-1])


alphas_dp
flocking_mdp.alphas

simulation_dp = np.array(simulation)
simulation_dp[:,0]
flocking_mdp.states[:,0]

simulation_dp[:,1]
flocking_mdp.states[:,1]


simulation_dp[:,-1]
flocking_mdp.states[:,-1]
