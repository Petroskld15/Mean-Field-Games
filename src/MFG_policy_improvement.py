# THIRD IMPLEMENTATION: we solve the ODE directly using Euler
###### SECOND IMPLEMENTATION USING SCIPY
from scipy.integrate import quad, ode, odeint, simps
import numpy as np
from numpy.linalg import norm
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

PATH_IMAGES = '/Users/msabate/Projects/Turing/Mean-Field-Games/images' # save this


class Policy_Iteration_Euler():    
    """
    Policty iteration of the LQC problem:
        dx = (bx+c*alpha)dt + sigma dW_t
        J(t,x) = E \int_t^T[b_f*x^2 + c_f*alpha^2 + d_f*law(x)^2 + e_f*x*law(x)] + gamma*(x_T)
    
    """
    
    
    def __init__(self,law, b=0.5, c=0.5, m=0, sigma=1, b_f=0.5, c_f=0.9, d_f=1, e_f=-2, gamma=1, T=10, init_t=9, solver='Euler',
                 init_p1=lambda t: -0.01*t, init_p2=lambda t: 0.1, p3=lambda t:1-1/(t+1)):
        
        self.b = b
        self.c = c
        self.m = m
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.d_f = d_f
        self.e_f = e_f
        self.gamma = gamma
        self.init_t = init_t
        self.T = T
        self.n = 1 # step of iteration. VERY IMPORTANT
        self.time = np.arange(self.init_t,self.T,0.01) # time is discretised in 1000 points
        self.init_p1 = init_p1 
        self.init_p2 = init_p2 
        self.p1_grid, self.p2_grid = [], []
        self.p3_grid = np.array([p3(t) for t in self.time])
        self.law_grid = law #np.array([law(t) for t in self.time])
        self.init_p()
        self.beta, self.phi, self.delta = [], [], [] # this will store the functions beta(t), phi(t),
        self.solver = solver
        
    def init_p(self):
        """
        Alpha_0 needs to be linear in x
        """
        self.p1_grid.append(np.array([self.init_p1(t) for t in self.time]))
        self.p2_grid.append(np.array([self.init_p2(t) for t in self.time]))
        self.p2_grid[0] = self.p2_grid[0] + self.p3_grid*self.law_grid
        #self.p2_debug = np.array([self.init_p2(t) for t in self.time])
        
        
    def get_alpha(self,x):
        """
        get alpha^n(t,x), where n is the last step of the iteration
        """
        res = x*self.p1_grid[-1] + self.p2_grid[-1]
        return res  # a vector of length time
    
    def get_value_function(self, x):
        beta_array = np.array(self.beta[-1])
        delta_array = np.array(self.delta[-1])
        phi_array = np.array(self.phi[-1])
        res = (x**2)*beta_array + x*delta_array + phi_array
        return res

    def evaluation_step(self):
        """
        This function solves the PDE given by HJB for MFG at the current iteration
        To solve a PDE we make a guess w of the solution, we take coefficients of
        x^2, x and 1, and create a system of 3 ODEs. 
        we create the system of three equations below
        
        We update the values of p1 and p2, that will be used in the next iteration
        
        Output:
            solution of the PDE
        """
        current_step = self.n
        # first ode: d beta(t) = (beta0(t) + beta1(t)beta(t))dt
        beta0 = [-(self.b_f + self.c_f*self.p1_grid[current_step-1][t]**2) for t in range(len(self.time))]
        beta1 = [-(2*self.b + 2*self.c*self.p1_grid[current_step-1][t]) for t in range(len(self.time))]
        if self.solver=='Euler':
            self.beta.append(self._solve_ode_euler(beta0, beta1, self.gamma)) # beta is a funcation lambda
        else:
            self.beta.append(self._solve_ode_explicit(beta0, beta1, self.gamma)) # beta is a funcation lambda
        
        # second ode: d delta(t) = (delta0(t) + delta1(t)delta(t))dt
        delta0 = [-(2*self.c_f * self.p1_grid[current_step-1][t] * self.p2_grid[current_step-1][t]) for t in range(len(self.time))]  
        delta0 = [delta0[t]- (self.e_f*self.law_grid[t]) for t in range(len(self.time))]
        delta0 = [delta0[t] - (2*self.c*self.beta[current_step-1][t]*self.p2_grid[current_step-1][t]) for t in range(len(self.time))]
        delta0 = [delta0[t] - self.m*self.law_grid[t]*2*self.beta[current_step-1][t] for t in range(len(self.time))] 
        delta1 = [-(self.b + self.c*self.p1_grid[current_step-1][t]) for t in range(len(self.time))]
        if self.solver == 'Euler':
            self.delta.append(self._solve_ode_euler(delta0, delta1, 0)) # delta is a function lambda
        else:
            self.delta.append(self._solve_ode_explicit(delta0, delta1, 0)) # delta is a function lambda
            
        # third ode: d phi = (phi0(t) + phi1(t)phi(t))dt
        phi0 =  [-self.sigma**2*self.beta[current_step-1][t] for t in range(len(self.time))]
        phi0 = [phi0[t] - self.c_f*self.p2_grid[current_step-1][t]**2 for t in range(len(self.time))]
        phi0 = [phi0[t] - self.d_f*self.law_grid[t]**2 for t in range(len(self.time))]
        phi0 = [phi0[t] - self.c*self.delta[current_step-1][t]*self.p2_grid[current_step-1][t] for t in range(len(self.time))]
        phi0 = [phi0[t] - self.m*self.law_grid[t]*self.delta[current_step-1][t] for t in range(len(self.time))]
        phi1 = [0]*len(self.time)
        if self.solver == 'Euler':
            self.phi.append(self._solve_ode_euler(phi0, phi1, 0)) # phi is a function lambda`A
        else:
            self.phi.append(self._solve_ode_explicit(phi0, phi1, 0)) # phi is a function lambda`A
        
        
        # we update p1 and p2:
        p1_new = np.array([-self.c/(2*self.c_f)*2*self.beta[current_step-1][t] for t in range(len(self.time))])
        p2_new = np.array([-self.c/(2*self.c_f)*self.delta[current_step-1][t] for t in range(len(self.time))])
        self.p1_grid.append(p1_new)
        self.p2_grid.append(p2_new)
        self.n += 1
        
        
    def _solve_ode_euler(self, a0, a1, y_T):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = y_T
        Input:
            - a0: grid of points between [0,T]
            - a1: grid of points between [0,T]
            - y_T = y(T)
        
        Output:
            - solution of ODE evaluated at the grid of time points beteen [0,T]
        Note: We apply Euler method to solve this ODE 
        """
        # y_T is terminal condition
        y0 = y_T
        res = [0]*len(self.time)
        res[-1] = y0
        for t in range(len(self.time)-1, 0, -1):
            m = a0[t]+a1[t]*res[t]
            y1 = y0 - m*(self.time[t]-self.time[t-1])
            res[t-1] = y1
            y0 = y1
        return res
    






class MFG():
    """
    Policy iteration of the MFG problem
    dx = (bx+x*alpha+m*law)dt + sigma*dW
    J(x,t) =  E \int_t^T[b_f*x^2 + c_f*alpha^2 + d_f*law(x)^2 + e_f*x*law(x)] + gamma*(x_T)
    """    
    def __init__(self, law_0=10, b=0.5, c=0.5, m=0, sigma=1, b_f=0.5, c_f=0.9, d_f=1, e_f=-2, gamma=1, 
                 T=10, init_t = 0, solver='Euler', init_law = lambda t: 0.5*t,
                 init_p1_alpha = lambda t: 1-1/(t+1), init_p2_alpha = lambda t:0.1, p3_alpha = lambda t: 0.1*t):
        self.law_0 = law_0
        self.b = b
        self.c = c
        self.sigma = sigma
        self.m = m
        self.b_f = b_f
        self.c_f = c_f
        self.d_f = d_f
        self.e_f = e_f
        self.gamma = gamma
        self.init_t = init_t
        self.T = T
        self.n = 1 # step of iteration. VERY IMPORTANT
        self.time = np.arange(self.init_t,self.T,0.01) # time is discretised in 1000 points
        self.init_law = init_law
        self.solver = solver        
        
        self.p1 = init_p1_alpha
        self.p2 = init_p2_alpha
        self.p3 = p3_alpha
        self.law=[np.array([self.law_0 + init_law(t) for t in self.time])]
        
    def solve_policy(self):
        pol = Policy_Iteration_Euler(b=self.b, c=self.c, m=self.m, sigma=self.sigma, 
                                     b_f=self.b_f, c_f=self.c_f, d_f=self.d_f, e_f=self.e_f,
                                     gamma=self.gamma, T=self.T,init_t=self.init_t, solver=self.solver,
                                     init_p1 = self.p1, init_p2=self.p2, p3=self.p3, law = self.law[-1])
        x = np.linspace(0,100,101)
        step = 0
        tol = 0.0001
        diff = 1
        alphas = []
        value_functions = []
        
        alpha = np.array([pol.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
        alphas.append(alpha)
        
        while diff > tol:
            print('diff is {}'.format(diff))
            pol.evaluation_step()
            alpha = np.array([pol.get_alpha(x_i) for x_i in x])
            alphas.append(alpha)
            value = np.array([pol.get_value_function(x_i) for x_i in x])
            value_functions.append(value)
            diff = norm(alphas[-1]-alphas[-2], ord='fro')
        
        return pol
    
    def update_law(self, pol):
        
        p1_grid = pol.p1_grid[-1]
        p2_grid = pol.p2_grid[-1]
        p3_grid = np.array([self.p3(t) for t in self.time])
        p2_grid = p2_grid - p3_grid * self.law[-1]
        
        a0 = self.c*p2_grid 
        a1 = self.b + self.c*p1_grid + self.c*p3_grid + self.m
        
        y_0 = self.law_0
        current_law = self._solve_ode_euler(a0,a1,y_0)
        self.law.append(np.array(current_law))

    
    def _solve_ode_euler(self, a0, a1, y_0):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(0) = y_0
        Input:
            - a0: grid of points between [0,T]
            - a1: grid of points between [0,T]
            - y_T = y(T)
        
        Output:
            - solution of ODE evaluated at the grid of time points beteen [0,T]
        Note: We apply Euler method to solve this ODE 
        """
        # y_T is terminal condition
        y0 = y_0
        res = [0]*len(self.time)
        res[0] = y0
        for t in range(len(self.time)-1):
            m = a0[t]+a1[t]*res[t]
            y1 = y0 + m*(self.time[t]-self.time[t-1])
            res[t+1] = y1
            y0 = y1
        return res
        
    
                    
game = MFG(init_t=9)            
law = game.law[-1]

# DEbUGGING Policy Iteration
pol = Policy_Iteration_Euler(law=law)
pol.p1_grid
pol.p2_grid
pol.law_grid
pol.p2_debug
x = np.linspace(0,100,101)
n_iterations=10

alphas = []
value_functions = []

alpha = np.array([pol.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
alphas.append(alpha)
alphas

for i in range(n_iterations):
    print(i)
    pol.evaluation_step()
    alpha = np.array([pol.get_alpha(x_i) for x_i in x])
    alphas.append(alpha)
    value = np.array([pol.get_value_function(x_i) for x_i in x])
    value_functions.append(value)

diff_alphas = [norm(alphas[i+1]-alphas[i], ord='fro') for i in range(len(alphas)-1)]
diff_value = [norm(value_functions[i+1]-value_functions[i], ord='fro') for i in range(len(value_functions)-1)]
     

# DEBUGGING MFG
game = MFG(init_t=9) 
pol = game.solve_policy()
pol.p1_grid
pol.p2_grid
game.update_law(pol)
game.law
pol = game.solve_policy()
pol.p2_grid[-1]
pol.p1_grid[-1]
game.update_law(pol)
            
    
    
        
    
    
    
    
    
    
    
    


