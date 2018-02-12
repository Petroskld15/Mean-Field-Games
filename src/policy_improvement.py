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
    
    def __init__(self, x_0=0, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=1, T=10, init_t = 9, solver='Euler', timestep=0.05):
        self.x_0 = x_0
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        self.init_t = init_t
        self.T = T
        self.n = 1 # step of iteration. VERY IMPORTANT
        self.timestep = 0.05
        self.time = np.arange(self.init_t,self.T+0.1*timestep,timestep) # time is discretised in 1000 points
        self.init_p1 = None 
        self.init_p2 = None 
        self.p1_grid, self.p2_grid = [], []
        self.init_p()
        self.beta, self.phi, self.delta = [], [], [] # this will store the functions beta(t), phi(t),
        self.solver = solver
        
    def init_p(self):
        """
        Alpha_0 needs to be linear in x
        """
        #self.p1 = self.t  
        self.init_p1= lambda t: -0.5
        self.init_p2 = lambda t: 0.1
        self.p1_grid.append(np.array([self.init_p1(t) for t in self.time]))
        self.p2_grid.append(np.array([self.init_p2(t) for t in self.time]))
    
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
        This function solves the PDE given by HJB at the current iteration
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
        delta0 = [-(2*self.c_f * self.p1_grid[current_step-1][t] * self.p2_grid[current_step-1][t] + 2*self.c*self.beta[current_step-1][t]*self.p2_grid[current_step-1][t]) for t in range(len(self.time))]
        delta1 = [-(self.b + self.c*self.p1_grid[current_step-1][t]) for t in range(len(self.time))]
        if self.solver == 'Euler':
            self.delta.append(self._solve_ode_euler(delta0, delta1, 0)) # delta is a function lambda
        else:
            self.delta.append(self._solve_ode_explicit(delta0, delta1, 0)) # delta is a function lambda
            
        # third ode: d phi = (phi0(t) + phi1(t)phi(t))dt
        phi0 =  [-(self.sigma**2*self.beta[current_step-1][t] + self.c_f*self.p2_grid[current_step-1][t]**2 + self.c*self.delta[current_step-1][t]*self.p2_grid[current_step-1][t]) for t in range(len(self.time))]
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
    
    def _solve_ode_explicit(self, a0, a1, y_T):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = gamma
        The solution is:
            y(s) = y(T)*exp[-int_s^T a1(t)dt] - int_s^T[a0(t)exp[-int_s^t a1(r)dr]]dt
        
        Input:
            - a0: function
            - a1: function
            - y_T: value of y(T)
        
        Output:
            - solution of ODE at time grid points. 
            
        Note: we use Simpson's rule to solve the integrals
        Note: This ffunction still doesn't work. Need to be corrected
        """
        a0 = np.array(a0)
        a1 = np.array(a1)
        integrand_const = -a1
        val_integral_const = simps(integrand_const, self.time)
        sol = np.zeros_like(self.time)
        for i in range(len(self.time)):
            integrand2 = np.copy(a0[i:])
            for j in range(i, len(self.time)):
                integrand3 = -np.copy(a1[:j+1])
                val_integral3 = simps(integrand3, self.time[:j+1])
                integrand2[j-i] = integrand2[j-i]*math.exp(val_integral3)
            val_integral2 = simps(integrand2, self.time[i:])
            integrand4 = -a1[:i+1]
            try:
                val_integral4 = simps(integrand4, self.time[:i+1])
            except:
                val_integral4 = 0
            #sol[i] = math.exp(-val_integral4)*(y_T*math.exp(val_integral_const) - val_integral2)
            sol[i] = y_T*math.exp(val_integral_const-val_integral4) - math.exp(-val_integral4)*val_integral2
        return(sol)
    
    



class HJB_LQR():
    
    def __init__(self, x_0=0, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=1, T=10, init_t = 9, solve_ode=True, timestep=0.05):
        self.x_0 = x_0
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        self.init_t = init_t
        self.T = T
        self.timestep = timestep
        self.time = np.arange(self.init_t,self.T+0.1*timestep,timestep) # time is discretised in 1000 points
        self.beta, self.phi, self.delta = None, None, None # this will store the functions beta(t), phi(t)
        self.solve_ode = solve_ode
    
    def _system_odes(self):

        beta0 = np.ones(self.time.shape[0])*(-self.b_f)
        beta1 = np.ones(self.time.shape[0])*(-2*self.b)
        beta2 = np.ones(self.time.shape[0])* (self.c**2 / self.c_f)
        self.beta = self._solve_ode_explicit(beta0, beta1, beta2, self.gamma)
        
        delta0 = np.zeros(self.time.shape[0])
        delta1 = np.ones(self.time.shape[0])*(-self.b+self.beta*self.c**2/self.c_f)
        delta2 = np.zeros(self.time.shape[0]) 
        self.delta = self._solve_ode_explicit(delta0, delta1, delta2, 0)
        
        phi0 = -self.beta * self.sigma**2 + self.c**2/(4*self.c_f) * self.delta**2
        phi1 = np.zeros(self.time.shape[0])
        phi2 = np.zeros(self.time.shape[0])
        self.phi = self._solve_ode_explicit(phi0, phi1, phi2, 0)
    
    def _explicit_solution(self):
        A = -self.c**2 / self.c**2
        B = math.sqrt(-self.b**2 - A*self.b_f)
        C = math.sqrt(B)
        # TODO
        return 1    
    
    def _explicit_solution(self):
        return 1
    
    def get_value_function(self,x): 
        if self.solve_ode:
            self._system_odes()
            res = (x**2)*self.beta + x*self.delta + self.phi
            return res
        else:
            #TODO
            return 1
        
    def get_alpha(self, x):
        if self.solve_ode:
            self._system_odes()
            res = -self.c/self.c_f * self.beta * x
            return res
        else:
            return 1


            
    def _solve_ode_explicit(self, a0, a1, a2, y_T):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = gamma
        
        Input:
            - a0: function
            - a1: function
            - y_T: value of y(T)
        
        Output:
            - solution of ODE at time grid points. 
        Note: this is solved using Euler method
        """
        a0 = np.array(a0)
        a1 = np.array(a1)
        a2 = np.array(a2)
        res = np.zeros(self.time.shape[0])            
        y0 = y_T
        #res = [0]*len(self.time)
        res[-1] = y0
        for t in range(len(self.time)-1, 0, -1):
            m = a0[t]+a1[t]*res[t]+(a2[t]*res[t]**2)
            y1 = y0 - m*(self.time[t]-self.time[t-1])
            res[t-1] = y1
            y0 = y1
        return res

    
    
    



def compare_ode_implementations():
    pol1 = Policy_Iteration_Euler()
    x = np.linspace(0,100,101)
    n_iterations=10
    
    alphas = []
    value_functions = []
    
    alpha = np.array([pol1.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
    alphas.append(alpha)
    
    for i in range(n_iterations):
        pol1.evaluation_step()
        alpha = np.array([pol1.get_alpha(x_i) for x_i in x])
        alphas.append(alpha)
        value = np.array([pol1.get_value_function(x_i) for x_i in x])
        value_functions.append(value)
    
    diff_alphas = [norm(alphas[i+1]-alphas[i], ord='fro') for i in range(len(alphas)-1)]
    diff_value = [norm(value_functions[i+1]-value_functions[i], ord='fro') for i in range(len(value_functions)-1)]
    
    pol2 = Policy_Iteration_Euler(solver='Explicit')
    x = np.linspace(0,100,101)
    n_iterations=10
    
    alphas2 = []
    value_functions2 = []
    
    alpha = np.array([pol2.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
    alphas2.append(alpha)
    
    for i in range(n_iterations):
        pol2.evaluation_step()
        alpha = np.array([pol2.get_alpha(x_i) for x_i in x])
        alphas2.append(alpha)
        value = np.array([pol2.get_value_function(x_i) for x_i in x])
        value_functions2.append(value)
    
    diff_alphas = [norm(alphas2[i+1]-alphas2[i], ord='fro') for i in range(len(alphas2)-1)]
    diff_value = [norm(value_functions[i+1]-value_functions[i], ord='fro') for i in range(len(value_functions)-1)]
      

        
        
        
        
def iterate(x_0, b, c, b_f, c_f, gamma, T, init_t, n_iterations, solver, timestep):
#    x_0 = 1
#    b = 1
#    c = 1
#    sigma = 1
#    b_f = 1
#    c_f = 100
#    gamma = 100
#    T = 10
#    init_t = 0
#    n_iterations = 20   
#    solver = 'Euler'
        
        
    pol = Policy_Iteration_Euler(x_0=x_0, b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, 
                                 gamma=gamma, T=T, init_t =init_t, solver=solver, timestep=timestep)
    #pol = Policy_Iteration_Euler()
    x = np.linspace(0,100,101)
    n_iterations=50
    
    alphas = []
    value_functions = []
    
    alpha = np.array([pol.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
    alphas.append(alpha)
    
    for i in range(n_iterations):
        print('iteration = {}'.format(i))
        pol.evaluation_step()
        alpha = np.array([pol.get_alpha(x_i) for x_i in x])
        alphas.append(alpha)
        value = np.array([pol.get_value_function(x_i) for x_i in x])
        value_functions.append(value)
    
    diff_alphas = [norm(alphas[i+1]-alphas[i], ord='fro') for i in range(len(alphas)-1)]
    diff_value = [norm(value_functions[i+1]-value_functions[i], ord='fro') for i in range(len(value_functions)-1)]
    
    return pol, alphas, value_functions, diff_alphas, diff_value




if __name__=='__main__':
    x_0 = 1
    b = 1
    c = 1
    sigma = 1
    b_f = 1
    c_f = 1
    gamma = 0
    T = 10
    init_t = 9
    n_iterations = 50   
    solver = 'explicit'    
    timestep = 0.005
    
    # new Policy Iteration
    pol, alphas, value_functions, diff_alphas, diff_value = iterate(x_0, b, c, b_f, c_f, gamma, T, init_t, n_iterations, solver, timestep)
    X = np.linspace(0,100,101)
    Y = pol.time
    X_grid, Y_grid = np.meshgrid(X,Y)
    X_grid, Y_grid = X_grid.T, Y_grid.T
    
    for i in range(len(alphas)): 
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        surf = ax.plot_surface(X_grid, Y_grid, alphas[i], cmap="coolwarm",
                               linewidth=0, antialiased=False)
        plt.show()
        fig.savefig(os.path.join(PATH_IMAGES, 'alpha_iteration'+str(i)+'.png'))

    for i in range(len(alphas)): 
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        surf = ax.plot_surface(X_grid, Y_grid, value_functions[i], cmap="coolwarm",
                               linewidth=0, antialiased=False)
        plt.show()
        fig.savefig(os.path.join(PATH_IMAGES, 'value_function_iteration'+str(i)+'png'))


    # we compare with explicit HJB solution
    x = np.linspace(0,100,101)
    hjb = HJB_LQR(b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, gamma=gamma, T=T, init_t = init_t, solve_ode=True, timestep=timestep)
    alpha = np.array([hjb.get_alpha(x_i) for x_i in x])
    value = np.array([hjb.get_value_function(x_i) for x_i in x])
    diff_alpha_HJB_iteration = norm(alphas[-1]-alpha, ord='fro')
    diff_value_HJB_iteration = norm(value_functions[-1]-value, ord='fro')
    
    # get difference between value function of the iterative method and explicit solution
    diff_value_HJB_iterative_method = [norm(val_iterative-value, ord='fro') for val_iterative in value_functions]







