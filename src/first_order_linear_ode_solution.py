"""
This script provides two methods to solve a First-order linear ODE
 - Explicit solution using an integrating factor
 - Euler method

ODE: y' = a0(t) + a1(t)y
 
We compare these two solutions with the example:
    y' = t+2ty, y(10) = 10
with solution:
    y(t) = 10.5 * exp(t^2-100) - 0.5
"""
import numpy as np
import math
from scipy.integrate import simps, trapz
import time as temps
from numpy.linalg import norm

# FIRST IMPLEMENTATION using integrating factor. ODE with terminal condition
timestep = 0.0001
t0 = 8
T = 10
time = np.arange(t0,T+timestep*0.1,timestep)


a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t
y_T = 10


a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])

integrand_const = -a1
val_integral_const = simps(integrand_const, time)
sol = np.zeros_like(time)
start = temps.time()
for i in range(len(time)):
    integrand2 = np.copy(a0[i:])
    for j in range(i, len(time)):
        integrand3 = -np.copy(a1[:j+1])
        val_integral3 = simps(integrand3, time[:j+1])
        integrand2[j-i] = integrand2[j-i]*math.exp(val_integral3)
    val_integral2 = simps(integrand2, time[i:])
    integrand4 = -a1[:i+1]
    try:
        val_integral4 = simps(integrand4, time[:i+1])
    except:
        val_integral4 = 0
    sol[i] = math.exp(-val_integral4)*(y_T*math.exp(val_integral_const) - val_integral2)
print('execution time: {:.3f}'.format(temps.time()-start))

      
# euler method of same equation
y0 = y_T
sol_euler = np.zeros_like(time)
sol_euler[-1] = y0
for t in range(len(time)-1, 0, -1):
    m = a0[t]+a1[t]*sol_euler[t]
    y1 = y0 - m*(time[t]-time[t-1])
    sol_euler[t-1] = y1
    y0 = y1


# we compare it with the explicit solution

expl_sol = lambda t: 10.5*math.exp(t**2-100)-1/2
sol_grid = np.array([expl_sol(t) for t in time])

norm(sol_grid-sol_euler)  
norm(sol_grid-sol)   # much better this solution than Euler method


#### Second implementation using integrating factor. ODE with terminal condition
# we try to make it faster by doing clever things with the integrals
# we calculate the integrals bit by bit and then do a cumsum.
# like this the algorithm will be much faster
timestep = 0.0001
t0 = 8
T = 10
time = np.arange(t0,T+timestep*0.1,timestep)

a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t

a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])
y_T = 10

integrand_const = -a1
val_integral_const = simps(integrand_const, time)
sol = np.zeros_like(time)
integrand2 = np.zeros_like(time)
val_integral2 = np.zeros_like(time)
val_integral3 = np.zeros_like(time)
val_integral4 = np.zeros_like(time)

# we fill integrand4
for i in range(1,len(time)):
    integrand4 = -np.copy(a1[i-1:i+1])
    val = simps(integrand4, time[i-1:i+1])
    val_integral4[i] = val
val_integral4 = np.cumsum(val_integral4)
e_integral4 = np.exp(-val_integral4)

# we fill val integral3
for i in range(1,len(time)):
    integrand3 = -np.copy(a1[i-1:i+1])
    val = simps(integrand3, time[i-1:i+1])
    val_integral3[i] = val
val_integral3 = np.cumsum(val_integral3)

# we get integrand2
integrand2 = a0 * np.exp(val_integral3)

# we get val_integral2
for i in range(len(time)-1,0,-1):
    val = simps(integrand2[i-1:i+1], time[i-1:i+1])
    val_integral2[i-1] = val

val_integral2 = np.flip(np.cumsum(np.flip(val_integral2,axis=0)),axis=0)

solution_ode = e_integral4*(y_T*math.exp(val_integral_const)-val_integral2)

    
    
# WITH INITIAL CONDITION AND NOT FINAL CONDITION
# very fast implementation
timestep = 0.0001
t0 = 0
T = 2
time = np.arange(t0,T+timestep*0.1,timestep)

a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t
y_0 = 0
a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])
sol = np.zeros_like(time)

val_integral1 = np.zeros_like(time)
val_integral2 = np.zeros_like(time)
val_integral3 = np.zeros_like(time)

# we fill val_integral1
for i in range(1, len(time)):
    integrand1 = -np.copy(a1[i-1:i+1])
    val_integral1[i] = trapz(integrand1, time[i-1:i+1])
val_integral1 = np.cumsum(val_integral1)
e_integral1 = np.exp(-val_integral1)

# we fill val_integral3
for i in range(1, len(time)):
    val_integral3[i] = trapz(-a1[i-1:i+1], time[i-1:i+1])
val_integral3 = np.cumsum(val_integral3)

# we get integrand2
integrand2 = a0 * np.exp(val_integral3)
for i in range(1, len(time)):
    val_integral2[i] = trapz(integrand2[i-1:i+1], time[i-1:i+1])
val_integral2 = np.cumsum(val_integral2) 

solution_ode = e_integral1*(y_0 + val_integral2)

   
timestep = 0.0001
t0 = 0
T = 2
time = np.arange(t0,T+timestep*0.1,timestep)

a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t

a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])
sol_grid = 0.5*np.exp(time**2)-0.5

norm(sol_grid-solution_ode)





