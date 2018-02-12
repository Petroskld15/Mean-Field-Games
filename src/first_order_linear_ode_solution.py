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
from scipy.integrate import simps


# FIRST IMPLEMENTATION using integrating factor. ODE with terminal condition
timestep = 0.01
t0 = 8
T = 10
time = np.arange(t0,T+timestep,timestep)


a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t
y_T = 10


a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])

integrand_const = -a1
val_integral_const = simps(integrand_const, time)
sol = np.zeros_like(time)
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


# WITH INITIAL CONDITION AND NOT FINAL CONDITION
timestep = 0.005
t0 = 0
T = 1
time = np.arange(t0,T+timestep*0.1,timestep)

a0_lambda = lambda t: t
a1_lambda = lambda t: 2*t
y_0 = 0
a0 = np.array([a0_lambda(t) for t in time])
a1 = np.array([a1_lambda(t) for t in time])
sol = np.zeros_like(time)
for i in range(len(time)):
    integrand1 = -a1[:i+1]
    val_integral1 = simps(integrand1, time[:i+1])
    integrand2 = np.copy(a0[:i+1])
    for j in range(i):
        integrand3 = -np.copy(a1[:j+1])
        val_integral3 = simps(integrand3, time[:j+1])
        integrand2[j] = integrand2[j]*math.exp(val_integral3)
    val_integral2 = simps(integrand2, time[:i+1])
    sol[i] = math.exp(-val_integral1)*(y_0+val_integral2)
    
expl_sol = lambda t: 0.5*math.exp(t**2)-1/2
sol_grid = np.array([expl_sol(t) for t in time])

norm(sol-sol_grid)

