import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


def hessian():
    """
    Very simple example on how to obtain second derivatives of f(x,t,theta)
        - df/dt
        - d^2f/dxdx
    We will need this to build the cost function for the model
    that approximates the solution of a parabolic PDE
    Ref: 
    """
    x = Variable(torch.randn(2), requires_grad=True)
    x = Variable(torch.Tensor([1,2]), requires_grad=True)
    theta = Variable(torch.Tensor([1]), requires_grad=True)
    y = x**2
    z = y.sum() + theta*x[0]*x[1]
    
    input_grad = torch.autograd.grad(z, x, create_graph=True)[0]
    hessian = [torch.autograd.grad(input_grad[i], x, create_graph=True)[0] for i in range(x.size()[0])]
    
    J = input_grad[0] - hessian[1][1]
    
    torch.autograd.grad(J, theta, create_graph=True)[0]
    
    return 1



class model(nn.Module):
    
    def __init__(self):
        #TODO
        self.i2h1 = nn.Sequential(nn.Linear(2,30,bias=True), nn.Sigmoid())
        self.i2h2 = nn.Sequential(nn.Linear(2,30, bias=True), nn.Tanh())
        self.h2o = nn.Linear(30,1,bias=True)
        
    def forward(self, x):
        #TODO
        h1 = self.i2h1(x)
        h2 = self.i2h2(x)
        h = h1 * h2
        output = self.h2o(h)
        return output
    

# PREPARING FOR TRAINING
        


        












