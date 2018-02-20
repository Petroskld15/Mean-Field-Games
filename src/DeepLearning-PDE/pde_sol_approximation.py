import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce

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



class Model_PDE(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=1):
        super(Model_PDE, self).__init__()
        # Layers of the network
        self.i2h1 = nn.Sequential(nn.Linear(2,50,bias=True), nn.Sigmoid())
        self.i2h2 = nn.Sequential(nn.Linear(2,50, bias=True), nn.Tanh())
        self.h2o = nn.Linear(50,1,bias=True)
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        
    def forward(self, x):
        # Design of neural network
        h1 = self.i2h1(x)
        h2 = self.i2h2(x)
        h = h1 * h2
        output = self.h2o(h)
        
        # PDE to calculate the error
        # we get df/dx and df/dt where f(t,x) is the approimation of the
        
        # PDE solution given by our neural network
        l = []
        for row in range(output.size()[0]):
            input_grad = torch.autograd.grad(output[row], x, create_graph=True)[0]
            l.append(input_grad)
        grad_f = reduce(lambda x,y: x+y, l)
        df_dx = grad_f[:,1]
        df_dt = grad_f[:,0]
        # we get second derivatives
        grad_df_dx = []    
        for batch in range(df_dx.size()[0]):
            grad_df_dx.append(torch.autograd.grad(df_dx[batch], x, create_graph=True)[0])
        grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
        df_dxdx = grad_df_dx[:,1]
        df_dxdt = grad_df_dx[:,0]
        
#        grad_df_dt = []    
#        for batch in range(df_dt.size()[0]):
#            grad_df_dt.append(torch.autograd.grad(grad_df_dt[batch], input, create_graph=True)[0])
#        grad_df_dt = reduce(lambda x,y: x+y, grad_df_dt)        
#        df_dtdx = grad_df_dx[:,1]
#        df_dtdt = grad_df_dx[:,0]
        alpha = -0.5*x[:,1] + 0.1 # guess we had for alpha
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx + self.c*alpha*df_dx + self.c_f*alpha**2
        return output, pde
    

def sample(time_interval, xlim, batch_size, gamma):
    xmin, xmax = xlim
    start_time,end_time = time_interval
    
    t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
    x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    points = torch.cat([t,x],dim=1)
    
    terminal_points_x = xmin + torch.rand([int(batch_size*0.1), 1])*(xmax-xmin)
    terminal_points_t = torch.ones_like(terminal_points_x)*end_time
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points


model = Model_PDE()



def train():
    batch_size = 20
    n_iter = 1000
    batch_size = 20
    time_interval = (0,10)
    xlim = (8,10)
    gamma = 1
    model = Model_PDE()
    criterion = nn.MSELoss()
    base_lr = 0.0002
    optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for it in range(n_iter):
        # learning rate decay
        lr = base_lr * (0.5 ** (it // 50))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        # randomly sample t,x
        data_batch, terminal_points = sample(time_interval, xlim, batch_size, gamma)
        data_batch = Variable(data_batch, requires_grad=True)
        terminal_points = Variable(terminal_points, requires_grad=True)
        
        # the target for the loss function is a vector of zeros, and terminal values        
        target = Variable(torch.zeros(batch_size))
        target_terminal = Variable(gamma*terminal_points.data[:,1]**2)
        
        # the output of the model is solution of pde, and the pde itself for the loss function
        output, PDE = model(data_batch)
        output_terminal, _ = model(terminal_points)
        loss = criterion(PDE, target) #+ criterion(output_terminal, target_terminal)

        # we get the the gradients of the loss function by backpropagation
        loss.backward()
        # optimization step
        optimizer.step()
        
        # print statistics
        print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))        
        
        
        
        
        
# PREPARING FOR TRAINING
        


        



# code tests

m = nn.Linear(2, 30)
input = autograd.Variable(torch.randn(20, 2), requires_grad=True)
output = m(input)
m2 = nn.Linear(30,1)
output = m2(output)
output = nn.Sigmoid()(output)
print(output.size())

l = []
for row in range(output.size()[0]):
    input_grad = torch.autograd.grad(output[row], input, create_graph=True)[0]
    l.append(input_grad)
gradient = reduce(lambda x,y: x+y, l)

gradient_x = gradient[:,1]
gradient_t = gradient[:,0]

grad_gradient_x = []    
for batch in range(gradient_x.size()[0]):
    grad_gradient_x.append(torch.autograd.grad(gradient_x[batch], input, create_graph=True)[0])
grad_gradient_x = reduce(lambda x,y: x+y, grad_gradient_x)

grad_gradient_t = []    
for batch in range(gradient_t.size()[0]):
    grad_gradient_t.append(torch.autograd.grad(gradient_t[batch], input, create_graph=True)[0])
grad_gradient_t = reduce(lambda x,y: x+y, grad_gradient_t)


    
input_grad = torch.autograd.grad(output, input, create_graph=True)[0]
hessian = [torch.autograd.grad(input_grad[i], x, create_graph=True)[0] for i in range(x.size()[0])]







