import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce
import shutil

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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, it, is_best):
    filename = 'checkpoint_w_it' + str(it) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_long.pth.tar')


class Model_PDE(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=0.1):
        super(Model_PDE, self).__init__()
        # Layers of the network
        self.i2h1 = nn.Sequential(nn.Linear(2,100,bias=True), nn.Sigmoid())
        self.i2h2 = nn.Sequential(nn.Linear(2,100, bias=True), nn.Sigmoid())
        #self.h2h = nn.Sequential(nn.Linear(50,50,bias=True), nn.ReLU)
        self.h2o = nn.Linear(100,1,bias=True)
        
        
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
        
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
        return output, pde
    
    
class Model_PDE_2(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=0.1):
        super(Model_PDE_2, self).__init__()
        # Layers of the network
        self.i2h = nn.Sequential(nn.Linear(2,1000,bias=True), nn.Tanh())
        self.h2o = nn.Linear(1000,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        
    def forward(self, x):
        # Design of neural network
        h = self.i2h(x)
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
        
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
        return output, pde

class Model_PDE_3(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=0.1):
        super(Model_PDE_3, self).__init__()
        # Layers of the network
        self.i_h1 = nn.Sequential(nn.Linear(2,20,bias=True), nn.Tanh())
        self.h1_h2 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h2_h3 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h3_h4 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h4_h5 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h5_h6 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h6_h7 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h7_h8 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h8_h9 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Tanh())
        self.h9_o = nn.Linear(20,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        
    def forward(self, x):
        # Design of neural network
        h1 = self.i_h1(x)
        h2 = self.h1_h2(h1)
        h3 = self.h2_h3(h2)
        h4 = self.h3_h4(h3)
        h5 = self.h4_h5(h4)
        h6 = self.h5_h6(h5)
        h7 = self.h6_h7(h6)
        h8 = self.h7_h8(h7)
        h9 = self.h8_h9(h8)
        output = self.h9_o(h9)
        
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
        
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
        return output, pde
    

def sample(time_interval, xlim, batch_size, gamma):
    xmin, xmax = xlim
    start_time,end_time = time_interval
    
    t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
    x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    points = torch.cat([t,x],dim=1)
    
    terminal_points_x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    terminal_points_t = torch.ones_like(terminal_points_x)*end_time
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points




def train_fixed_iterations():
    batch_size = 10
    n_iter = 5000
    time_interval = (0,10)
    xlim = (8,10)
    gamma = 2
    model = Model_PDE()
    criterion = nn.MSELoss()
    base_lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for it in range(n_iter):
        # learning rate decay
        lr = base_lr * (0.5 ** (it // 100))
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
        loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)

        # we get the the gradients of the loss function by backpropagation
        loss.backward()
        # optimization step
        optimizer.step()
        
        # print statistics
        print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))        
        
        

def iterate(model, criterion, optimizer, losses, time_interval, xlim, batch_size, gamma):
    optimizer.zero_grad()
    data_batch, terminal_points = sample(time_interval, xlim, batch_size, gamma)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    
    # the target for the loss function is a vector of zeros, and terminal values        
    target = Variable(torch.zeros(batch_size))
    target_terminal = Variable(gamma*terminal_points.data[:,1]**2)
    
    # the output of the model is solution of pde, and the pde itself for the loss function
    output, PDE = model(data_batch)
    output_terminal, _ = model(terminal_points)
    loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    

    # we get the the gradients of the loss function by backpropagation
    loss.backward()
    # optimization step
    optimizer.step()
    
    losses.update(loss.data[0])
    
    

def train():
    batch_size = 10
    #n_iter = 5000
    time_interval = (0,1)
    xlim = (8,10)
    gamma = 2
    model = Model_PDE_3()
    criterion = nn.MSELoss()
    base_lr = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum=0.9)

    losses = AverageMeter()

    # first iteration
    iterate(model, criterion, optimizer, losses, time_interval, xlim, batch_size, gamma)
    n_iter = 1
    best_loss = losses.val
    
    while losses.val>1:
        n_iter+=1
        lr = base_lr * (0.5 ** (n_iter // 100))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        iterate(model, criterion, optimizer, losses, time_interval, xlim, batch_size, gamma)
        
        print("iteration: [{it}]\t loss: {loss:.3f}\t avg_loss: {avg_loss:.3f}".format(it=n_iter, loss=losses.val, avg_loss=losses.avg))  
            

def train_LBFGS():
    batch_size = 10
    time_interval = (0,1)
    xlim = (8,10)
    gamma = 2
    model = Model_PDE_3()
    criterion = nn.MSELoss()
    base_lr = 0.8
    n_iter = 10
    optimizer = torch.optim.LBFGS(model.parameters(),lr=base_lr, max_iter=20)    
    
    # we load all the data
    data_batch, terminal_points = sample(time_interval, xlim, batch_size, gamma)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    
    # the target for the loss function is a vector of zeros, and terminal values        
    target = Variable(torch.zeros(batch_size))
    target_terminal = Variable(gamma*terminal_points.data[:,1]**2)



    for it in range(n_iter):
        
        def closure():
            optimizer.zero_grad()
            # the output of the model is solution of pde, and the pde itself for the loss function
            output, PDE = model(data_batch)
            output_terminal, _ = model(terminal_points)
            loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    
            loss.backward()
            print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            return loss
        lr = base_lr * (0.5 ** (it // 5))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
            
        optimizer.step(closure)
    
    # testing
    data_batch, terminal_points = sample(time_interval, xlim, 1, gamma)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    output, PDE = model(data_batch)
    output_terminal, _ = model(terminal_points)
    
    # we compute alpha from value function: 
    l = []
    for row in range(output.size()[0]):
        input_grad = torch.autograd.grad(output[row], data_batch, create_graph=True)[0]
        l.append(input_grad)
    grad_f = reduce(lambda x,y: x+y, l)
    df_dx = grad_f[:,1]
    alpha = -1/(2*model.c_f)*model.c*df_dx
    
    
    
            
        
# debugging lines

m = nn.Linear(2, 1)
input = autograd.Variable(torch.randn(20, 2), requires_grad=True)
output = m(input)
output = nn.Sigmoid()(output)
print(output)
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







