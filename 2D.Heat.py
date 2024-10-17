import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import math


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import savgol_filter
from scipy.signal import medfilt


import os
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset
from torch.nn.functional import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.set_default_dtype(torch.float64)

class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        self.net = nn.Sequential(nn.Linear(3,64),
                                 nn.Tanh(),
                                 nn.Linear(64,64),
                                 nn.Tanh(),
                                 nn.Linear(64,1))
        
    def forward(self,x):
        return self.net(x)
    

def initial_condition(x,y):
    return torch.sin(torch.pi*x) * torch.sin(torch.pi*y)

def boundary_condition(x,y,t, custom_value):
    return torch.full_like(x, custom_value).to(device)
    
def generate_training_data(num_points):
    x = torch.rand(num_points, 1, requires_grad = True).to(device)
    y = torch.rand(num_points, 1, requires_grad = True).to(device)
    t = torch.rand(num_points, 1, requires_grad = True).to(device)
    
    return x, y, t

def generate_boundary_points(num_points):
    x_boundary = torch.tensor([0.0,1.0]).repeat(num_points//2).to(device)
    y_boundary = torch.rand(num_points).to(device)
    
    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary
        
    return x_boundary.view(-1,1), y_boundary.view(-1,1)

def generate_boundary_training_data(num_points):
    x_boundary, y_boundary = generate_boundary_points(num_points)
    t = torch.rand(num_points, 1, requires_grad=True).to(device)
    
    return x_boundary, y_boundary, t

def pde(x,y,t,model):
    input_data = torch.cat([x,y,t],dim=1)
    u = model(input_data)
    u_x,u_y = torch.autograd.grad(u,[x,y],grad_outputs= torch.ones_like(u), create_graph= True, retain_graph=True) 
    u_xx = torch.autograd.grad(u_x,x,grad_outputs= torch.ones_like(u_x), create_graph= True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y,y,grad_outputs= torch.ones_like(u_y), create_graph= True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u,t,grad_outputs= torch.ones_like(u), create_graph= True, retain_graph=True)[0]
    heat_eq_residual = 1 * u_xx + 1 * u_yy - u_t
    return heat_eq_residual

def train_PINN(model, num_iterations, num_points):
    optimiser = optim.Adam(model.parameters(), lr=1e-03)
    
    for iteration in range(num_iterations):
        optimiser.zero_grad()
        
        x,y,t = generate_training_data(num_points)
        
        x_b, y_b, t_b = generate_boundary_training_data(num_points)
        
        t_initial = torch.zeros_like(x)
        u_initial = initial_condition(x,y)
        
        custom_value = 0
        u_boundary_x = boundary_condition(x_b,y_b,t_b,custom_value)
        u_boundary_y = boundary_condition(y_b,x_b,t_b,custom_value)
        
        residual = pde(x,y,t,model)
        
        loss =  nn.MSELoss()(u_initial, model(torch.cat([x,y,t_initial], dim=1))) + \
                nn.MSELoss()(u_boundary_x, model(torch.cat([x_b,y_b,t_b], dim=1))) + \
                nn.MSELoss()(u_boundary_y, model(torch.cat([y_b,x_b,t_b], dim=1))) + \
                nn.MSELoss()(residual, torch.zeros_like(residual))
                
        loss.backward()
        optimiser.step()
        
        if iteration % 100 ==0:
            print(f"itration:, {iteration}, loss:, {loss}" )

model = PINN().to(device)
num_iterations = 10000
num_points = 1000
train_PINN(model,num_iterations,num_points)