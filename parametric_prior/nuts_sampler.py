import logging
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS, MCMC
from common import FastRadonTransform, load_params
from pathlib import Path

from data_processor import Processor

def circulant(tensor, dim=0):
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

def gauss_smoothness_prior(size=(256, 256)):
    kernel = torch.tensor([1, -2, 1])
    filler = torch.zeros(int(size[0]-kernel.shape[0]))
    d_row = torch.cat((kernel, filler), 0)
    D_mat = circulant(d_row)
    D_mat[-1, -1] = 1
    D_mat[-1, 0:1] = 0
    # Find inverse approximation
    #print(D_mat[-1])
    mat = torch.kron(torch.eye(128), D_mat) + torch.kron(torch.eye(128), D_mat)

    inv_mat = torch.inverse(mat)

    return inv_mat, mat

class NUTS_sampler_Beta():

    def __init__(self, dims, deg, device, likel_std=0.05, beta_vals=[2, 5]):
        self.dims = dims
        self.dim = dims[0]
        self.deg = deg
        self.device = device
        self.likel_std = likel_std
        self.beta_vals = beta_vals

    def _projector(self):
        return FastRadonTransform([1, 1, self.dim, self.dim], torch.arange(self.deg))

    def _pyro_model(self, sino):
        log_normal = dist.Beta(self.beta_vals[0]*torch.ones(self.dim**2).to(self.device), self.beta_vals[1]*torch.ones(self.dim**2).to(self.device))
        x = pyro.sample("x", dist.Independent(log_normal, 1))
        projector = FastRadonTransform([1, 1, self.dim, self.dim], torch.arange(self.deg).to(self.device)).to(self.device)
        proj = projector.forward(x.view(1, 1, self.dim, self.dim))
        normal = dist.Normal(proj.view(self.dim*self.deg), (self.likel_std**2)*torch.ones(self.dim*self.deg).to(self.device))
        with pyro.plate("projections1", 1):
            return pyro.sample("y", dist.Independent(normal, 1), obs=sino)
    
    def run(self, img, num_samples, num_burnin=0, noise_power=0):
        torch_slice = torch.Tensor(img).view(1, 1, self.dim, self.dim).to(self.device)

        projector = self._projector().to(self.device)
        torch_sino = projector(torch_slice).view(self.deg*self.dim)

        if noise_power != 0:
            sq_factor = 10**(noise_power/10)
            noise_sq = torch.mean(torch_sino**2)/sq_factor
            noise = torch.sqrt(noise_sq)
            torch_sino += noise.to(self.device)*torch.randn(*torch_sino.shape).to(self.device)

        model_func = lambda img: self._pyro_model(img)

        nuts_kernel = NUTS(model_func, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_burnin)

        mcmc.run(torch_sino)
        samples = mcmc.get_samples()['x']
        return samples

class NUTS_sampler_LogN():

    def __init__(self, dims, deg, device, likel_std=0.05):
        self.dims = dims
        self.dim = dims[0]
        self.deg = deg
        self.device = device
        self.likel_std = likel_std

    def _projector(self):
        return FastRadonTransform([1, 1, self.dim, self.dim], torch.arange(self.deg))

    def _pyro_model(self, sino):

        #theta = pyro.sample("theta", dist.Gamma(1, 1e-4))
        
        #x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(self.dim**2).to(self.device), torch.eye(self.dim**2).to(self.device)))
        log_normal = dist.LogNormal(torch.zeros(self.dim**2).to(self.device), torch.ones(self.dim**2).to(self.device))
        x = pyro.sample("x", dist.Independent(log_normal, 1))
        #x = pyro.sample("x", dist.MultivariateNormal(loc=self.mean, scale_tril=self.cov))
        #x = torch.log(x)
        x = x.to(self.device) # 0.1*self.chol_mat.to(self.device)@

        x = x.view(self.dim, self.dim)
        projector = FastRadonTransform([1, 1, self.dim, self.dim], torch.arange(self.deg).to(self.device)).to(self.device)
        proj = projector.forward(x.view(1, 1, self.dim, self.dim))
        normal = dist.Normal(proj.view(self.dim*self.deg), self.likel_std**2*torch.ones(self.dim*self.deg).to(self.device))
        with pyro.plate("projections1", 1):
            return pyro.sample("y", dist.Independent(normal, 1), obs=sino)
    
    def run(self, img, num_samples, num_burnin=0, noise_power=0):

        torch_slice = torch.Tensor(img).view(1, 1, self.dim, self.dim).to(self.device)

        projector = self._projector().to(self.device)
        torch_sino = projector(torch_slice.to(self.device)).view(self.deg*self.dim) 

        if noise_power != 0:
            sq_factor = 10**(noise_power/10)
            noise_sq = torch.mean(torch_sino**2)/sq_factor
            noise = torch.sqrt(noise_sq)
            torch_sino += noise.to(self.device)*torch.randn(*torch_sino.shape).to(self.device)

        
        model_func = lambda img: self._pyro_model(img)

        nuts_kernel = NUTS(model_func, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_burnin)

        mcmc.run(torch_sino)
        samples = mcmc.get_samples()['x']
        return samples


'''

'''