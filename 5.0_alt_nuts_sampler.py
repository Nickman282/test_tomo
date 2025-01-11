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
from parametric_prior import FastRadonTransform

dims = [64, 64]
dim = dims[0]
deg = 180
num_samples = 800
num_chains = 1

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/pcn_samples.mymemmap') # Second moment memmap
pcn_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='r', shape=(num_samples, int(num_chains*dim**2)))

def pyro_model(img):

    x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(dim**2), scale_tril=torch.eye(dim**2)))
    projector = FastRadonTransform([1, 1, dim, dim], torch.arange(deg))
    proj = projector.forward(x.view(1, 1, dim, dim))
    with pyro.plate("projections1", 100):
        return pyro.sample("y", dist.MultivariateNormal(proj.view(dim*deg), scale_tril=0.05*torch.eye(dim*deg)), obs=img)

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

test_slice = processor_cl.norm_loader(batch_idx=0, batch_size=1, final_dims=dims)[0]
test_slice = test_slice.reshape(dims)
torch_slice = torch.Tensor(test_slice).view(1, 1, dim, dim)

'''Project Test Slice'''
base_projector = FastRadonTransform([1, 1, dim, dim], torch.arange(deg))

torch_sino = base_projector(torch_slice).view(deg*dim)

nuts_kernel = NUTS(pyro_model, adapt_step_size=True)

mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=0)
mcmc.run(torch_sino)
samples = mcmc.get_samples()['x']

samples = samples.numpy()

samples[:num_samples, :] = samples[:, :]

'''
(hmc_kernel, num_samples=1, warmup_steps=10, num_chains=4096, 
                       initial_params=pyro.distributions.Normal(loc=0, scale=1)) #fails too
                       '''