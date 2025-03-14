import logging
import os

import timeit
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import FastRadonTransform, load_params, MSE
from pathlib import Path

from data_processor import Processor
from parametric_prior import NUTS_sampler_LogN, NUTS_sampler_Beta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

dims = [256, 256]
dim = dims[0]
deg = 135
num_imgs = 10
num_samples = 100
burn_in = 100
img_idx = 5
noise_power = 0

# Sample storage
#mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_LMRF_135_40.mymemmap') 
#store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) 

#-----------------------------------------------------------------------------------------------------------------------------

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)


deg = 135
deg_rad = (deg/180)*np.pi
noise_power = 40

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_40.mymemmap')
store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) 


for i in range(num_imgs):
    start = timeit.default_timer()
    print(f"Start time:{start}")

    test_slice = test_slices[i]
    test_slice = test_slice.reshape(dims)

    # Create Sampler model
    nuts_sampler = NUTS_sampler_LogN(dims, deg, device, likel_std=0.05)

    samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power=noise_power)

    samples = samples.detach().cpu().numpy()

    store_samples[i*num_samples:(i+1)*num_samples] = samples

    print(f"End time: {(timeit.default_timer() - start)/60}")

#-----------------------------------------------------------------------------------------------------------------------------


'''

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

# Create Sampler model
nuts_sampler = NUTS_sampler_LogN(dims, deg, device, likel_std=0.05)

samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power=noise_power)

samples = samples.detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(np.abs(test_slice-np.mean(samples, axis=0).reshape(dims)), cmap='Greys', aspect='auto')

var = np.std(samples, axis=0)
#var = 1/(var.max() - var.min())*var

im = ax[1].imshow(var.reshape(dims), cmap='Greys', aspect='auto')

plt.show()

'''








'''
nuts_sampler = NUTS_sampler_LogN(dims, deg, device, likel_std=0.05)

samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power=noise_power)

samples = samples.detach().cpu().numpy()

fig, ax = plt.subplots()

im = ax.imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')

plt.show()

fig, ax = plt.subplots()

im = ax.imshow(np.std(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')

plt.show()








test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
for i in range(num_imgs):

    start = timeit.default_timer()
    print(f"Start time:{start}")

    test_slice = test_slices[i]
    test_slice = test_slice.reshape(dims)

    nuts_sampler = NUTS_sampler_LogN(dims, deg, device)

    samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power = noise_power)

    samples = samples.detach().cpu().numpy()

    print(f"End time:{(timeit.default_timer() - start)/60}")

'''





'''

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs, final_dims=dims)
test_slice = test_slices[0]

nuts_sampler = NUTS_sampler_LMRF(dims, deg, device)
samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power = noise_power)
samples = samples.detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=3)
im = ax[0].imshow(test_slice.reshape(dims), cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax[1])

#var = np.var(samples, axis=0)
#var = np.log(1/(var.max() - var.min())*var)

im = ax[2].imshow(np.var(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax[2])


plt.show()

'''

'''
test_slice = processor_cl.norm_loader(batch_idx=0, batch_size=1, final_dims=dims)[0]

nuts_sampler = NUTS_sampler_LMRF(dims, deg, device)

samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in)

samples = samples.detach().cpu().numpy()

mse_func = lambda x : MSE(test_slice, x)

fig, ax = plt.subplots()

mse_val = np.array(list(map(mse_func, samples)))

ax.plot(np.arange(mse_val.shape[0]), mse_val)

#ax.legend(loc='upper right')

plt.show()

'''
