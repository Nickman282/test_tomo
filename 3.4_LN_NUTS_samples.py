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
from parametric_prior import NUTS_sampler_LMRF, NUTS_sampler_Beta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

dims = [256, 256]
dim = dims[0]
deg = 100
num_imgs = 5
num_samples = 500
burn_in = 0
img_idx = 5
noise_power = 40

# Sample storage
mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/burnNUTS_100_0.05.mymemmap') # Second moment memmap
store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) # CVAR training data memmap

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

# Select a random test image
#rng = np.random.default_rng(seed=43)
#idx = rng.integers(0, processor_cl.len_filepaths, 1)[0]

start = timeit.default_timer()
print(f"Start time:{start}")
'''Burn-In estimation'''

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
for i in range(num_imgs):

    test_slice = test_slices[i]
    test_slice = test_slice.reshape(dims)

    nuts_sampler = NUTS_sampler_LMRF(dims, deg, device)

    samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in, noise_power = noise_power)

    samples = samples.detach().cpu().numpy()

    store_samples[i*num_samples:(i+1)*num_samples] = samples

print(f"End time:{timeit.default_timer() - start}")
'''
    fig, ax = plt.subplots(nrows=1, ncols=3)
    im = ax[0].imshow(test_slice.reshape(dims), cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
    #plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
    #plt.colorbar(im, ax=ax[1])

    var = np.var(samples, axis=0)
    var = np.log(1/(var.max() - var.min())*var)

    im = ax[2].imshow(np.var(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')
    #plt.colorbar(im, ax=ax[2])


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

#test_slice = processor_cl.norm_loader(batch_idx=0, batch_size=1, final_dims=dims)[0]

#nuts_sampler = NUTS_sampler_LMRF(dims, deg, device)

#samples = nuts_sampler.run(test_slice, num_samples, num_burnin=burn_in)

#samples = samples.detach().cpu().numpy()