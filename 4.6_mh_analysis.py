import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from common import load_params
from data_processor import Processor
from parametric_prior import MHSampler


dims = [256, 256]
dim = dims[0]
deg = 180
num_chains = 4
num_imgs = 10
num_samples = 100000
iter_size = 5000
scale = [0.001, 0.01, 0.05, 0.1]
img_idx = 5

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/mh_samples.mymemmap') # Second moment memmap
mh_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='r', shape=(num_chains, num_samples)) # CVAR training data memmap


fig, ax = plt.subplots()

for i in range(num_chains):

    mse_val = mh_samples[i]

    ax.plot(np.arange(len(mse_val)), mse_val, label=f"$\sigma$ value: {scale[i]}")
    
ax.grid()
plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
ax.legend(loc='upper right')

plt.show()
plt.close()


mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/pcn_samples.mymemmap') # Second moment memmap
pcn_samples = np.memmap(filename = mem_file_2, dtype='float32', mode='r', shape=(5, num_samples)) # CVAR training data memmap


fig, ax = plt.subplots()

for i in range(4):

    mse_val = pcn_samples[i]

    ax.plot(np.arange(len(mse_val)), mse_val, label=f"Beta value: {scale[i]}")
    
ax.grid()
plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
ax.legend(loc='upper right')

plt.show()

