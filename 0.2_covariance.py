import os
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from common import load_params
from data_processor import Processor
from tqdm import tqdm

import numpy as np


mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap

first_moment_path = np.memmap(filename = mem_file_3, dtype='float64', mode='r', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='r', shape=(128**2,128**2))
covariance_path = np.memmap(filename = mem_file_1, dtype='float64', mode='w+', shape=(128**2,128**2))

dim = 128

fig, ax = plt.subplots(nrows=1, ncols=1)
im = ax.imshow(first_moment_path.reshape(dim, dim), cmap='Greys_r', aspect='auto')

plt.show()

print("Covariance Calculation:")

mul_exp = first_moment_path.reshape(-1, 1)@ first_moment_path.reshape(1, -1)
covariance_path = second_moment_path - mul_exp 

print(covariance_path.max())

# Add prior regularization to ensure positive definiteness

print("Adding prior offset to covariance")
covariance_path[:] = (covariance_path[:] + 9*np.diag(np.ones(128**2)))/10

# Cholesky decomposition
print("Cholesky Decomposition")
cholesky_val = torch.linalg.cholesky(torch.Tensor(covariance_path))

covariance_path[:] = (cholesky_val.numpy())[:]