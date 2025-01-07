from common import *

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid




params_path = os.path.join(os.getcwd(), "params.json")
param_dict = load_params(params_path)

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

df = processor_cl.sample_loadin(idx=range(50, 60))
norm_slices = processor_cl.rescaler(df)

fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(norm_slices[i], cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax[i//5, i%5])
plt.show()



'''
        if not isinstance(z, torch.Tensor):
            try:
                torch.from_numpy(z)
            except:
                raise ValueError("Invalid decoder input")
'''


'''
mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap
mem_file_4 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cholesky.mymemmap') # Cholesky memmap
mem_file_5 = Path('D:/Studies/MEng_Project/LIDC-IDRI/inv_cholesky.mymemmap') # Inv Cholesky memmap
mem_file_6 = Path('D:/Studies/MEng_Project/LIDC-IDRI/proposal_distr.mymemmap') # Inv Cholesky memmap

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='r', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_1, dtype='float64', mode='r', shape=(128**2,128**2))
covariance_path = np.memmap(filename = mem_file_3, dtype='float64', mode='w+', shape=(128**2,128**2))
cholesky_path = np.memmap(filename = mem_file_4, dtype='float64', mode='w+', shape=(128**2,128**2))
inv_cholesky_path = np.memmap(filename = mem_file_5, dtype='float64', mode='w+', shape=(128**2,128**2))
proposal_distr = np.memmap(filename = mem_file_6, dtype='float64', mode='w+', shape=(128**2,128**2))

# Covariance Calculation
step = 128

print("Covariance Calculation:")
for sub_i in tqdm(range(step)):
    sec_st_i = step*sub_i
    sec_en_i = step*sub_i + step
    for sub_j in range(step):
        sec_st_j = step*sub_j
        sec_en_j = step*sub_j + step

        mul_exp = first_moment_path.reshape(-1, 1)[sec_st_i : sec_en_i, :] @ first_moment_path.reshape(1, -1)[:, sec_st_j : sec_en_j]
        covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = second_moment_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] - mul_exp

        temp = 1*np.diag(np.diag(covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j]))
        proposal_distr[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = temp 

# Add prior regularization to ensure positive definiteness
print("Adding prior offset to covariance")

covariance_path[:] = covariance_path[:] + 1e-3*np.diag(np.ones(128**2))
proposal_distr[:] = proposal_distr[:] + 1e-3*np.diag(np.ones(128**2))

# Cholesky decomposition
print("Cholesky Decomposition")
cholesky_val = torch.linalg.cholesky(torch.Tensor(covariance_path))

cholesky_path[:] = (cholesky_val.numpy())[:]
inv_cholesky_path[:] = (torch.linalg.inv(cholesky_val).numpy())[:]

proposal_distr[:] = torch.linalg.cholesky(torch.Tensor(proposal_distr)).numpy()[:]
'''



'''
def init_sampler(fm_path, chol_path, dims=(128, 128)):

    z = np.random.normal(0, 1, size=(dims[0]*dims[1]))

    sample = fm_path.T + chol_path @ z.T
    sample = sample.T

    return sample

'''