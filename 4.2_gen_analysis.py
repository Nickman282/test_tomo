import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import load_params, MSE, sq_PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler


# Init parameters
dims = [256, 256]
dim = dims[0]

num_imgs = 10
num_samples = 200
img_idx = 5

'''
# Hybrid Gibbs Load-in
hg_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.mymemmap') 
hg_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.05.mymemmap') 
hg_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_100_0.mymemmap') 
hg_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_100_0.05.mymemmap') 

samples_HG_135_0 = np.memmap(filename = hg_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_135_005 = np.memmap(filename = hg_mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_100_0 = np.memmap(filename = hg_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_100_005 = np.memmap(filename = hg_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))


# RTO Load-In
rto_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_135_0.mymemmap') 
rto_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_135_0.05.mymemmap') 
rto_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.mymemmap') 
rto_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.05.mymemmap') 

samples_RTO_135_0 = np.memmap(filename = rto_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_135_005 = np.memmap(filename = rto_mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_100_0 = np.memmap(filename = rto_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_100_005 = np.memmap(filename = rto_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))


# NUTS Load-In
nuts_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.mymemmap') 
nuts_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.05.mymemmap') 
nuts_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_100_0.mymemmap')
nuts_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_100_0.05.mymemmap')

samples_NUTS_135_0 = np.memmap(filename = nuts_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
#samples_NUTS_135_005 = np.memmap(filename = nuts_mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
#samples_NUTS_100_0 = np.memmap(filename = nuts_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
#samples_NUTS_135_005 = np.memmap(filename = nuts_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
'''

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)
'''
i=2
ground_truth = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)[i]
samples = samples_NUTS_135_0[i*num_samples:(i+1)*num_samples]


'''

hg_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.05.mymemmap') 

samples_HG_135_005 = np.memmap(filename = hg_mem_file_2, dtype='float32', mode='r', shape=(num_samples*1, int(dim**2)))


fig, ax = plt.subplots(nrows=1, ncols=1)
im = ax.imshow(np.mean(samples_HG_135_005, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')
plt.show()


'''
fig, ax = plt.subplots(nrows=1, ncols=2)

mean = np.mean(samples, axis=0)

# Find difference
diff = abs(ground_truth - mean)
#diff = diff/(diff.max()-diff.min())

# Find variance
var = np.std(samples, axis=0)
#var = 1/(var.max() - var.min())*var

im = ax[0].imshow(diff.reshape(dims), cmap='Greys', aspect='auto')
ax[0].set_title("abs(ground_truth - recon)")
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(var.reshape(dims), cmap='Greys', aspect='auto')
ax[1].set_title("std")
plt.colorbar(im, ax=ax[1])

plt.show()
'''