import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

from common import load_params, MSE, sq_PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler


# Init parameters
dims = [256, 256]
dim = dims[0]

num_imgs = 10
num_samples = 200
img_idx = 5


hg_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.mymemmap') 
samples_HG_135_0 = np.memmap(filename = hg_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))

rto_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_135_0.mymemmap') 
samples_RTO_135_0 = np.memmap(filename = rto_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))

nuts_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.mymemmap')
samples_NUTS_135_0 = np.memmap(filename = nuts_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)


hg_out = []
rto_out = []
nuts_out = []
#samples = samples_NUTS_135_0[i*num_samples:(i+1)*num_samples]
batch = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
for i in range(num_imgs):
    ground_truth = batch[i]

    hg_samples = samples_HG_135_0[i*num_samples:(i+1)*num_samples]
    rto_samples = samples_RTO_135_0[i*num_samples:(i+1)*num_samples]
    nuts_samples = samples_NUTS_135_0[i*num_samples:(i+1)*num_samples]

    hg_mean = np.mean(hg_samples, axis=0)
    rto_mean = np.mean(rto_samples, axis=0)
    nuts_mean = np.mean(nuts_samples, axis=0)

    hg_diff = abs(ground_truth - hg_mean)
    rto_diff = abs(ground_truth - rto_mean)
    nuts_diff = abs(ground_truth - nuts_mean)

    hg_diff = hg_diff/(hg_diff.max()-hg_diff.min())
    rto_diff = rto_diff/(rto_diff.max()-rto_diff.min())
    nuts_diff = nuts_diff/(nuts_diff.max()-nuts_diff.min())

    hg_var = np.std(hg_samples, axis=0)
    rto_var = np.std(rto_samples, axis=0)
    nuts_var = np.std(nuts_samples, axis=0)

    hg_var = 1/(hg_var.max() - hg_var.min())*hg_var
    rto_var = 1/(rto_var.max() - rto_var.min())*rto_var
    nuts_var = 1/(nuts_var.max() - nuts_var.min())*nuts_var

    hg_out.append(pearsonr(hg_var, hg_diff).statistic)
    rto_out.append(pearsonr(rto_var, rto_diff).statistic)
    nuts_out.append(pearsonr(nuts_var, nuts_diff).statistic)

print(np.mean(hg_out))
print(np.mean(rto_out))
print(np.mean(nuts_out))

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