import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import load_params, MSE
from data_processor import Processor
from parametric_prior import HybridGibbsSampler

dims = [256, 256]
dim = dims[0]
deg = 100
num_imgs = 5
num_samples = 500
burn_in = 0
img_idx = 5

deg_rad = (deg/180)*np.pi

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/burnHG_100_0.05.mymemmap') 
mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/burnRTO_100_0.05.mymemmap') 
mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/burnNUTS_100_0.05.mymemmap') 
#mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.05.mymemmap')

samples_HG_90_20 = np.memmap(filename = mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_90_20 = np.memmap(filename = mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_NUTS_90_20 = np.memmap(filename = mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
#samples_RTO_no_noise = np.memmap(filename = mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

samples_HG = []
samples_RTO = []
samples_NUTS = []
#samples_RTO_no = []

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
for i in range(num_imgs):

    test_slice = test_slices[i]
    mse_func = lambda x : MSE(test_slice, x)

    sample_HG_90_20_mse = np.array(list(map(mse_func, samples_HG_90_20[i*num_samples:(i+1)*num_samples]))).ravel()
    sample_RTO_90_20_mse = np.array(list(map(mse_func, samples_RTO_90_20[i*num_samples:(i+1)*num_samples]))).ravel()
    
    if i <= 2:
        sample_NUTS_90_20_mse = np.array(list(map(mse_func, samples_NUTS_90_20[i*num_samples:(i+1)*num_samples]))).ravel()
        samples_NUTS.append(sample_NUTS_90_20_mse)
    #sample_RTO_no_mse = np.array(list(map(mse_func, samples_RTO_no_noise[i*num_samples:(i+1)*num_samples]))).ravel()

    samples_HG.append(sample_HG_90_20_mse)
    samples_RTO.append(sample_RTO_90_20_mse)
    
    #samples_RTO_no.append(sample_RTO_no_mse)


samples_HG = np.array(samples_HG)
samples_RTO = np.array(samples_RTO)
samples_NUTS = np.array(samples_NUTS)
#samples_RTO_no = np.array(samples_RTO_no)

samples_HG_mean, samples_HG_std = np.mean(samples_HG, axis=0), np.std(samples_HG, axis=0)
samples_RTO_mean, samples_RTO_std = np.mean(samples_RTO, axis=0), np.std(samples_RTO, axis=0)
samples_NUTS_mean, samples_NUTS_std = np.mean(samples_NUTS, axis=0), np.std(samples_NUTS, axis=0)
#samples_RTO_no_mean, samples_RTO_no_std = np.mean(samples_RTO_no, axis=0), np.std(samples_RTO_no, axis=0)

x = range(samples_HG_mean.shape[0])

fig, ax = plt.subplots(figsize=(10,4))

ax.fill_between(x, np.log10(samples_HG_mean+samples_HG_std), np.log10(samples_HG_mean-samples_HG_std), alpha=0.2, label='Hybrid-Gibbs 40dB')
ax.plot(x, np.log10(samples_HG_mean))

ax.fill_between(x, np.log10(samples_RTO_mean+samples_RTO_std), np.log10(samples_RTO_mean-samples_RTO_std), alpha=0.2, label='RTO 40dB')
ax.plot(x, np.log10(samples_RTO_mean))

ax.fill_between(x, np.log10(samples_NUTS_mean+samples_NUTS_std), np.log10(samples_NUTS_mean-samples_NUTS_std), alpha=0.2, label='NUTS 40dB')
ax.plot(x, np.log10(samples_NUTS_mean))

#ax.fill_between(x, np.log10(samples_RTO_no_mean+samples_RTO_no_std), np.log10(samples_RTO_no_mean-samples_RTO_no_std), alpha=0.2, label='RTO 0%')
#ax.plot(x, np.log10(samples_RTO_no_mean))

plt.xlabel("Number of Samples")
plt.ylabel("Log(Mean Squared Error)")
ax.legend(loc='upper right')
ax.grid()

plt.show()