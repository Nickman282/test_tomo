import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import timeit

from common import load_params, PSNR
from data_processor import Processor
from parametric_prior import RTOSampler

dims = [256, 256]
dim = dims[0]
deg = 100
num_imgs = 10
num_samples = 200
burn_in = 10
img_idx = 5
noise_power = 0

deg_rad = (deg/180)*np.pi

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.mymemmap')
store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) 

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

# Select a random test image
#rng = np.random.default_rng(seed=43)
#idx = rng.integers(0, processor_cl.len_filepaths, 1)[0]

#test_slice = processor_cl.norm_loader(batch_idx=25, batch_size=1, final_dims=dims)[0]
#test_slice = test_slice.reshape(dims)

#j=1

start = timeit.default_timer()
print(f"Start time:{start}")
for i in range(num_imgs):
    test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)

    test_slice = test_slices[i] 

    test_slice = test_slice.reshape(dims)

    # Create Sampler model
    qc_sampler = RTOSampler(curr_dims=dims, og_dims=(512, 512), 
                            num_angles=deg, max_angle=deg_rad)

    samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in, noise_power=noise_power)

    store_samples[i*num_samples:(i+1)*num_samples] = samples

print(f"End time:{timeit.default_timer() - start}")
'''

test_slice = test_slices
test_slice = test_slice.reshape(dims)

# Create Sampler model
qc_sampler = RTOSampler(curr_dims=dims, og_dims=(512, 512), 
                        num_angles=deg, max_angle=deg_rad)

test_sino = qc_sampler._projection(test_slice)

test_sino = test_sino+np.mean(test_sino)*noise*np.random.randn(*test_sino.shape)

samples = qc_sampler.run(test_sino, N=num_samples, Nb=burn_in)

#store_samples[i*num_samples:(i+1)*num_samples] = samples

print(PSNR(samples[-1], test_slice.ravel()))

fig, ax = plt.subplots(nrows=1, ncols=3)

im = ax[0].imshow(test_slice, cmap='Greys_r', aspect='auto')
#plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')
#plt.colorbar(im, ax=ax[1])

var = np.var(samples, axis=0)
var = np.log(1/(var.max() - var.min())*var)

im = ax[2].imshow(var.reshape(dims), cmap='Greys', aspect='auto')
#plt.colorbar(im, ax=ax[2])

plt.show()
'''
