import os
import numpy as np
import matplotlib.pyplot as plt

from common import load_params, MSE
from data_processor import Processor
from parametric_prior import pCNSampler
from pathlib import Path

dims = [256, 256]
dim = dims[0]
deg = 180
num_chains = 4
num_imgs = 10
num_samples = 100000
iter_size = 5000
scale = [0.001, 0.01, 0.05, 0.1]
img_idx = 5


# Sample storage
mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/pcn_samples.mymemmap') # Second moment memmap
pcn_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_chains, num_samples)) # CVAR training data memmap

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


test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)

test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

# Run multiple chains
for i in range(num_chains):
    # Run for 100000 iterations
    init_val = np.zeros(dim**2)
    for j in range(round(num_samples/iter_size)):
        # Create Sampler model with a given scale
        qc_sampler = pCNSampler(curr_dims=dims, og_dims=(512, 512), 
                            num_angles=180, max_angle=np.pi,
                            pix_spacing=pix_space, scale=scale[i])
        
        
        samples = qc_sampler.run(test_slice, N=iter_size, Nb=0, init=init_val)
        
        distances = [np.sqrt(MSE(np.ravel(test_slice, order='C'), sample)) for sample in samples]

        pcn_samples[i, j*iter_size:(j+1)*iter_size] = np.array(distances)

        init_val = samples[-1]

        print(np.mean(distances))


'''
fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(test_slice, cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[1])

plt.show()
'''