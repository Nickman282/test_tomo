import os
import numpy as np
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor
from parametric_prior import MHSampler


dims = [64, 64]
dim = dims[0]
num_chains = 5
num_samples = 100000
scale = 5e-2
img_idx = 10

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

# Select a random test image
rng = np.random.default_rng(seed=43)
idx = rng.integers(0, processor_cl.len_filepaths, 1)[0]

test_slice = processor_cl.norm_loader(batch_idx=idx, batch_size=1, final_dims=dims)[0]
test_slice = test_slice.reshape(dims)

# Create Sampler model
qc_sampler = MHSampler(curr_dims=dims, og_dims=(512, 512), 
                          num_angles=180, max_angle=np.pi,
                          pix_spacing=pix_space, scale=5e-2)

samples = qc_sampler.run(test_slice, N=500000)


fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(test_slice, cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[1])

plt.show()