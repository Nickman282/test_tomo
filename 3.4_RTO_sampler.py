import os
import numpy as np
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor
from parametric_prior import RTOSampler

dims = [256, 256]

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

test_slice = processor_cl.norm_loader(batch_idx=25, batch_size=1, final_dims=dims)[0]
test_slice = test_slice.reshape(dims)

qc_sampler = RTOSampler(curr_dims=dims, og_dims=(512, 512), 
                         num_angles=135, max_angle=3*np.pi/4)

samples = qc_sampler.run(test_slice, N=500, Nb=500)


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

