import os
import numpy as np
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor
from parametric_prior import pCNSampler
from pathlib import Path

dims = [64, 64]
dim = dims[0]
num_chains = 5
num_samples = 500000
scale = 5e-2
img_idx = 10

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/pcn_samples.mymemmap') # Second moment memmap
pcn_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='r', shape=(num_samples, int(num_chains*dim**2)))


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

test_slice = processor_cl.norm_loader(batch_idx=img_idx, batch_size=1, final_dims=dims)[0]
test_slice = test_slice
