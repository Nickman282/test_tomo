import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import load_params, MSE, sq_PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler


dims = [256, 256]
dim = dims[0]
deg = 135

num_imgs = 10
num_samples = 100

img_idx = 5

# Sample storage
mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.mymemmap') # Second moment memmap
store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2))) # CVAR training data memmap

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

ground_truths = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)


def plot_hist(map, idx, ground_truth):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    samples = map[idx*num_samples: (idx+1)*num_samples]

    #im = ax.imshow(np.mean(samples, axis=0).reshape(256, 256), cmap='Greys_r', vmin = 0)
    distances = np.array([np.linalg.norm(ground_truth - sample) for sample in samples])

    print(distances)

    ax.hist(distances)

    plt.show()

    return None

num_idx = 0

def mean_std_plot(map, idx):
    fig, ax = plt.subplots(nrows=1, ncols=2)

    samples = map[idx*num_samples: (idx+1)*num_samples]

    im = ax[0].imshow(np.mean(samples, axis=0).reshape(256, 256), cmap='Greys_r')
    im = ax[1].imshow(np.std(samples, axis=0).reshape(256, 256), cmap='Greys')

    ax[0].axis('off')
    ax[1].axis('off')

    plt.show()

mean_std_plot(store_samples, idx=0)
