import os
import torch
import math
import matplotlib.pyplot as plt
from torch.optim import Adam

import numpy as np
from pathlib import Path
from common import load_params
from data_processor import Processor
from common import LimitedAngleFBP
from unet import UNet

#mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/unet_training.mymemmap') 

'''Load Files'''
dims = [256, 256]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

'''Load Model'''
model = torch.load(os.path.join(os.getcwd(), "unet/unet_fbp_90.pt")).to(device)

optimizer = Adam(model.parameters(), lr=1e-4, amsgrad=True)

'''Load projector'''
max_angle = 90
num_angles = max_angle

projector = LimitedAngleFBP(curr_dims=dims, init_dims=(512, 512), 
                            num_angles=num_angles, max_angle=max_angle,
                            init_pix_space=pix_space)

# Load batch
batch = processor_cl.norm_loader(batch_idx=0, batch_size=1)

# Convert to sinogram 
batch_corrput = projector.transform_batch(batch)

batch = batch.reshape(batch.shape[0], 1, dims[0], dims[1])

batch_corrput = batch_corrput.reshape(batch.shape[0], 1, dims[0], dims[1])
batch_corrput = torch.from_numpy(batch_corrput).to(device)

batch_restore = model(batch_corrput)
batch_restore = batch_restore.cpu().detach().numpy()
batch_corrput = batch_corrput.cpu().detach().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(batch_restore.reshape(dims), cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(batch.reshape(dims), cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax[1])

plt.show()