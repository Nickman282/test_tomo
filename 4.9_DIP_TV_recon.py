import os
import torch
import math
import matplotlib.pyplot as plt
from torch.optim import Adam

import numpy as np
from pathlib import Path
from common import load_params
from data_processor import Processor
from common import FastRadonTransform
from deep_image_prior import UNet

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

'''Load projector'''
max_angle = 135
num_angles = max_angle
radon = FastRadonTransform(image_size=[1, 1, 256, 256], theta=torch.arange(0, max_angle)).to(device)

'''Load Model'''
model = UNet(image_size = [1, 1, 256, 256], theta = torch.arange(0, max_angle), device=device).to(device)
optimizer = Adam(model.parameters(), lr=1e-4, amsgrad=True)


# Load batch
batch = processor_cl.norm_loader(batch_idx=0, batch_size=1)
batch = torch.from_numpy(batch.reshape(batch.shape[0], 1, dims[0], dims[1])).to(device)
# Convert to sinogram 
sino = radon.forward(batch).to(device)

out_plot = []
num_iter = 50001
for i in range(num_iter):
    x_recon, loss = model.loss_fn(sino)

    #print(f"Epoch: {i+1}/{num_iter}")
    print("Loss:", loss.item())

    if i % 10000 == 0:
        out_plot.append(torch.reshape(x_recon, (256, 256)).detach().cpu().numpy())
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

    loss.backward()

    optimizer.step()

torch.save(model, os.path.join(os.getcwd(), "deep_image_prior/dip.pt"))

fig, ax = plt.subplots(nrows=1, ncols=5)
for i in range(5):
    im = ax[i].imshow(out_plot[i], cmap='Greys_r', aspect='auto')
    #.set_title(f"{df['Image Type'].values[i]}")
    #plt.colorbar(im, ax=ax[i//5, i%5])
    #fig.suptitle('LIDC-IDRI images')
    ax[i].set_title(label=f"n={10000*i}")

plt.show()