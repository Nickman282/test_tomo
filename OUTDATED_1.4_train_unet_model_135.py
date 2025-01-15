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

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/unet_135_training.mymemmap') 

'''Load Files'''
dims = [256, 256]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch_size = 32
num_batches = math.ceil(processor_cl.len_filepaths / batch_size)

'''Load Model'''
#model = UNet().to(device)
model = torch.load(os.path.join(os.getcwd(), "unet/unet_fbp_135.pt")).to(device)

optimizer = Adam(model.parameters(), lr=1e-4, amsgrad=True)

'''Load projector'''
max_angle = 135
num_angles = max_angle

projector = LimitedAngleFBP(curr_dims=dims, init_dims=(512, 512), 
                            num_angles=num_angles, max_angle=max_angle,
                            init_pix_space=pix_space)

losses_list = []
epochs = 25
for epoch in range(epochs):
    for i in range(num_batches):

        # Load batch
        batch = processor_cl.norm_loader(batch_idx=i, batch_size=batch_size)

        # Convert to sinogram 
        batch_corrput = projector.transform_batch(batch)

        batch = batch.reshape(batch.shape[0], 1, dims[0], dims[1])
        batch = torch.from_numpy(batch).to(device)

        batch_corrput = batch_corrput.reshape(batch.shape[0], 1, dims[0], dims[1])
        batch_corrput = torch.from_numpy(batch_corrput).to(device)

        loss  = model.loss_fn(batch)
        losses_list.append([loss.item()])

        print(f"Epoch: {epoch+1}/{epochs}")
        print("Loss:", loss.item())


        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        loss.backward()

        optimizer.step()

torch.save(model, os.path.join(os.getcwd(), "unet/unet_fbp_135.pt"))

unet_135_training = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_batches*epochs, 1)) # CVAR training data memmap
unet_135_training[:] = np.stack(losses_list)[:]