import os
import torch
import math
import matplotlib.pyplot as plt
from torch.optim import Adam
import timeit

import numpy as np
from pathlib import Path
from common import load_params, MSE
from data_processor import Processor
from common import FastRadonTransform
from deep_image_prior import UNet

dims = [256, 256]
dim = dims[0]
deg = 135
num_imgs = 10
img_idx = 5
noise_power = 0

'''Data Store'''
#mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_135_0.mymemmap') # Second moment memmap
#store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_imgs, int(dim**2))) # CVAR training data memmap


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

radon = FastRadonTransform(image_size=[1, 1, 256, 256], theta=torch.arange(0, deg)).to(device)

'''Load Model'''
model = UNet(image_size = [1, 1, 256, 256], theta = torch.arange(0, deg), device=device).to(device)
optimizer = Adam(model.parameters(), lr=1e-4, amsgrad=True)

start = timeit.default_timer()
print(f"Start time:{start}")

'''
# Load batch
batch = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
for i in range(num_imgs):
    img = torch.from_numpy(batch[i].reshape(1, 1, dims[0], dims[1])).to(device)
    # Convert to sinogram 
    sino = radon.forward(img).to(device)

    if noise_power != 0:
        sq_factor = 10**(noise_power/10)
        noise_sq = torch.mean(sino**2)/sq_factor
        noise = torch.sqrt(noise_sq)
        print(torch.mean(sino))
        print(noise)
    else:
        noise=0

    # Add noise to sino
    sino += noise*torch.randn(*sino.shape).to(device)

    out_plot = []
    num_iter = 10001
    for j in range(num_iter):
        x_recon, loss = model.loss_fn(sino)

        #print(f"Epoch: {i+1}/{num_iter}")
        print("Loss:", loss.item())

        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        loss.backward()

        optimizer.step()

        #if j == 0 or j == 1000 or j == 10000:
        #    out_plot.append(torch.reshape(x_recon, (256, 256)).detach().cpu().numpy())

    #store_samples[i] = x_recon.detach().cpu().numpy().ravel()
    print(x_recon.detach().cpu().numpy().ravel())

print(f"End time:{timeit.default_timer() - start}")
'''

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)

test_slice = test_slices[5]
test_slice_torch = test_slice.reshape(dims)

img = torch.from_numpy(test_slice_torch.reshape(1, 1, dims[0], dims[1])).to(device)

# Convert to sinogram 
sino = radon.forward(img).to(device)

num_iter = 50001

mem_file = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_rate.mymemmap') 
store_mse = np.memmap(filename = mem_file, dtype='float32', mode='w+', shape=(num_iter)) 


out = []
out_plot = []
for j in range(num_iter):
    x_recon, loss = model.loss_fn(sino)

    #print(f"Epoch: {i+1}/{num_iter}")
    print("Loss:", loss.item())

    
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

    loss.backward()

    optimizer.step()

    out.append(loss.item())

    if j == 10000 or j == 20000 or j == 30000:
        out_plot.append(torch.reshape(x_recon, (256, 256)).detach().cpu().numpy())

store_mse[:] = np.array(out)[:]

#torch.save(model, os.path.join(os.getcwd(), "deep_image_prior/dip.pt"))
'''
fig, ax = plt.subplots(nrows=1, ncols=3)
for i in range(3):
    im = ax[i].imshow(out_plot[i], cmap='Greys_r', aspect='auto')
    #.set_title(f"{df['Image Type'].values[i]}")
    #plt.colorbar(im, ax=ax[i//5, i%5])
    #fig.suptitle('LIDC-IDRI images')
    ax[i].axis('off')
    if i == 0:
        ax[i].set_title(label=f"n={10000}")

    elif i == 1:
        ax[i].set_title(label=f"n={20000}")
    
    elif i == 2:
        ax[i].set_title(label=f"n={30000}")

plt.show()
'''