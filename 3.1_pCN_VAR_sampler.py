import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from common import load_params
from data_processor import Processor
from cvar_prior import pCN

dims = [256, 256]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

'''Dataset Loader'''

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch = processor_cl.norm_loader(batch_idx=1, batch_size=1)
batch = batch.reshape(batch.shape[0], 1, 256, 256)
batch = torch.from_numpy(batch).to(device)

'''Model Loader'''
model = torch.load(os.path.join(os.getcwd(), "cvar_prior/cvar.pt")).to(device)


'''pCN Sampler'''

sampler = pCN(num_angles=135)

data = sampler.projection(batch.detach().cpu().numpy())
z0 = np.random.randn(16*16*256)

x_samples, z_samples, log_like, acc = sampler.sample_adapt(10000, 1000, model, data, z0, device)

test_img = np.squeeze(batch.detach().cpu().numpy())

fig, ax = plt.subplots(nrows=1, ncols=2)
im = ax[0].imshow(test_img, cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(np.mean(x_samples, axis=0), cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax[1])

plt.show()


