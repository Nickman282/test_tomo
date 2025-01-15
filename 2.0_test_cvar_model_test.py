import os
import torch
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor
import numpy as np

'''Dataset Loader'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch = processor_cl.norm_loader(batch_idx=2, batch_size=8)
batch = batch.reshape(batch.shape[0], 1, 256, 256)
batch = torch.from_numpy(batch).to(device)

'''Model Loader'''
model = torch.load(os.path.join(os.getcwd(), "cvar_prior/cvar.pt")).to(device)


#prior = torch.distributions.MultivariateNormal(loc=torch.zeros(16*16*256), scale_tril=torch.eye(16*16*256))



#z_samples = prior.rsample(sample_shape=torch.Size([10])).to(device)



z_samples, mu, logvar = model.encode(batch)

#z_samples = z_samples.reshape(10, 1024, 2, 2)

x_samples = model.decode(z_samples).detach().cpu().numpy()
#print(x_samples[0].shape)

x_ground = np.squeeze(batch.detach().cpu().numpy())


fig, ax = plt.subplots(nrows=2, ncols=2)
im = ax[0, 1].imshow(x_samples[0].reshape(256, 256), cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
im = ax[1, 1].imshow(x_samples[1].reshape(256, 256), cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
im = ax[0, 0].imshow(x_ground[0], cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
im = ax[1, 0].imshow(x_ground[1], cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
        #plt.colorbar(im, ax=ax[i//5, i%5])        

plt.show()

