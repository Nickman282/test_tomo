import os
import torch
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor

'''Dataset Loader'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch = processor_cl.norm_loader(batch_idx=2, batch_size=10)
batch = batch.reshape(batch.shape[0], 1, 256, 256)
batch = torch.from_numpy(batch).to(device)

'''Model Loader'''
model = torch.load(os.path.join(os.getcwd(), "cvar_prior/cvar.pt")).to(device)


prior = torch.distributions.MultivariateNormal(loc=torch.zeros(2*2*2048), scale_tril=torch.eye(2*2*2048))



#z_samples = prior.rsample(sample_shape=torch.Size([10])).to(device)

#z_samples = z_samples.reshape(10, 1024, 2, 2)

#x_samples = model.decode(z_samples).detach().cpu().numpy()

#x_samples = model.encode(batch)
#print(x_samples[0].shape)

x_samples = model(batch).detach().cpu().numpy()


fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    if i//5 == 0:
        im = ax[i//5, i%5].imshow(x_samples[i].reshape(256, 256), cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
        #plt.colorbar(im, ax=ax[i//5, i%5])
    else:
        im = ax[i//5, i%5].imshow(batch[i-5].detach().cpu().numpy().reshape(256, 256), cmap='Greys_r', aspect='auto', vmin = 0, vmax = 1)
        #plt.colorbar(im, ax=ax[i//5, i%5])        

plt.show()

