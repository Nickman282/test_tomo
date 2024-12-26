import os
import torch
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor

param_dict = load_params(os.path.join(os.getcwd(), "params.json"))

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch = processor_cl.norm_loader(batch_idx=0, batch_size=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(os.path.join(os.getcwd(), "data_driven_prior/var_model.pt")).to(device)

prior = torch.distributions.MultivariateNormal(loc=torch.zeros(256), scale_tril=torch.eye(256))

z_samples = prior.rsample(sample_shape=torch.Size([10])).to(device)

x_samples = model.decode(z_samples).detach().cpu().numpy()


fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(x_samples[i].reshape(64, 64), cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax[i//5, i%5])

plt.show()
