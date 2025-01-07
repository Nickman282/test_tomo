import os
import torch
import math
import matplotlib.pyplot as plt
from torch.optim import Adam

from common import load_params
from data_processor import Processor
from var_prior import VAE, model_train
from dpmc_model import UNetModel, GaussianDiffusion, sigmoid_beta_schedule

dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch_size = 32
num_batches = math.ceil(processor_cl.len_filepaths / batch_size)

model = UNetModel()
model.to(device=device)

num_timesteps = 1000
betas = sigmoid_beta_schedule(num_timesteps=num_timesteps)
diffusion = GaussianDiffusion(model=model,
                              betas=betas
                              )

optimizer = Adam(model.parameters(), lr=1e-6)


epochs = 10

for epoch in range(epochs):
    for i in range(num_batches):

        # Load batch
        batch = processor_cl.norm_loader(batch_idx=i, batch_size=batch_size)
        batch = batch.reshape(batch.shape[0], 1, dim, dim)
        batch = torch.from_numpy(batch).to(device)

        t = torch.randint(0, num_timesteps, (batch.shape[0],), device=device)

        loss = diffusion.p_losses(x_start=batch, t=t)

        print(f"Epoch: {epoch+1}/{epochs}")
        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()



torch.save(model, os.path.join(os.getcwd(), "dpmc_model/diffusion.pt"))









print(loss)

'''
batch_process = model(torch.from_numpy(batch).to(device), timesteps=torch.arange(0, 10).to(device))
batch_process = batch_process.detach().cpu().numpy()


fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(batch_process[i, 0], cmap='Greys_r', aspect='auto')

plt.show()
'''