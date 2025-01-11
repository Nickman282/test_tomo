import os
import torch
import math
import matplotlib.pyplot as plt
from torch.optim import Adam

import numpy as np
from pathlib import Path
from common import load_params
from data_processor import Processor
from cvar_prior import DCVAE2

mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/cvar_training.mymemmap') 

dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch_size = 32
num_batches = math.ceil(processor_cl.len_filepaths / batch_size)

model = DCVAE2(device=device).to(device)
#model = torch.load(os.path.join(os.getcwd(), "cvar_prior/cvar.pt")).to(device)

optimizer = Adam(model.parameters(), lr=1e-5, amsgrad=True)

losses_list = []
epochs = 50
for epoch in range(epochs):
    for i in range(num_batches):

        # Load batch
        batch = processor_cl.norm_loader(batch_idx=i, batch_size=batch_size)
        batch = batch.reshape(batch.shape[0], 1, dim, dim)
        batch = torch.from_numpy(batch).to(device)

        loss, loss_recon, loss_kl = model.loss_fn(batch)
        losses_list.append([loss_recon.item(), loss_kl.item()])

        print(f"Epoch: {epoch+1}/{epochs}")
        print("Loss_KL:", loss_kl.item())
        print("Loss_Recon:", loss_recon.item())


        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        loss.backward()

        optimizer.step()


torch.save(model, os.path.join(os.getcwd(), "cvar_prior/cvar.pt"))

cvar_training = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_batches*epochs, 2)) # CVAR training data memmap
cvar_training[:] = np.stack(losses_list)[:]

