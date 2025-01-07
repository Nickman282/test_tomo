import os
import torch
import math

from common import load_params
from data_processor import Processor
from var_prior import VAE, model_train

dim = 64

param_dict = load_params(os.path.join(os.getcwd(), "params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

data_loader = lambda batch_idx, batch_size : processor_cl.norm_loader(batch_idx, batch_size, final_dims=(dim, dim))



learning_rate = 1e-3
weight_decay = 1e-2

batch_size = 25
num_batches = math.ceil(processor_cl.len_filepaths / batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #  
model = VAE(input_dim = dim**2, hidden_dim = dim**2 // 2, latent_dim = dim**2 // 16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


num_epochs = 25
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    model_train(model=model, optimizer=optimizer, 
                data_loader=data_loader, 
                device=device, num_batches=num_batches,
                batch_size=batch_size)

torch.save(model, os.path.join(os.getcwd(), "data_driven_prior/var_model.pt"))




'''
fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(batch[i].reshape(256, 256), cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax[i//5, i%5])

plt.show()
'''
