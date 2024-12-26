import os
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor

param_dict = load_params(os.path.join(os.getcwd(), "params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

batch = processor_cl.norm_loader(batch_idx=1, batch_size=10)

print(len(filepaths))

fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(batch[i].reshape(256, 256), cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax[i//5, i%5])

plt.show()