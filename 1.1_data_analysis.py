import os
import matplotlib.pyplot as plt

from common import load_params
from data_processor import Processor

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

print(len(filepaths))

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

i = 11

df = processor_cl.sample_loadin(idx=range(10*i, 10*i+10))
batch = processor_cl.rescaler(df, final_dims=(256, 256))

fig, ax = plt.subplots(nrows=2, ncols=2)
#plt.axis('off')
for i in range(4):
    im = ax[i//2, i%2].imshow(batch[i].reshape(256, 256), cmap='Greys_r', aspect='auto')
    #.set_title(f"{df['Image Type'].values[i]}")
    #plt.colorbar(im, ax=ax[i//5, i%5])
    #fig.suptitle('LIDC-IDRI images')
    #ax[i//2, i%2].axis("off")

plt.show()