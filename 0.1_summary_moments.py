import os
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from common import load_params
from data_processor import Processor

import numpy as np

mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='w+', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_1, dtype='float64', mode='w+', shape=(128**2,128**2))

dim = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

sm_path, fm_path = processor_cl.prior_updater(chunk_size=2500, 
                            fm_path=first_moment_path, 
                            sm_path=second_moment_path)

first_moment_path[:] = fm_path[:]
second_moment_path[:] = sm_path[:] 