# Standard imports + DICOM
import numpy as np
import pandas as pd
import os
import pydicom 
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import os
import json

from tqdm import tqdm

'''
List of tags used:
1. (0x0020, 0x0032) Image Position (Patient) (x, y z coordinates for the upper-left corner)
2. (0x0020, 0x0037) Image Orientation (Patient) (Direction of cosines of the first row and the first column)
3. (0x0028, 0x0010) Number of Rows
4. (0x0028, 0x0011) Number of Columns
5. (0x0028, 0x0030) Pixel Spacing (x, y) - physical distance between pixel centres, [row spacing, column spacing]
6. (0x0028, 0x1052) Rescale Intercept - b in Output units = m*SV+b
7. (0x0028, 0x1053) Rescale Slope - m in Output units = m*SV+b
-1. (0x7FE0, 0x0010) Pixel Data (always loaded in at the end)
'''


TAG_LIST = [[0x0010, 0x0020],
            [0x0020, 0x0032], 
            [0x0020, 0x0037],  
            [0x0028, 0x0010], 
            [0x0028, 0x0011],
            [0x0028, 0x0030], 
            [0x0028, 0x1052],
            [0x0028, 0x1053],
            [0x0028, 0x3000],
            [0x7FE0, 0x0010]]

'''
Select z-coordinates of the slices
'''

SLICE_Z = -130

def filter_flies(directory = os.getcwd(), slice_z = SLICE_Z):
    '''
    Load full set of data to filter images that are unusable for the algorithm
    Find mean reconstruction diameter to rescale images

    Uses the following tags:

    (0x0018, 0x1100) Reconstruction Diameter
    (0x0018, 0x0050) Slice Thickness
    (0x0020, 0x0037) Image Orientation (Patient) (Direction of cosines of the first row and the first column)
    '''
    # Load all files present in the directory
    file_dirs = glob(f"{directory}/**/*.dcm", recursive=True)

    filtered_list = []
    recon_center = []
    for slice_filepath in tqdm(file_dirs):
        slice_file = pydicom.dcmread(slice_filepath)  
        
        if slice_file[0x0008,0x0060].value == "CT":

            print(slice_file[0x0008, 0x9007].value)
            
            slice_thick = slice_file[0x0018, 0x0050].value
            z_axis_val = slice_file[0x0020, 0x0032].value[2]

            if z_axis_val > (slice_z + slice_thick) or z_axis_val < (slice_z - slice_thick):
                continue

            orientation = slice_file[0x0020, 0x0037].value

            if orientation[0] == -1 or orientation[4] == -1:
                continue

            else:  
                filtered_list.append(slice_filepath)

            
                    
    median_recon_diam = np.median(recon_center)

    return {
        "filepaths": filtered_list,
        "median_diameter": median_recon_diam
    }  

def store_params(param_dict):
    with open(os.path.join(os.getcwd(), "settings.json"), "w") as f:
        return json.dump(param_dict, f)

param_dict = filter_flies(directory='D:/Studies/MEng_Project/LIDC-IDRI')
store_params(param_dict)