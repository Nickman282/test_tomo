# Standard imports + DICOM
import numpy as np
import pandas as pd
import os
import math
import pydicom 
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from PIL import Image
import json

# CIL framework
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.framework import BlockDataContainer

from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator, \
                                       IdentityOperator, \
                                       GradientOperator, FiniteDifferenceOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, \
                                       L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, \
                                       TotalVariation, \
                                       ZeroFunction

# CIL Processors
from cil.processors import CentreOfRotationCorrector, Slicer

from cil.utilities.display import show2D

# Plugins
from cil.plugins.astra.processors import FBP, AstraBackProjector3D
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins import TomoPhantom
from tqdm import tqdm

'''
Function loads all valid .dcm files in a directory, 
returns filepaths of CT images only.

Additionally, allows to pass functions that will constitute
additional filtering conditionals.

Functions must be of the form:  

  func(pydicom_file) -> bool
  
Must return False for the file to be excluded
'''

def filter_flies(directory = os.getcwd(), *args):
    # Load all files present in the directory
    file_dirs = glob(f"{directory}/**/*.dcm", recursive=True)

    filtered_dir_list = []
    for slice_filepath in tqdm(file_dirs):
        slice_file = pydicom.dcmread(slice_filepath) # Load image with pydicom
        
        if slice_file[0x0008,0x0060].value == "CT": # Load CT images only

            # Filters each value through functions
            for arg in args:
                if arg(slice_file) == False:
                    continue

            filtered_dir_list.append(slice_filepath)

        
    return  {
            "filepaths": filtered_dir_list,
            }  
        
    
# Filter all files with incorrect orientation
def select_orientation(slice_file):
    
    orientation = slice_file[0x0020, 0x0037].value
    
    if orientation[0] == -1 or orientation[4] == -1:
        return False
    
    else:
        return True
    
# Filter all files by z-axis coordinate
def select_z(slice_file, z_coord, error_bound):
    z_axis_val = slice_file[0x0020, 0x0032].value[2]
    
    if z_axis_val < (z_coord + error_bound):
        if z_axis_val > (z_coord - error_bound):
            return True
        else:
            return False 
    else:
        return False

# Stores the filtered list of files generated by "filter_files"
# into a "settings.json" file
def store_params(param_dict, filepath=os.path.join(os.getcwd(), "settings.json")):
    with open(filepath, "w") as f:
        return json.dump(param_dict, f)
    
# Load the "settings.json" file as a dict
def load_params(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
# Splits the patient into train and test sets
def train_test_split(filepaths, prop_train=0.8, random_seed=None):
    
    rng = np.random.default_rng(seed=random_seed)
    rng.shuffle(filepaths)

    prop_ind = round(len(filepaths)*prop_train)
    train_list = filepaths[:prop_ind]
    test_list = filepaths[prop_ind:]

    return train_list, test_list

# Takes in pandas dataframe of len 1 or more
def rescaler(self, pixel_data):

    #pixel_array = data_element['Rescaled Pixel Data']

    if diam_ratio > 1:
        dims = round(512*diam_ratio)
        offset = dims - 512

        if offset % 2 != 0:
            ul_offset = np.ceil(offset/2)
            br_offset = np.floor(offset/2)
                                
        temp_array = np.zeros((dims, dims))