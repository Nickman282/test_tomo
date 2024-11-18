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
    im_means = [] 
    im_sds = []
    for slice_filepath in tqdm(file_dirs):
        slice_file = pydicom.dcmread(slice_filepath) # Load image with pydicom
        
        if slice_file[0x0008,0x0060].value == "CT": # Load CT images only

            # Filters each value through functions
            for arg in args:
                if arg(slice_file) == False:
                    continue
            
            mean, sd = centering(slice_file.pixel_array)
            im_means.append(mean)
            im_sds.append(sd)
            
            filtered_dir_list.append(slice_filepath)

    
    out_means = np.quantile(im_means, 0.9, axis=0)
    out_sds = np.quantile(im_sds, 0.9, axis=0)
        
    return  {
            "filepaths": filtered_dir_list,
            "center_mean": out_means,
            "center_sd": out_sds
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

# Centering function, takes in a pixel array
# In the range of 0 -> 1
def centering(pixel_array):
    shape = pixel_array.shape
    
    marginal_x = np.mean(pixel_array, axis=0)
    sum_x = np.sum(marginal_x)
    
    marginal_y = np.mean(pixel_array, axis=1)
    sum_y = np.sum(marginal_y)
    
    mean_x = np.dot(np.arange(0, shape[0]), marginal_x)/sum_x
    var_x = np.dot((np.arange(0, shape[0]) - mean_x)**2, marginal_x)/sum_x
    
    mean_y = np.dot(np.arange(0, shape[1]), marginal_y)/sum_y
    var_y = np.dot((np.arange(0, shape[1]) - mean_y)**2, marginal_y)/sum_y
    
    mean = np.array([mean_x, mean_y])
    sd = np.sqrt([var_x, var_y])
    
    return mean, sd
    

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


# Takes in HU image pixels of len 1 or more
def rescaler(pixel_data, ref_mean, ref_sd, final_dims=(128, 128)):

    img_shape = pixel_data.shape

    if len(img_shape) <= 2:
        pixel_data = [pixel_data]
        num_img = 1
    else:
        num_img = img_shape[0]
    
    out_img = []
    
    for img in pixel_data:
        
        # Rescale image to 0 -> 1
        img = 1/(img.min() - img.max())*img-img.min()
        
        mean, sd = centering(img)
        
        # Image rescaling
        sd_ratio_x = sd[0]/ref_sd[0]
        sd_ratio_y = sd[1]/ref_sd[1]
        
        # Calculates half-window lengths including the relevant image features
        hwindow_x = np.floor(0.5*pixel_data.shape[0]*sd_ratio_x)
        hwindow_y = np.floor(0.5*pixel_data.shape[1]*sd_ratio_y)
        
        if hwindow_x % 2 != 0:
            hwindow_x -= 1
            
        if hwindow_y % 2 != 0:
            hwindow_y -= 1
        
        if sd_ratio_x < 0.9 and sd_ratio_y < 0.9:
            temp = img[mean[0] - hwindow_x: mean[0] + hwindow_x, mean[1] - hwindow_y: mean[1] + hwindow_y]
            
        elif sd_ratio_x < 0.9:
            temp = img[mean[0] - hwindow_x: mean[0] + hwindow_x, :]
            
        elif sd_ratio_y < 0.9:
            temp = img[:, mean[1] - hwindow_y: mean[1] + hwindow_y]
            
        else:
            temp = img
        
        # Rescale image to "final_dims" pixels 
        im_array = Image.fromarray(temp)
        im_resized = im_array.resize(final_dims)
        img = np.array(im_resized)
        
        centre = (final_dims[0]-1)/2
        
        # Set all out of range pixels to 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                arg_dist = np.floor(np.sqrt((centre - i)**2 + (centre - j)**2))
                if arg_dist > centre:
                    img[i, j] = 0
                    
        out_img.append(img)
        
    return out_img