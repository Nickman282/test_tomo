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
import imageio

# Create a Processor class to extract and hold relevant information from the dataset/individual samples
class Processor:
    
    # Initially takes in main datapath, the standard chunk size for the data processing, and rescaling parameters
    def __init__(self, filepaths, median_diam):
        self.filepaths = filepaths
        self.median_diam = median_diam
        return None

    def sample_loadin(self, path):
        slice_file = pydicom.dcmread(path) 
        rescaled_array = pydicom.pixels.apply_modality_lut(slice_file.pixel_array, slice_file)

        df = pd.DataFrame(slice_file.values())
        
        df[0] = df[0].apply(lambda x: pydicom.dataelem.convert_raw_data_element(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
        df['name'] = df[0].apply(lambda x: x.name)
        df['value'] = df[0].apply(lambda x: x.value)
        df = df[['name', 'value']]

        df = df.set_index('name').T.reset_index(drop=True) 


        df = df.drop('Pixel Data', axis=1)
        df["Pixel Data"] = [slice_file.pixel_array]
        df["Rescaled Pixel Data"] = [rescaled_array]

        return df

    def rescaler(self, data_element):

        pixel_array = data_element['Rescaled Pixel Data']
        orientation = data_element['Image Orientation (Patient)']
        recon_diam = data_element['Reconstruction Diameter']

        # Flip relevant images
        if orientation[0] == -1 and orientation[4] == -1:
            pixel_array = np.rot90(pixel_array, 2)
            position = -1*np.array(position[:2])

        diam_ratio = self.median_diam/recon_diam

        if diam_ratio > 1:
            dims = round(512*diam_ratio)
            offset = dims - 512

            if offset % 2 != 0:
               ul_offset = np.ceil(offset/2)
               br_offset = np.floor(offset/2)
                                   
            temp_array = np.zeros((dims, dims))

            




                    

# 'Reconstruction Diameter' 'Rescale Intercept' 'Rescale Slope'
        



with open(os.path.join(os.getcwd(), "settings.json")) as f:
    param_dict = json.load(f)

filepaths = param_dict["filepaths"]

processor = Processor()
print(processor.sample_loadin(filepaths[0]).columns.values)


