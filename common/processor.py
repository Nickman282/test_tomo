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

# Create a Processor class, allowing for easy 
# extraction of DICOM data and conversion to 
# a pandas dataframe
class Processor:
    
    # Takes in a list of valid filepaths of DICOM files
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.len_filepaths = len(filepaths)
        return None

    def sample_loadin(self, idx=None):
        
        # Select random file if not specified
        if idx == None:
            idx = np.random.randint(0, self.len_filepaths)
            num = 1
        else:
            num = len(idx)
        
        if num == 1:
            idx = [idx]

        # Extract DICOM data into dataframe, row-by-row
        df_out = pd.DataFrame()
        for i in idx:
            
            path = self.filepaths[i]
                
            slice_file = pydicom.dcmread(path) # Load slice_file
            
            # Automatically rescale image to HU regadless of original modality
            rescaled_array = pydicom.pixels.apply_modality_lut(slice_file.pixel_array, slice_file)

            # Convert the image and metadata to
            df = pd.DataFrame(slice_file.values())
            df[0] = df[0].apply(lambda x: pydicom.dataelem.convert_raw_data_element(x) 
                                if isinstance(x, pydicom.dataelem.RawDataElement) else x)
            df['name'] = df[0].apply(lambda x: x.name)
            df['value'] = df[0].apply(lambda x: x.value)
            df = df[['name', 'value']]

            df = df.set_index('name').T.reset_index(drop=True) 

            # Replace pixel data with processed values and store rescaled ones
            df = df.drop('Pixel Data', axis=1)
            df["Pixel Data"] = [slice_file.pixel_array]
            df["Rescaled Pixel Data"] = [rescaled_array]
            
            # Concatenate into a singular dataframe
            df_out = pd.concat([
            df_out if not df_out.empty else None,
            df])

        return df_out

