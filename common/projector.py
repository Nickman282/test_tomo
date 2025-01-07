import astra
import numpy as np
import torch

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

class BasicProjector():
    
    def __init__(self, num_angles, is_limited=False, init_dims=(512, 512), 
                    curr_dims=(128, 128), init_pix_space=None):
    
        if init_pix_space is None:
            voxel_dims = 1.0
        else:
            voxel_dims = (init_dims[0]/curr_dims[0])*init_pix_space

        self.ig = ImageGeometry(voxel_num_x=curr_dims[0], 
                            voxel_num_y=curr_dims[1], 
                            voxel_size_x=voxel_dims[0], 
                            voxel_size_y=voxel_dims[1])
        
        self.img_dims = curr_dims
        
        if is_limited == True:
            max_angle = 2*num_angles
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, max_angle, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])               
        else:
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, 180, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])   

        self.proj_op = ProjectionOperator(self.ig, self.ag)

        self.proj_dims = [num_angles, curr_dims[0]]

        return None
    
    def projection(self, img):
            
        img_container = ImageData(img, geometry=self.ig)
        img_phantom = self.proj_op.direct(img_container).as_array()    

        return img_phantom