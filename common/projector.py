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
    
    def __init__(self, num_angles, max_angle, is_limited=False, init_dims=(512, 512), 
                    curr_dims=(256, 256), init_pix_space=None):
    
        if init_pix_space is None:
            voxel_dims = [1, 1]
        else:
            voxel_dims = voxel_dims = [(init_dims[i]/curr_dims[i])*init_pix_space[i] for i in range(len(curr_dims))]   

        self.ig = ImageGeometry(voxel_num_x=curr_dims[0], 
                            voxel_num_y=curr_dims[1], 
                            voxel_size_x=voxel_dims[0], 
                            voxel_size_y=voxel_dims[1])
        
        self.img_dims = curr_dims
        
        self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, max_angle, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])                

        self.proj_op = ProjectionOperator(self.ig, self.ag)

        self.proj_dims = [num_angles, curr_dims[0]]

        return None
    
    def projection(self, img):
            
        img_container = ImageData(img, geometry=self.ig)
        img_phantom = self.proj_op.direct(img_container).as_array()    

        return img_phantom
    

class LimitedAngleFBP(BasicProjector):

    def __init__(self, num_angles, max_angle, is_limited=False, init_dims=(512, 512), 
                    curr_dims=(256, 256), init_pix_space=None):
        super().__init__(num_angles=num_angles, max_angle=max_angle, is_limited=is_limited, init_dims=init_dims, 
                    curr_dims=curr_dims, init_pix_space=init_pix_space)
        
        self.fbp = FBP(self.ig, self.ag, device='gpu')

    def transform_single(self, img, noise_power=0):
        img = img.reshape(self.img_dims)

        if noise_power != 0:
            sq_factor = 10**(noise_power/10)
            noise_sq = np.mean(img**2)/sq_factor
            noise = np.sqrt(noise_sq)
        else:
            noise = 0

        img_container = ImageData(img, geometry=self.ig)
        img_phantom = self.proj_op.direct(img_container)

        reconstruction = self.fbp(img_phantom).as_array()
        return reconstruction

    def transform_batch(self, batch):
        
        out_img = []
        for img in batch:
            img = img.reshape(self.img_dims)

            img_container = ImageData(img, geometry=self.ig)
            img_phantom = self.proj_op.direct(img_container)

            reconstruction = self.fbp(img_phantom)
            out_img.append(reconstruction.as_array())

        out_img = np.stack(out_img)

        return out_img

"""
Pytorch-friendly Radon Transform, adapted from
https://github.com/Cardio-AI/mfvi-dip-mia/
""" 
class FastRadonTransform(torch.nn.Module):
    """
    Calculates the radon transform of an image given specified
    projection angles. This is a generator that returns a closure.

    Parameters
    ----------
    image : Tensor
        Input image of shape (B, C, H, W).
    theta : Tensor, optional
        Projection angles (in degrees) of shape (T,). If `None`, the value is set to
        torch.arange(180).

    Returns
    -------
    radon_image : Tensor
        Radon transform (sinogram) of shape (B, C, T, W).
    """

    def __init__(self, image_size, theta=None):
        super().__init__()
        assert image_size[-2] == image_size[-1]

        if theta is None:
            theta = torch.deg2rad(torch.arange(180.))
        else:
            theta = torch.deg2rad(theta)

        ts = torch.sin(theta)
        tc = torch.cos(theta)
        z = torch.zeros_like(tc)

        trans = torch.stack([tc, -ts, z, ts, tc, z]).permute(1, 0).reshape(theta.size(0), 2, 3)
        grid = torch.nn.functional.affine_grid(trans,
                                               (theta.size(0), image_size[1], image_size[2], image_size[3]),
                                               align_corners=False)

        self.register_buffer("theta", theta)
        self.register_buffer("ts", ts)
        self.register_buffer("tc", tc)
        self.register_buffer("z", z)
        self.register_buffer("trans", trans)
        self.register_buffer("grid", grid)

    def forward(self, image):
        img_r = torch.nn.functional.grid_sample(
            image.expand(self.theta.size(0), -1, -1, -1),
            self.grid,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        radon_image = img_r.sum(2, keepdims=True).permute(2, 1, 0, 3)

        return radon_image