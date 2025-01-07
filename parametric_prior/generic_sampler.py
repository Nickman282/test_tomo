import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor
import scipy.sparse as sps

from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData


class GenericSampler():

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None):
            
            self.num_angles = num_angles
            self.num_dets = curr_dims[0]
            
            if pix_spacing is None:
                voxel_dims = [1, 1]
            else:
                voxel_dims = [(og_dims[i]/curr_dims[i])*pix_spacing[i] for i in range(len(curr_dims))]   

            aq_angles = np.linspace(0, max_angle, num_angles)
            ag_domain = [voxel_dims[i]*curr_dims[i] for i in range(len(curr_dims))]

            self.A = cq_cil.model.ParallelBeam2DModel(im_size=curr_dims,
                                                    det_count=curr_dims[0],
                                                    angles=aq_angles,
                                                    det_spacing=voxel_dims[0],
                                                    domain=ag_domain)
            return None

    def _projection(self, img):
        
        ig = self.A.image_geometry

        img_container = ImageData(img, geometry=ig)
        img_phantom = self.A.ProjectionOperator.direct(img_container).as_array()    

        return img_phantom