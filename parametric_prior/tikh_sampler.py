import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor
import scipy.sparse as sps

from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

class TikhLikeSampler():

    def __init__(self):
        
        return None

    def projector_init(self, curr_dims, og_dims, num_angles, 
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

    def _post_distribution(self, data):
        
        x = cq.distribution.Gaussian(0, 1, geometry=self.A.domain_geometry)
        y = cq.distribution.Gaussian(self.A@x, 0.05**2)

        likelihood = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(likelihood, x)

        return posterior
    
    def run(self, test_img, N=1000, Nb=500):

        test_sino = self._projection(test_img)
        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        #st = time.time()
        posterior = self._post_distribution(data=y_obs)

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.LinearRTO(posterior)

        #et = time.time() - st

        return sampler.sample(N, Nb).samples.T
    