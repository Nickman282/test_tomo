import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor
import scipy.sparse as sps

from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData
from .generic_sampler import GenericSampler

class MHSampler(GenericSampler):

    def __init__(self, curr_dims, og_dims, num_angles, scale, 
                        max_angle = np.pi, pix_spacing=None):
        super().__init__(curr_dims, og_dims, num_angles, 
                        max_angle = max_angle, pix_spacing=pix_spacing)
        self.scale = scale
        return None
    def _post_distribution(self, data):
        
        x = cq.distribution.Gaussian(0, 1, geometry=self.A.domain_geometry)
        y = cq.distribution.Gaussian(self.A@x, 0.05**2)

        likelihood = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(likelihood, x)

        return posterior
    
    def run(self, test_img, init, N=500, Nb=500):

        test_sino = self._projection(test_img)
        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        posterior = self._post_distribution(data=y_obs)

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.MH(posterior, scale=self.scale, x0=init)


        return sampler.sample(N, Nb).samples.T
    
    