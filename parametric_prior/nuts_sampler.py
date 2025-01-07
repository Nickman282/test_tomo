import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor

from .generic_sampler import GenericSampler
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

class NutsSampler():

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None):

        return None

    def _post_distribution(self, data):
        
        #d = cq.distribution.Gamma(1, 1e-4)
        #s = cq.distribution.Gamma(1, 1e-4)
        x = cq.distribution.Gaussian(0, 1, geometry=self.A.domain_geometry)
        y = cq.distribution.Gaussian(self.A@x, 0.05**2)

        y_like = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(y_like, x)

        return posterior

    def run(self, test_img, N=1000, Nb=500):

        test_sino = self._projection(test_img)
        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        posterior = self._post_distribution(data=y_obs)

        # NUTS sampler
        sampler = cq.sampler.NUTS(posterior)

        #et = time.time() - st

        return sampler.sample(N, Nb).samples.T