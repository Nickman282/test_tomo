import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import pymc as pm
import time
from data_processor import Processor

from .generic_sampler import GenericSampler
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData



    def _post_distribution(self, data, model=None):
        
        x = cq.distribution.Gaussian(0, 1, geometry=[1, 2048, 2, 2])

        #x = model.decode(z)

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