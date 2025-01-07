import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor

from .generic_sampler import GenericSampler
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

class HybridGibbsSampler(GenericSampler):

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None):
        super().__init__(curr_dims, og_dims, num_angles, 
                        max_angle = max_angle, pix_spacing=pix_spacing)
        return None

    def _post_distribution(self, data):
        
        d = cq.distribution.Gamma(1, 1e-4)
        s = cq.distribution.Gamma(1, 1e-4)
        x = cq.distribution.LMRF(0, lambda d: 1/d, geometry=self.A.domain_geometry)
        y = cq.distribution.Gaussian(self.A@x, lambda s: 1/s)

        posterior = cq.distribution.JointDistribution(y, x, s, d)(y = data)

        return posterior

    def run(self, test_img, N=1000, Nb=500):

        test_sino = self._projection(test_img)
        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        #st = time.time()
        posterior = self._post_distribution(data=y_obs)
        
        sampling_strategy = {
            'd': cq.sampler.ConjugateApprox,
            's': cq.sampler.Conjugate,
            'x': cq.sampler.UGLA}

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.Gibbs(posterior, sampling_strategy)

        #et = time.time() - st

        return sampler.sample(N, Nb)['x'].samples.T
    

            
    


