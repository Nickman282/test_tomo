import numpy as np

import cuqi as cq
import cuqipy_cil as cq_cil
import time
from data_processor import Processor
import scipy.sparse as sps

from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData
from .generic_sampler import GenericSampler

class RTOSampler(GenericSampler):

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None, prior_val=[0, 1], likel_std=0.05):
        super().__init__(curr_dims, og_dims, num_angles, 
                        max_angle = max_angle, pix_spacing=pix_spacing)
        
        self.prior_val = prior_val
        self.likel_std = likel_std

        return None
    def _post_distribution(self, data):
        
        x = cq.distribution.Gaussian(self.prior_val[0], self.prior_val[1]**2, geometry=self.A.domain_geometry) 
        y = cq.distribution.Gaussian(self.A@x, self.likel_std**2) #

        likelihood = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(likelihood, x)

        return posterior
    
    def run(self, test_img, N=1000, Nb=500, noise_power=0):

        test_sino = self._projection(test_img)

        if noise_power != 0:
            sq_factor = 10**(noise_power/10)
            noise_sq = np.mean(test_sino**2)/sq_factor
            noise = np.sqrt(noise_sq)
            print(np.mean(test_sino))
            print(noise)
        else:
            noise = 0


        test_sino += noise*np.random.randn(*test_sino.shape)

        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        posterior = self._post_distribution(data=y_obs)

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.LinearRTO(posterior)


        return sampler.sample(N, Nb).samples.T
    
class RegularizedRTOSampler(GenericSampler):

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None, prior_val=[0, 1], likel_std=0.05):
        super().__init__(curr_dims, og_dims, num_angles, 
                        max_angle = max_angle, pix_spacing=pix_spacing)
        
        self.prior_val = prior_val
        self.likel_std = likel_std

        return None
    def _post_distribution(self, data):
        
        x = cq.implicitprior.NonnegativeGaussian(self.prior_val[0], self.prior_val[1]**2, geometry=self.A.domain_geometry)
        y = cq.distribution.Gaussian(self.A@x, self.prior_val[1]**2) #

        likelihood = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(likelihood, x)

        return posterior
    
    def run(self, test_img, N=1000, Nb=500, noise_power=0):

        test_sino = self._projection(test_img)

        if noise_power != 0:
            sq_factor = 10**(noise_power/10)
            noise_sq = np.mean(test_sino**2)/sq_factor
            noise = np.sqrt(noise_sq)
        else:
            noise = 0


        test_sino += noise*np.random.randn(*test_sino.shape)

        y_obs = cq.array.CUQIarray(test_sino.flatten(order="C"), is_par=True, 
                geometry=cq.geometry.Image2D((self.num_angles, self.num_dets)))
        
        posterior = self._post_distribution(data=y_obs)

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.RegularizedLinearRTO(posterior)


        return sampler.sample(N, Nb).samples.T