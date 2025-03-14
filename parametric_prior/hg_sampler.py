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

    def run(self, test_img, N=1000, Nb=500, noise_power = 0, return_var=False):

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
        
        #st = time.time()
        posterior = self._post_distribution(data=y_obs)
        
        sampling_strategy = {
            'd': cq.sampler.ConjugateApprox,
            's': cq.sampler.Conjugate,
            'x': cq.sampler.UGLA}

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.Gibbs(posterior, sampling_strategy)

        #et = time.time() - st

        sampler_func = sampler.sample(N, Nb)

        if return_var == True:
            sampler_func['x'].samples.T, sampler_func['d'].samples.T, sampler_func['s'].samples.T

        return sampler_func['x'].samples.T
    


class SimpleUGLASampler(GenericSampler):

    def __init__(self, curr_dims, og_dims, num_angles, 
                        max_angle = np.pi, pix_spacing=None, likel_std=0.25, prior_std=0.1):
        super().__init__(curr_dims, og_dims, num_angles, 
                        max_angle = max_angle, pix_spacing=pix_spacing)
        
        self.prior_std = prior_std
        self.likel_std = likel_std
        
        return None

    def _post_distribution(self, data):
        
        x = cq.distribution.LMRF(0, self.prior_std, geometry=self.A.domain_geometry) #lambda d: 1/d
        y = cq.distribution.Gaussian(self.A@x, self.likel_std**2)

        likelihood = cq.likelihood.Likelihood(y, data)

        posterior = cq.distribution.Posterior(likelihood, x)

        return posterior

    def run(self, test_img, N=1000, Nb=500, noise_power = 0):

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
        
        #st = time.time()
        posterior = self._post_distribution(data=y_obs)
        

        # Gibbs sampler on p(d,s,x|y=y_obs)
        sampler = cq.sampler.UGLA(posterior)

        #et = time.time() - st

        return sampler.sample(N, Nb).samples.T
    

            
    


