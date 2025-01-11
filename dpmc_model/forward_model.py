from abc import ABC
import torch
import numpy as np
from cuqipy_cil.model import ParallelBeam2DModel

"""Adapted from CUQIpy-CIL plugin"""

class TorchProjector(ParallelBeam2DModel):
    def __init__(self, 
        im_size = (45,45),
        det_count = 50,
        angles = np.linspace(0,np.pi,60),
        det_spacing = None,
        domain = None
        ):

        super().__init__(im_size=im_size,
                         det_count=det_count,
                         angles=angles,
                         det_spacing=det_spacing,
                         domain=domain)
        


"""Adapted from Diffusion Posterior Sampler""" 

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm