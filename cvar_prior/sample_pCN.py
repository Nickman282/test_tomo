import torch
import numpy as np
import cuqi

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

class pCN():

    def __init__(self, num_angles, max_angle = 180, init_dims=(512, 512), 
                    curr_dims=(256, 256), init_pix_space=None,
                    prior_dims=[1, 2048, 2, 2]):
        '''Define Projector'''

        if init_pix_space is None:
            voxel_dims = [1, 1]
        else:
            voxel_dims = voxel_dims = [(init_dims[i]/curr_dims[i])*init_pix_space[i] for i in range(len(curr_dims))]   

        self.ig = ImageGeometry(voxel_num_x=curr_dims[0], 
                            voxel_num_y=curr_dims[1], 
                            voxel_size_x=voxel_dims[0], 
                            voxel_size_y=voxel_dims[1])
        
        self.img_dims = curr_dims
        
        if max_angle < 180:
            max_angle = 2*num_angles
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, max_angle, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])               
        else:
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, 180, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])   

        self.proj_op = ProjectionOperator(self.ig, self.ag)

        self.proj_dims = [num_angles, curr_dims[0]]

        '''Define prior'''
        self.prior_dims = prior_dims
        self.prior_lendim = int(np.prod(prior_dims))

        self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(self.prior_lendim), scale_tril=torch.eye(self.prior_lendim))

        return None
    
    def projection(self, img):
            
        img_container = ImageData(img, geometry=self.ig)
        img_phantom = self.proj_op.direct(img_container).as_array()    

        return img_phantom
    
    
    def log_likelihood(self, x):
        '''
        Evaluate log likelihood
        param: x - np array of shape 256x256
        '''
        x = x.detach().cpu().numpy()
        sino = self.projection(x)

        log_like = (-np.linalg.norm(sino - self.data)/(2*0.05))**2

        return log_like
    

    def basic_sample(self, N, Nb, model, data, z0, device, scale=0.05):
        '''
        Sampler object
        z0 - tensor of shape (1x2x2x2048) raveled, gpu based
        data - initially as num_anglesx256 numpy array
        '''
        self.scale = scale
        self.device = device
        self.model = model.to(device)
        self.data = data

        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.zeros((Ns, z0.ravel().shape[0]))
        loglike_eval = np.zeros(Ns)
        acc = np.zeros(Ns, dtype=int)

        z0_tch = torch.Tensor(z0.reshape(self.prior_dims)).to(self.device)
        x0 = self.model.decode(z0_tch)

        # States are saved as np objects  
        samples[0, :] = z0
        loglike_eval[0] = self.log_likelihood(x0)
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            print(f"Sample: {s+1}/{Ns}")
            samples[s+1, :], loglike_eval[s+1], acc[s+1] = self.single_update(samples[s, :], loglike_eval[s])

            #self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            #self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[Nb:, :]
        loglike_eval = loglike_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')

        # project samples back to image domain
        x_samples_list = []
        for i in range(samples.shape[0]):
            
            torch_z_samples = torch.Tensor(samples[i].reshape(self.prior_dims)).to(self.device)
            
            torch_x_samples = self.model.decode(torch_z_samples)

            x_samples = torch_x_samples.detach().cpu().numpy()
            x_samples_list.append(np.squeeze(x_samples))

        x_samples_list = np.stack(x_samples_list)
        print(x_samples_list.shape)

        return x_samples_list, samples, loglike_eval, accave
    

    def single_update(self, z_t, loglike_eval_t):

        z_w_tch = self.prior.rsample()
        z_w = z_w_tch.detach().cpu().numpy().reshape(self.prior_dims) # sample from the prior

        # Reshape z_t
        z_t = z_t.reshape(self.prior_dims)

        # propose state
        z_star = np.sqrt(1-self.scale**2)*z_t + self.scale*z_w   # pCN proposal

        # evaluate target
        x_star = self.model.decode(torch.Tensor(z_star).to(self.device))
        loglike_eval_star = self.log_likelihood(x_star) 

        # ratio and acceptance probability
        ratio = loglike_eval_star - loglike_eval_t  # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            z_next = z_star
            loglike_eval_next = loglike_eval_star
            acc = 1
        else:
            z_next = z_t
            loglike_eval_next = loglike_eval_t
            acc = 0

        # Reshape z_next bacl to flat
        z_next = z_next.ravel()
        
        return z_next, loglike_eval_next, acc

'''
    def _sample_adapt(self, N, Nb):
        # Set intial scale if not set
        if self.scale is None:
            self.scale = 0.1

        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        loglike_eval = np.empty(Ns)
        acc = np.zeros(Ns)

        # initial state    
        samples[:, 0] = self.x0
        loglike_eval[0] = self._loglikelihood(self.x0) 
        acc[0] = 1

        # initial adaptation params 
        Na = int(0.1*N)                              # iterations to adapt
        hat_acc = np.empty(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
        lambd = self.scale
        star_acc = 0.44    # target acceptance rate RW
        i, idx = 0, 0

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], loglike_eval[s+1], acc[s+1] = self.single_update(samples[:, s], loglike_eval[s])
            
            # adapt prop spread using acc of past samples
            if ((s+1) % Na == 0):
                # evaluate average acceptance rate
                hat_acc[i] = np.mean(acc[idx:idx+Na])

                # d. compute new scaling parameter
                zeta = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd = np.exp(np.log(lambd) + zeta*(hat_acc[i]-star_acc))

                # update parameters
                self.scale = min(lambd, 1)

                # update counters
                i += 1
                idx += Na

            # display iterations
            if ((s+1) % (max(Ns//100,1))) == 0 or (s+1) == Ns-1:
                print("\r",'Sample', s+1, '/', Ns, end="")

            self._call_callback(samples[:, s+1], s+1)

        print("\r",'Sample', s+2, '/', Ns)

        # remove burn-in
        samples = samples[:, Nb:]
        loglike_eval = loglike_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, 'MCMC scale:', self.scale, '\n')
        
        return samples, loglike_eval, accave
'''


