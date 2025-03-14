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
from cil.plugins import TomoPhantom

# All external imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy

from common import load_params, BasicProjector, LimitedAngleFBP, MSE, PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler

dims = [256, 256]
dim = dims[0]
deg = 135
num_imgs = 10
num_samples = 200
burn_in = 200
img_idx = 5
noise_power = 40

deg_rad = (deg/180)*np.pi

#mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.mymemmap') # Second moment memmap
#store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) # CVAR training data memmap

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

# Load test slice
test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)

test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

# Set up projection operators (A in Au=b) for 15 and 90 angles

projector = BasicProjector(num_angles=deg, max_angle=deg, curr_dims=dims)
sino_slice = projector.projection(test_slice)

sino_slice_cil = AcquisitionData(sino_slice, geometry=projector.ag)



# Total Variation (TV) Regularization using PDHG Algorithm
lamb_TV = 0.1
Grad = GradientOperator(projector.ig)
G = IndicatorBox(lower=0)

Agrad = BlockOperator(projector.proj_op, Grad)

# Primal/Dual stepsizes
normAgrad = Agrad.norm()
sigma = 1./normAgrad
tau = 1./normAgrad

# 1. 90 angles, no Noise
F_phantom = BlockFunction(L2NormSquared(b=sino_slice_cil), lamb_TV*MixedL21Norm())
# Phantom 1
PDHG_phantom = PDHG(f=F_phantom, operator=Agrad, g=G, sigma=sigma, tau=tau, 
                         max_iteration = 500)
PDHG_phantom.run(verbose=2)

TV_phantom = PDHG_phantom.solution

recon_TV = TV_phantom.as_array()

fig, ax = plt.subplots(nrows=1, ncols=1)

im = ax.imshow(recon_TV, cmap='Greys_r', aspect='auto')

plt.show()
plt.close()

print(MSE(test_slice, recon_TV))
print(PSNR(test_slice, recon_TV))
print(SSIM_2(test_slice, recon_TV))

