import os
import matplotlib.pyplot as plt

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

from common import load_params
from data_processor import Processor

param_dict = load_params(os.path.join(os.getcwd(), "common/params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

i = 11

df = processor_cl.sample_loadin(idx=range(10*i, 10*i+10))
batch = processor_cl.rescaler(df, final_dims=(256, 256))

sample = np.array(batch[0]).reshape(256, 256)

ig = ImageGeometry(voxel_num_x=256, voxel_num_y=256, 
                   voxel_size_x=pix_space[0], voxel_size_y=pix_space[1])

ag = AcquisitionGeometry.create_Parallel2D()\
                   .set_angles(np.linspace(0, 135, 135, endpoint=False))\
                   .set_panel(num_pixels=256, pixel_size=pix_space[0])

A = ProjectionOperator(ig, ag)

sample = ImageData(sample, geometry=ig)

sample_sino = A.direct(sample)
sample_sino = sample_sino.as_array()

fig, ax = plt.subplots(nrows=2, ncols=2)

'''
im = ax.imshow(sample_sino, cmap='Greys_r', aspect='auto')

plt.show()

'''
#
for i in range(4):
    
    im = ax[i//2, i%2].imshow(batch[i].reshape(256, 256), cmap='Greys_r', aspect='auto')
    ax[i//2, i%2].axis('off')
    #.set_title(f"{df['Image Type'].values[i]}")
    #plt.colorbar(im, ax=ax[i//5, i%5])
    #fig.suptitle('LIDC-IDRI images')
    #ax[i//2, i%2].axis("off")

plt.show()
