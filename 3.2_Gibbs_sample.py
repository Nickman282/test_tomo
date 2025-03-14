import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import timeit

from common import load_params, BasicProjector, LimitedAngleFBP
from data_processor import Processor
from parametric_prior import HybridGibbsSampler, SimpleUGLASampler


dims = [256, 256]
dim = dims[0]
num_imgs = 10
num_samples = 200
burn_in = 100
img_idx = 5

deg = 135
deg_rad = (deg/180)*np.pi
noise_power = 0

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)



#-----------------------------------------------------------------------------------------------------------------------------

#mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_90_0.mymemmap') 
#store_samples = np.memmap(filename = mem_file_1, dtype='float32', mode='w+', shape=(num_samples*num_imgs, int(dim**2))) 







for i in range(num_imgs):
    start = timeit.default_timer()
    print(f"Start time:{start}")
    test_slice = test_slices[i]
    test_slice = test_slice.reshape(dims)

    # Create Sampler model
    qc_sampler = HybridGibbsSampler(curr_dims=dims, og_dims=(512, 512), 
                                    num_angles=deg, max_angle=deg_rad)

    samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                             noise_power=noise_power)

    #store_samples[i*num_samples:(i+1)*num_samples] = samples

    print(f"End time: {(timeit.default_timer() - start)/60}")

#-----------------------------------------------------------------------------------------------------------------------------


'''

    fig, ax = plt.subplots(nrows=1, ncols=3)

    im = ax[0].imshow(test_slice, cmap='Greys_r', aspect='auto')
    #plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')
    #plt.colorbar(im, ax=ax[1])

    var = np.var(samples, axis=0)
    var = np.log(1/(var.max() - var.min())*var)

    im = ax[2].imshow(var.reshape(dims), cmap='Greys', aspect='auto')
    #plt.colorbar(im, ax=ax[2])

    plt.show()

'''



'''
test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)

test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

prior_stds = [0.1, 1, 10, 100]
likel_stds = [0.05, 0.1, 1, 10]

noise_power = 0

mem_file_means = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/UGLA_Grid_means_0.mymemmap')
mean_store = np.memmap(filename = mem_file_means, dtype='float32', mode='w+', shape=(int(dim**2), 4, 4))
'''

'''
# Create Sampler model
qc_sampler = HybridGibbsSampler(curr_dims=dims, og_dims=(512, 512), 
                                    num_angles=deg, max_angle=deg_rad)

samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                            noise_power=noise_power)

fig, ax = plt.subplots()

im = ax.imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys_r', aspect='auto')

plt.show()

fig, ax = plt.subplots()

im = ax.imshow(np.std(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')

plt.show()
'''
'''
test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

noise_power = 0

# Create Sampler model
qc_sampler = HybridGibbsSampler(curr_dims=dims, og_dims=(512, 512), 
                                    num_angles=deg, max_angle=deg_rad)

samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                            noise_power=noise_power)

mean_no_noise = np.mean(samples, axis=0)

noise_power = 40

qc_sampler = HybridGibbsSampler(curr_dims=dims, og_dims=(512, 512), 
                                    num_angles=deg, max_angle=deg_rad)

samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                            noise_power=noise_power)

mean_40_noise = np.mean(samples, axis=0)


fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(mean_no_noise.reshape(dims), cmap='Greys_r', aspect='auto')

im = ax[1].imshow(mean_40_noise.reshape(dims), cmap='Greys_r', aspect='auto')

plt.show()


test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
test_slice = test_slices[5]
test_slice = test_slice.reshape(dims)

noise_power = 0

# Create Sampler model
qc_sampler = HybridGibbsSampler(curr_dims=dims, og_dims=(512, 512), 
                                    num_angles=deg, max_angle=deg_rad)

samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                            noise_power=noise_power)

fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(np.abs(test_slice-np.mean(samples, axis=0).reshape(dims)), cmap='Greys', aspect='auto')

var = np.std(samples, axis=0)
var = 1/(var.max() - var.min())*var

im = ax[1].imshow(var.reshape(dims), cmap='Greys', aspect='auto')

plt.show()
'''
'''
for i, prior_std in enumerate(prior_stds):

    for j, likel_std in enumerate(likel_stds):
        
        # Create Sampler model
        qc_sampler = SimpleUGLASampler(curr_dims=dims, og_dims=(512, 512), 
                                        num_angles=deg, max_angle=deg_rad,
                                        prior_std=prior_std, likel_std=likel_std)

        samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                                    noise_power=noise_power)
        

        mean = np.mean(samples, axis=0)

        mean_store[:, i, j] = mean

noise_power = 40

mem_file_means_noise = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/UGLA_Grid_means_20.mymemmap')
mean_store_noise = np.memmap(filename = mem_file_means_noise, dtype='float32', mode='w+', shape=(int(dim**2), 4, 4))

for i, prior_std in enumerate(prior_stds):

    for j, likel_std in enumerate(likel_stds):

        # Create Sampler model
        qc_sampler = SimpleUGLASampler(curr_dims=dims, og_dims=(512, 512), 
                                        num_angles=deg, max_angle=deg_rad,
                                        prior_std=prior_std, likel_std=likel_std)

        samples = qc_sampler.run(test_slice, N=num_samples, Nb=burn_in,
                                    noise_power=noise_power)
        

        mean = np.mean(samples, axis=0)

        mean_store_noise[:, i, j] = mean
'''

'''
#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

im = ax.imshow(test_slice, cmap='Greys_r', aspect='auto')

plt.show()
plt.close()

#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

projector = BasicProjector(num_angles=deg, max_angle=deg, curr_dims=dims)
sino_slice = projector.projection(test_slice)

im = ax.imshow(sino_slice, cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax)

plt.show()
plt.close()


#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

fbp = LimitedAngleFBP(num_angles=deg, max_angle=deg, curr_dims=dims)

fbp_recon = fbp.transform_single(test_slice, noise_power=40)

im = ax.imshow(fbp_recon, cmap='Greys_r', aspect='auto')

plt.show()
plt.close()

#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

bins = 50
norms = [np.linalg.norm(sample-np.ravel(test_slice)) for sample in samples]
ax.hist(norms, bins, ec='k')
ax.grid()

plt.show()
plt.close()

#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

im = ax.imshow(samples[np.argmin(norms)].reshape(dims), cmap='Greys_r', aspect='auto')

plt.show()
plt.close()

#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

im = ax.imshow(samples[np.argmax(norms)].reshape(dims), cmap='Greys_r', aspect='auto')

plt.show()
plt.close()

#-----------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1)

im = ax.imshow(samples[np.argmax(norms)].reshape(dims) - samples[np.argmin(norms)].reshape(dims), cmap='Greys_r', aspect='auto')
plt.colorbar(im, ax=ax)

plt.show()
plt.close()

#-----------------------------------------------------------------
'''