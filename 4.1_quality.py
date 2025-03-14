import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import load_params, MSE, sq_PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler


dims = [256, 256]
dim = dims[0]
deg = 135

num_imgs = 10
num_samples = 200
num_samples_nuts = 100
img_idx = 5


# Hybrid Gibbs Load-in
hg_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.mymemmap') 
hg_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_135_0.05.mymemmap') 
hg_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_100_0.mymemmap') 
hg_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/HG_100_0.05.mymemmap') 

samples_HG_135_0 = np.memmap(filename = hg_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_135_005 = np.memmap(filename = hg_mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_100_0 = np.memmap(filename = hg_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_HG_100_005 = np.memmap(filename = hg_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))


# RTO Load-In
rto_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_135_0.mymemmap') 
rto_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_135_0.05.mymemmap') 
rto_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.mymemmap') 
rto_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/RTO_100_0.05.mymemmap') 

samples_RTO_135_0 = np.memmap(filename = rto_mem_file_1, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_135_005 = np.memmap(filename = rto_mem_file_2, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_100_0 = np.memmap(filename = rto_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_RTO_100_005 = np.memmap(filename = rto_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))


# NUTS Load-In
nuts_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.mymemmap') 
nuts_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_135_0.05.mymemmap') 
nuts_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_100_0.mymemmap')
nuts_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/NUTS_100_0.05.mymemmap')

samples_NUTS_135_0 = np.memmap(filename = nuts_mem_file_1, dtype='float32', mode='r', shape=(num_samples_nuts*num_imgs, int(dim**2)))
samples_NUTS_135_005 = np.memmap(filename = nuts_mem_file_2, dtype='float32', mode='r', shape=(num_samples//4*num_imgs, int(dim**2)))
samples_NUTS_100_0 = np.memmap(filename = nuts_mem_file_3, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))
samples_NUTS_100_005 = np.memmap(filename = nuts_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))


# DIP Load-In
dip_mem_file_1 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_135_0.mymemmap') 
dip_mem_file_2 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_135_0.05.mymemmap') 
dip_mem_file_3 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_100_0.mymemmap') 
#dip_mem_file_4 = Path('D:/Studies/MEng_Project/Lung_CT/raw_results/DIP_100_0.05.mymemmap') 

samples_DIP_135_0 = np.memmap(filename = dip_mem_file_1, dtype='float32', mode='r', shape=(num_imgs, int(dim**2)))
samples_DIP_135_005 = np.memmap(filename = dip_mem_file_2, dtype='float32', mode='r', shape=(num_imgs, int(dim**2)))
samples_DIP_100_0 = np.memmap(filename = dip_mem_file_3, dtype='float32', mode='r', shape=(num_imgs, int(dim**2)))
#samples_DIP_100_005 = np.memmap(filename = dip_mem_file_4, dtype='float32', mode='r', shape=(num_samples*num_imgs, int(dim**2)))

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "common/params.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]
pix_space = param_dict["pix_space"]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)



def get_stats(samples, img_idx=img_idx, num_imgs=num_imgs, num_samples=num_samples):
    samples_mse = []
    samples_psnr = []
    samples_ssim = []

    test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
    for i in range(num_imgs):

        mse_func = lambda x : MSE(test_slice, x)
        psnr_func = lambda x : sq_PSNR(test_slice, x)
        ssim_func = lambda x : SSIM_2(test_slice, x)

        test_slice = test_slices[i]

        sample_mse = mse_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))
        sample_psnr = psnr_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))
        sample_ssim = ssim_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))

        samples_mse.append(sample_mse)
        samples_psnr.append(sample_psnr)
        samples_ssim.append(sample_ssim)

        norm = test_slice.mean()

    rmse_HG = np.sqrt(np.mean(np.array(samples_mse).ravel()))/norm
    psnr_HG = 10*np.log10(np.mean(np.array(samples_psnr).ravel()))
    ssim_HG = np.mean(np.array(samples_ssim).ravel())

    print(f"Normalized Root Mean Square Error: {rmse_HG}")
    print(f"Peak Signal to Noise: {psnr_HG}")
    print(f"Structural Similarity Index: {ssim_HG}")

def draw_tbt(num_img, HG, RTO, NUTS, DIP, num_samples, config=1):

    fig, ax = plt.subplots(nrows=2, ncols=2)

    HG_mean = np.mean(HG[num_img*num_samples: (num_img+1)*num_samples], axis=0)
    RTO_mean = np.mean(RTO[num_img*num_samples: (num_img+1)*num_samples], axis=0)
    NUTS_mean = np.mean(NUTS[num_img*num_samples: (num_img+1)*num_samples], axis=0)


    im = ax[0, 1].imshow(HG_mean.reshape(256, 256), cmap='Greys_r', label='Gibbs', vmin = 0, vmax = 1)
    ax[0, 1].set_title("Hybrid Gibbs")
    im = ax[1, 1].imshow(RTO_mean.reshape(256, 256), cmap='Greys_r', label='RTO', vmin = 0, vmax = 1)
    ax[1, 1].set_title("RTO")
    im = ax[0, 0].imshow(NUTS_mean.reshape(256, 256), cmap='Greys_r', label='NUTS', vmin = 0, vmax = 1)
    ax[0, 0].set_title("NUTS")
    im = ax[1, 0].imshow(DIP[num_img].reshape(256, 256), cmap='Greys_r', label='DIP', vmin = 0, vmax = 1)
    ax[1, 0].set_title("DIP")

    if config == 1:
        fig.suptitle(f"Configuration 1: 135 max. degree, no noise")
    elif config==2: 
        fig.suptitle(f"Configuration 2: 135 max. degree, 40dB noise")
    elif config==3: 
        fig.suptitle(f"Configuration 3: 90 max. degree, no noise")
    plt.show()

    return None
#print(samples_DIP_135_0.05)
#get_stats(samples_DIP_100_0, num_samples=1)

#draw_tbt(5, samples_HG_100_0, samples_RTO_100_0, 
#         samples_NUTS_100_0, samples_DIP_100_0, num_samples=num_samples, config=3)

def gen_stats(samples, img_idx=img_idx, num_imgs=num_imgs, num_samples=num_samples):

    samples_mse = []
    samples_psnr = []
    samples_ssim = []

    test_slices = processor_cl.norm_loader(batch_idx=img_idx, batch_size=num_imgs)
    for i in range(num_imgs):

        test_slice = test_slices[i]

        mse_func = lambda x : MSE(test_slice, x)
        psnr_func = lambda x : sq_PSNR(test_slice, x)
        ssim_func = lambda x : SSIM_2(test_slice, x)

        sample_mse = mse_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))
        sample_psnr = psnr_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))
        sample_ssim = ssim_func(np.mean(samples[i*num_samples:(i+1)*num_samples], axis = 0))

        samples_mse.append(sample_mse)
        samples_psnr.append(sample_psnr)
        samples_ssim.append(sample_ssim)

        norm = test_slice.mean()

    rmse = np.sqrt(np.mean(np.array(samples_mse).ravel()))/norm
    psnr = 10*np.log10(np.mean(np.array(samples_psnr).ravel()))
    ssim = np.mean(np.array(samples_ssim).ravel())

    return np.sqrt(np.array(samples_mse).ravel())/norm, 10*np.log10(np.array(samples_psnr).ravel()), np.array(samples_ssim).ravel()

def plot_stats(HG, RTO, NUTS, DIP, img_idx=img_idx, num_imgs=num_imgs, num_samples=num_samples, num_samples_nuts=num_samples_nuts):

    gen_mse = []
    gen_psnr = []
    gen_ssim = []

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i in range(4):

        if i == 0:
            samples = HG
            spec_num_samples = num_samples
            color = 'r--'
            label = 'UGLA'

        elif i == 1:
            samples = RTO
            spec_num_samples = num_samples
            color = 'b:'
            label = 'RTO'

        elif i == 2:
            samples = NUTS
            spec_num_samples = num_samples_nuts
            color = 'g-'
            label = 'NUTS'

        elif i == 3:
            samples = DIP
            spec_num_samples = 1
            color = 'yo'
            label = 'DIP'
            
        mse, psnr, ssim = gen_stats(samples=samples, img_idx=img_idx, num_imgs=num_imgs, num_samples=spec_num_samples)

        gen_mse.append(mse)
        gen_psnr.append(psnr)
        gen_ssim.append(ssim)

        ax.plot(range(len(psnr)), psnr, color, label=label)

    ax.legend(loc='upper right')

    plt.show()


#plot_stats(HG=samples_HG_135_0, RTO=samples_RTO_135_0, NUTS=samples_NUTS_135_0, DIP=samples_DIP_135_0)

#samples_RTO = np.array(samples_RTO)
#samples_NUTS = np.array(samples_NUTS)

#samples_HG_mean, samples_HG_std = np.mean(samples_HG, axis=0), np.std(samples_HG, axis=0)
#samples_RTO_mean, samples_RTO_std = np.mean(samples_RTO, axis=0), np.std(samples_RTO, axis=0)
#samples_NUTS_mean, samples_NUTS_std = np.mean(samples_NUTS, axis=0), np.std(samples_NUTS, axis=0)

