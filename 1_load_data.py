from common import *

mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='w+', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_1, dtype='float64', mode='w+', shape=(128**2,128**2))


with open(os.path.join(os.getcwd(), "settings.json")) as f:
    param_dict = json.load(f)

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]


processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

processor_cl.prior_updater(chunk_size=100, 
                           fm_path=first_moment_path, 
                           sm_path=second_moment_path)


'''
indices = np.random.randint(0, len(train_split), size=10)
slice_df = processor_cl.sample_loadin(idx = indices)


norm_slices = rescaler(slice_df, pmin=0, pmax=1, diameter_bounds=diameter_bounds)



fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    im = ax[i//5, i%5].imshow(norm_slices[i].reshape(128, 128), cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax[i//5, i%5])
    ax[i//5, i%5].set_title(f"Reconstruction Diameter: {slice_df['Reconstruction Diameter'].values[i]}")

plt.show()
'''