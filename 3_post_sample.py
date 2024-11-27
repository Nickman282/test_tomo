from common import *

mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap
mem_file_4 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cholesky.mymemmap') # Cholesky memmap
mem_file_5 = Path('D:/Studies/MEng_Project/LIDC-IDRI/inv_cholesky.mymemmap') # Inv Cholesky memmap
mem_file_6 = Path('D:/Studies/MEng_Project/LIDC-IDRI/proposal_distr.mymemmap') # Inv Cholesky memmap
mem_file_7 = Path('D:/Studies/MEng_Project/LIDC-IDRI/samples.mymemmap') # Sample storage

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='r', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_1, dtype='float64', mode='r', shape=(128**2,128**2))
covariance_path = np.memmap(filename = mem_file_3, dtype='float64', mode='r', shape=(128**2,128**2))
cholesky_path = np.memmap(filename = mem_file_4, dtype='float64', mode='r', shape=(128**2,128**2))
inv_cholesky_path = np.memmap(filename = mem_file_5, dtype='float64', mode='r', shape=(128**2,128**2))
proposal_distr = np.memmap(filename = mem_file_6, dtype='float64', mode='r', shape=(128**2,128**2))
sample_storage = np.memmap(filename = mem_file_7, dtype='float64', mode='w+', shape=(2*128**2,128**2))

with open(os.path.join(os.getcwd(), "settings.json")) as f:
    param_dict = json.load(f)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

sampler_cl = Sampler(filepaths=filepaths, pmin=0, pmax=1,
                     diameter_bounds=diameter_bounds)

sampler_cl.projection_init(num_angles=90,
                           init_pix_space=np.array(param_dict["pix_space"]))

sampler_cl.posterior_sampler(fm_path=first_moment_path,
                             chol_path=cholesky_path,
                             chol_inv_path=inv_cholesky_path,
                             prop_path=proposal_distr,
                             sample_storage=sample_storage,
                             num_iterations=2*128**2,
                             burn_in=2000)



'''
sample = np.zeros(128**2)
for i in tqdm(range(1000)):
    sample += init_sampler(first_moment_path, cholesky_path)

sample = sample/1000

fig, ax = plt.subplots()
im = ax.imshow(np.exp(sample.reshape(128, 128)), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax)
plt.show()
'''



