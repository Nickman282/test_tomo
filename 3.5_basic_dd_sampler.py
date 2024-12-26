from common import *

mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap
mem_file_7 = Path('D:/Studies/MEng_Project/LIDC-IDRI/samples.mymemmap') # Sample storage

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='r', shape=(128**2,))
covariance_path = np.memmap(filename = mem_file_3, dtype='float64', mode='r', shape=(128**2,128**2))
sample_storage = np.memmap(filename = mem_file_7, dtype='float64', mode='w+', shape=(2*128**2,128**2))

settings_path = os.path.join(os.getcwd(), "settings.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                       diameter_bounds=diameter_bounds)

# Select a random test image
rng = np.random.default_rng(seed=43)
idx = rng.integers(0, len(filepaths), 1)

df = processor_cl.sample_loadin(idx=idx)
test_slice = processor_cl.rescaler(df)
test_slice = test_slice[0]

cq_sampler = CQSampler()

cq_sampler.projector_init(curr_dims=(128, 128), og_dims=(512, 512), 
                          num_angles=180, max_angle=np.pi, 
                          pix_spacing=param_dict["pix_space"])

prior = cq_sampler.distribution(test_slice, prior_mean=first_moment_path,
                                    prior_cov=covariance_path)


print(type(prior))
print(sys.getsizeof(prior))

sampler = cq.sampler.MH(prior)
sample_storage[:1000] = sampler.sample(1000, 500).samples.T
