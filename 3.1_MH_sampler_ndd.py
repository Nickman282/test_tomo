from common import *

dims = [256, 256]

# Load in DICOM Processor
settings_path = os.path.join(os.getcwd(), "settings.json")
param_dict = load_params(settings_path)

filepaths = param_dict["test_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths=filepaths, pmin=0, pmax=1,
                         diameter_bounds=diameter_bounds)

# Select a random test image
rng = np.random.default_rng(seed=43)
idx = rng.integers(0, processor_cl.len_filepaths, 1)

df = processor_cl.sample_loadin(idx=idx)
test_slice = processor_cl.rescaler(df, final_dims = dims)
test_slice = test_slice[0]

# Create Sampler model
qc_sampler = GenericSampler()

# Define limited angle projection of the image
qc_sampler.projector_init(curr_dims=dims, og_dims=(512, 512), 
                          num_angles=180, max_angle=np.pi/4, 
                          pix_spacing=param_dict["pix_space"])

samples = qc_sampler.run(test_slice)

# PCA for potential separation



fig, ax = plt.subplots(nrows=1, ncols=2)

im = ax[0].imshow(np.mean(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(np.var(samples, axis=0).reshape(dims), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax[1])

plt.show()