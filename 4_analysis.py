from common import *

mem_file_7 = Path('D:/Studies/MEng_Project/LIDC-IDRI/samples.mymemmap') # Sample storage

sample_storage = np.memmap(filename = mem_file_7, dtype='float64', mode='r', shape=(2*128**2,128**2))


test_img = sample_storage[0].reshape(128, 128)
out_img = sample_storage[:1000]

out_img = np.mean(out_img, axis=0).reshape(128, 128)

fig, ax = plt.subplots(nrows=2, ncols=1)
im = ax[0].imshow(out_img, cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(test_img, cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax[1])
plt.show()


