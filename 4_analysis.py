from common import *

mem_file_7 = Path('D:/Studies/MEng_Project/LIDC-IDRI/samples.mymemmap') # Sample storage

sample_storage = np.memmap(filename = mem_file_7, dtype='float64', mode='r', shape=(128**2,128**2))
'''
shape = sample_storage.shape

out_img = sample_storage[shape[0]-1000:shape[0]]

out_img = np.mean(out_img[0], axis=0).reshape(128, 128)

fig, ax = plt.subplots()
im = ax.imshow(np.exp(out_img), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax)
plt.show()
'''

