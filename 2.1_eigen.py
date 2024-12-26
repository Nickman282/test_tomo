from common import *

mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap

covariance_path = np.memmap(filename = mem_file_3, dtype='float64', mode='r', shape=(128**2,128**2))

# Calculate first k eigenvalues and eigenvectors
cov_tensor = torch.from_numpy(covariance_path)
eigenvals, eigenvec = torch.lobpcg(cov_tensor, k=100)

eigenvec = eigenvec.numpy()
gif_list = []
for i in range(eigenvec.T.shape[0]):
    gif_list.append(eigenvec.T[i].reshape(128, 128))

gif_list = np.array(gif_list)

#imageio.mimsave('D:/Studies/MEng_Project/test_file40.gif', 1e6*gif_list, duration=1000)

fig, ax = plt.subplots()
ax.scatter(range(eigenvals.shape[0])[5:], eigenvals[5:])
plt.show()

