from common import *

mem_file_1 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap') # Second moment memmap
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap') # Means memmap
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap') # Covariance memmap
mem_file_4 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cholesky.mymemmap') # Cholesky memmap
mem_file_5 = Path('D:/Studies/MEng_Project/LIDC-IDRI/inv_cholesky.mymemmap') # Inv Cholesky memmap
mem_file_6 = Path('D:/Studies/MEng_Project/LIDC-IDRI/proposal_distr.mymemmap') # Inv Cholesky memmap

first_moment_path = np.memmap(filename = mem_file_2, dtype='float64', mode='r', shape=(128**2,))
second_moment_path = np.memmap(filename = mem_file_1, dtype='float64', mode='r', shape=(128**2,128**2))
covariance_path = np.memmap(filename = mem_file_3, dtype='float64', mode='w+', shape=(128**2,128**2))
cholesky_path = np.memmap(filename = mem_file_4, dtype='float64', mode='w+', shape=(128**2,128**2))
inv_cholesky_path = np.memmap(filename = mem_file_5, dtype='float64', mode='w+', shape=(128**2,128**2))
proposal_distr = np.memmap(filename = mem_file_6, dtype='float64', mode='w+', shape=(128**2,128**2))

fig, ax = plt.subplots()
im = ax.imshow(np.exp(first_moment_path.reshape(128, 128)), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax)
plt.show()


# Covariance Calculation
step = 128

print("Covariance Calculation:")
for sub_i in tqdm(range(step)):
    sec_st_i = step*sub_i
    sec_en_i = step*sub_i + step
    for sub_j in range(step):
        sec_st_j = step*sub_j
        sec_en_j = step*sub_j + step

        mul_exp = first_moment_path.reshape(-1, 1)[sec_st_i : sec_en_i, :] @ first_moment_path.reshape(1, -1)[:, sec_st_j : sec_en_j]
        covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = second_moment_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] - mul_exp
        proposal_distr[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j]
        
        #if sub_i == sub_j:
        #    temp = 0.9*np.diag(np.diag(covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j]))
        #    proposal_distr[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = 0.1*covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] + temp
        #else:
        #    proposal_distr[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = 0.1*covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j]


# Add prior regularization to ensure positive definiteness
print("Adding prior offset to covariance")
covariance_path[:] = covariance_path[:] + 1e-3*np.diag(np.ones(128**2))
proposal_distr[:] = proposal_distr[:] + 1e-3*np.diag(np.ones(128**2))

print(covariance_path.min())
print(covariance_path.max())

# Cholesky decomposition
print("Cholesky Decomposition")
cholesky_val = torch.linalg.cholesky(torch.Tensor(covariance_path))

cholesky_path[:] = (cholesky_val.numpy())[:]
inv_cholesky_path[:] = (torch.linalg.inv(cholesky_val).numpy())[:]

proposal_distr[:] = torch.linalg.cholesky(torch.Tensor(proposal_distr)).numpy()[:]
