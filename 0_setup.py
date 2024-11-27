from common import *
                   
Z_COORD = -130
ERR_BOUND = 2.5

param_dict = filter_flies('D:/Studies/MEng_Project/LIDC-IDRI',
                          select_orientation,
                          lambda x : select_z(x, z_coord=Z_COORD, error_bound=ERR_BOUND)
                          )

# Split data into train and test, store paths in settings
train_split, test_split = train_test_split(param_dict['filepaths'])

param_dict['train_filepaths'] = train_split
param_dict['test_filepaths'] = test_split
param_dict['filepaths'] = []

store_params(param_dict)