from common import preprocessor, select_orientation, \
                   select_z, train_test_split, \
                   store_params, get_item
                   
# Select coordinates of the slice to be used in model training
Z_COORD = -130
ERR_BOUND = 2.5

param_dict = preprocessor('D:/Studies/MEng_Project/LIDC-IDRI',
                          select_orientation,
                          lambda x : select_z(x, z_coord=Z_COORD, error_bound=ERR_BOUND)
                          )

# Store paths in param.json file
'''
train_split, test_split = train_test_split(param_dict['filepaths'])

param_dict['train_filepaths'] = train_split
param_dict['test_filepaths'] = test_split
param_dict['filepaths'] = []
'''

store_params(param_dict)