import os
from common import preprocessor, select_orientation, \
                   select_z, train_test_split, \
                   store_params, get_item, remove_spiral
                   
# Select coordinates of the slice to be used in model training
Z_COORD = -130
ERR_BOUND = 2.5

# Select filepaths fitting the necessary criteria
param_dict = preprocessor('D:/Studies/MEng_Project/LIDC-IDRI',
                          select_orientation,
                          lambda x : select_z(x, z_coord=Z_COORD, error_bound=ERR_BOUND),
                          remove_spiral
                          )

# Store paths in param.json file
store_params(param_dict, filepath=os.path.join(os.getcwd(), "common/params.json"))