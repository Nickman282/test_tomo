from common import *
                   
Z_COORD = -130
ERR_BOUND = 2.5

param_dict = filter_flies('D:/Studies/MEng_Project/LIDC-IDRI',
                          select_orientation,
                          lambda x : select_z(x, z_coord=Z_COORD, error_bound=ERR_BOUND)
                          )

store_params(param_dict)