from common import *
                   

param_dict = filter_flies('D:/Studies/MEng_Project/LIDC-IDRI',
                          select_orientation,
                          lambda x : select_z(x, z_coord=-130, error_bound=2.5)
                          )

store_params(param_dict)