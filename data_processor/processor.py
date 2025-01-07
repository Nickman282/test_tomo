import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pydicom
import numpy as np
import pandas as pd
import tqdm

from common.func import normalization, zoom, \
                      resize, zero_oor



# Create a Processor class, allowing for easy 
# extraction of DICOM data and conversion to 
# a pandas dataframe
class Processor:
    
    # Takes in a list of valid filepaths of DICOM files
    def __init__(self, filepaths, pmin, pmax, diameter_bounds=None):
        self.filepaths = filepaths
        self.len_filepaths = len(filepaths)
        self.pmin = pmin
        self.pmax = pmax
        self.diameter_bounds = diameter_bounds
        return None

    # Loads one or more samples by index of the provided filepaths 
    def sample_loadin(self, idx=None):
        
        # Select random file if not specified
        if idx is None:
            idx = [np.random.randint(0, self.len_filepaths)]
            num = 1
        else:
            num = len(idx)

        # Extract DICOM data into dataframe, row-by-row
        df_out = pd.DataFrame()
        for i in idx:
            path = self.filepaths[i]
        
            slice_file = pydicom.dcmread(path) # Load slice_file specific_tags = tags
            
            # Extract array-like pixel data
            slice_array = slice_file.pixel_array

            # Convert the image and metadata to
            try:
                df = pd.DataFrame.from_dict((slice_file.values()))
                df[0] = df[0].apply(lambda x: pydicom.dataelem.convert_raw_data_element(x) 
                                    if isinstance(x, pydicom.dataelem.RawDataElement) else x)
                df['name'] = df[0].apply(lambda x: x.name)
                df['value'] = df[0].apply(lambda x: x.value)
                df = df[['name', 'value']]

                df = df.set_index('name').T.reset_index(drop=True)

            except:
                df = pd.DataFrame(index=[0])
                for element in slice_file:
                    # Try to create a DataFrame for each column
                    df[element.name] = pd.Series()
                    df[element.name][0] = element.value

            # Drop raw pixel data
            df = df.drop('Pixel Data', axis=1)
            df["Pixel Data"] = [slice_array]

            # Drop duplicate columns
            cols = df.columns.values
            set_cols = set(cols)
            if len(cols) != len(set_cols):
                df = df.loc[:, ~df.columns.duplicated()]
            
            # Concatenate into a singular dataframe
            if not df_out.empty:
                cols_common = np.intersect1d(df.columns.values, df_out.columns.values)
                df_out = pd.concat([df_out, df], ignore_index=True, keys=cols_common).reset_index(drop=True)
            else:
                df_out = df

        return df_out
    
    # Rescaler function:
    # Takes in a dataframe of DICOM images and metadata,
    # returns a set of regularized CT arrays suitable for the prior
    def rescaler(self, df, final_dims=(256, 256)):

        num_img = df.shape[0]

        pixel_arrays = np.stack(df["Pixel Data"].values)
        diameter = df["Reconstruction Diameter"].values.astype('int16')

        out_img = []
        for i in range(num_img):

            img = pixel_arrays[i]

            img[img < 0] = 0
            img[img > 3000] = 3000
            
            img = normalization(img, min_val=0, max_val=1)

            if type(self.diameter_bounds) != 'NoneType':
                img = zoom(img, diameter[i], self.diameter_bounds)

            img = resize(img, dims=final_dims)

            img = normalization(img, min_val=self.pmin, max_val=self.pmax)
            
            img = zero_oor(img)

            out_img.append(img)

        return out_img
    
    # Batch Loader function
    def norm_loader(self, batch_idx, batch_size, final_dims=(256, 256)):
        
        start_idx = batch_idx*batch_size
        end_idx = start_idx + batch_size

        if end_idx > self.len_filepaths:
            end_idx = self.len_filepaths

        if end_idx - start_idx <= 1:
            idx_range = [start_idx]
        else:
            idx_range = range(start_idx, end_idx)

        df = self.sample_loadin(idx_range)
        norm_slices = self.rescaler(df, final_dims=final_dims)
        norm_slices = np.stack(norm_slices)

        norm_slices = np.reshape(norm_slices, (norm_slices.shape[0], 
                                               norm_slices[0].ravel().shape[0]))
        
        return norm_slices


    # MvGaussion Prior - OUTDATED
    def prior_updater(self, chunk_size, fm_path, sm_path):

        total_size = 0
        num_chunks = np.ceil(self.len_filepaths / chunk_size).astype(int)
        fm_path[:] = 0.0
        sm_path[:] = 0.0

        for i in range(num_chunks):
            start = i*chunk_size
            stop = (i+1)*chunk_size if (i+1)*chunk_size < self.len_filepaths else self.len_filepaths

            indexes = range(start, stop)
            df = self.sample_loadin(idx=indexes)

            # Select only one file per patient
            patient_ids = df['Patient ID'].unique()
            indices = []
            for id in patient_ids:
                df_pat = df[df['Patient ID'] == id]
                pat_idx = df_pat.index.to_list()
                sel_idx = pat_idx[np.random.randint(len(pat_idx))]
                indices.append(sel_idx)

            indices = list(set(np.ravel(indices)))
            df = df.iloc[indices]

            curr_size = len(indices)
            total_size += curr_size

            rescaled_arrays = self.rescaler(df)

            # Ravel images
            ravel_arrays = np.zeros(shape=(curr_size, rescaled_arrays[0].shape[0]**2))
            for j in range(curr_size):
                ravel_arrays[j] = rescaled_arrays[j].reshape(rescaled_arrays[0].shape[0]**2)
                
            print(f"Chunk Data Shape: {ravel_arrays.shape[0]}")

            print(f"Current chunk index: {i}")
            print(f"Total size: {total_size}")
            print(f"Current chunk size: {curr_size}")   

            # Take log of the images for prior
            log_arrays = np.log(ravel_arrays + self.pmax*1e-6)

            # First Moment Update
            temp_first_moment = np.mean(log_arrays, axis=0)

            fm_path[:] = ((total_size - curr_size)*fm_path[:] + curr_size*temp_first_moment[:])/total_size    

            print(f"Current first mom. range: {fm_path.min()} -> {fm_path.max()}") 

            # Second Moment Update
            print("Second Moment Update Progress:")
            
            step = rescaled_arrays[0].shape[0]

            for sub_i in tqdm(range(step)):
                sec_st_i = step*sub_i
                sec_en_i = step*sub_i + step

                for sub_j in range(step):
                    sec_st_j = step*sub_j
                    sec_en_j = step*sub_j + step

                    N = log_arrays.shape[0]
                    temp_second_moment = (1/N)*(log_arrays.T[sec_st_i : sec_en_i] @ log_arrays[:, sec_st_j : sec_en_j])
                    temp_second_moment = ((total_size - curr_size)*sm_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] + curr_size*temp_second_moment)/total_size
                    sm_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = temp_second_moment

            print(f"Current second mom. range: {sm_path.min()} -> {sm_path.max()}")                
            
        return None