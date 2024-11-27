from .processor import *

class Sampler(Processor):
    
    def __init__(self, filepaths, pmin, pmax, diameter_bounds):

        super().__init__(filepaths, pmin, pmax, diameter_bounds)
        return None
    
    def projection_init(self, num_angles, is_limited=False, init_dims=(512, 512), 
                        curr_dims=(128, 128), init_pix_space=None):
        
        if init_pix_space is None:
            voxel_dims = 0.5
        else:
            voxel_dims = (init_dims[0]/curr_dims[0])*init_pix_space

        self.ig = ImageGeometry(voxel_num_x=curr_dims[0], 
                            voxel_num_y=curr_dims[1], 
                            voxel_size_x=voxel_dims[0], 
                            voxel_size_y=voxel_dims[1])
        
        if is_limited == True:
            max_angle = 2*num_angles
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, max_angle, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])               
        else:
            self.ag = AcquisitionGeometry.create_Parallel2D()\
                    .set_angles(np.linspace(0, 180, num_angles, endpoint=False))\
                    .set_panel(num_pixels=curr_dims[0], pixel_size=voxel_dims[0])   

        self.proj_op = ProjectionOperator(self.ig, self.ag)

        return None
    
    def projection(self, img):
            
        img_container = ImageData(img, geometry=self.ig)
        img_phantom = self.proj_op.direct(img_container).as_array()    

        return img_phantom
    
    # Calculate log likelihood
    def log_likelihood_ratio(self, prior_phantom, data_phantom):

        background_count = 5000

        prior_phantom = prior_phantom.ravel() 
        prior_counts = np.round(background_count*np.exp(-4*prior_phantom/data_phantom.max())).astype('int32')

        data_phantom = data_phantom.ravel() 
        data_counts = np.round(background_count*np.exp(-4*data_phantom/data_phantom.max())).astype('int32')

        log_likelihood = 0

        for i in range(len(prior_counts)):
            # Calculate factorial of data phantom
            log_fact = np.sum(np.log([i+1 for i in range(data_counts[i])]))
            log_likelihood += data_counts[i]*np.log(prior_counts[i]) - log_fact - prior_counts[i]
            
        return log_likelihood
    
    # Draw posterior samples
    def posterior_sampler(self, fm_path, chol_path, chol_inv_path, prop_path, 
                          sample_storage, num_iterations=5000, burn_in=1000):
        
        burn_counter = 0

        # Draw random test image
        df_test = self.sample_loadin()
        test_img = self.rescaler(df_test)[0]

        # Take test image projection
        test_phantom = self.projection(test_img)
        
        # Record distances of test image from samples "np.linalg.norm(test_img - log_sample)""
        l2_list = []

        # Convert values to tensor for faster calc.
        means_tensor = torch.Tensor(fm_path)
        chol_tensor = torch.Tensor(chol_path)
        chol_inv_tensor = torch.Tensor(chol_inv_path)
        prop_tensor = torch.Tensor(prop_path)
        

        # Draw initial sample, calculate probabilities
        curr_sample = init_sampler(fm_path, chol_path)
        curr_sample_tens = torch.Tensor(curr_sample)

        curr_log_prior = -0.5*np.sum(((curr_sample_tens - means_tensor) @ chol_inv_tensor.T).numpy()**2)

        log_curr_sample = np.exp(curr_sample)
        log_curr_sample = log_curr_sample.reshape(test_img.shape[0], test_img.shape[1])

        curr_sample_phantom = self.projection(log_curr_sample)
        curr_log_likelihood = self.log_likelihood_ratio(curr_sample_phantom, test_phantom)
        curr_log_post = curr_log_prior + curr_log_likelihood


        num_acc = 0
        roll_length = 25
        rolling_acc = np.zeros(roll_length)
        adapt = 10e-2

        print("Num. samples drawn:")
        for i in tqdm(range(num_iterations + burn_in)):

            burn_counter += 1
        
            # Draw next sample
            z = np.random.normal(0, 1, size=(test_img.shape[0]*test_img.shape[1]))
            z_tensor = torch.Tensor(z)
            z_tensor = z_tensor.permute(*torch.arange(z_tensor.ndim-1, -1, -1))
            step = adapt * prop_tensor @ z_tensor.T
            next_sample_tens = curr_sample_tens + step

            next_log_prior = -0.5*np.sum(((next_sample_tens - means_tensor) @ chol_inv_tensor.T).numpy()**2)

            print(f"Curr prior log:{curr_log_prior}")
            print(f"Next prior log:{next_log_prior}")

            next_sample = next_sample_tens.numpy()

            log_next_sample = np.exp(next_sample)
            log_next_sample = log_next_sample.reshape(test_img.shape[0], test_img.shape[1])

            next_sample_phantom = self.projection(log_next_sample)
            
            next_log_likelihood = self.log_likelihood_ratio(next_sample_phantom, test_phantom)

            print(f"Curr likel log:{curr_log_likelihood}")
            print(f"Next likel log:{next_log_likelihood}")

            next_log_post = next_log_prior + next_log_likelihood


            log_ratio = next_log_post - curr_log_post

            #rolling_acc = np.roll(rolling_acc, 1)
            if log_ratio > 0:
                curr_sample = next_sample
                curr_sample_tens = next_sample_tens
                curr_log_prior = next_log_prior
                curr_log_likelihood = next_log_likelihood
                curr_log_post = next_log_post

                #num_acc += 1
                #rolling_acc[0] = 1
            else:
                log_alpha = np.log(np.random.uniform(0, 1))
                print(f"Log ratio: {log_ratio}")
                print(f"Log alpha: {log_alpha}")
                if log_ratio > log_alpha:
                    curr_sample = next_sample
                    curr_sample_tens = next_sample_tens
                    curr_log_prior = next_log_prior
                    curr_log_likelihood = next_log_likelihood
                    curr_log_post = next_log_post

                    #num_acc += 1
                    #rolling_acc[0] = 1
                #else:
                    #rolling_acc[0] = 0

            if burn_counter > burn_in:
                adapt = 10e-5

                print(f"Distance: {np.linalg.norm(test_img - np.exp(curr_sample_tens.numpy()).reshape(test_img.shape[0], test_img.shape[1]))}")
                l2_list.append(np.linalg.norm(test_img - np.exp(curr_sample_tens.numpy()).reshape(test_img.shape[0], test_img.shape[1])))
                sample_storage[i-burn_in, :] = curr_sample[:]

                '''
                if len(samples) % 100 == 0:
                    fig, ax = plt.subplots(nrows=1, ncols=2)

                    im0 = ax[0].imshow(test_img, cmap='Greys', aspect='auto')
                    plt.colorbar(im0, ax=ax[0])

                    im1 = ax[1].imshow(np.exp(np.mean(samples, axis=0).reshape(256, 256)), cmap='Greys', aspect='auto')
                    plt.colorbar(im1, ax=ax[1])
                    plt.show()
                '''
            print("--------------------------------------------")

            #acc_ratio = np.sum(rolling_acc)/roll_length
            #adapt = 10**(10*(acc_ratio-0.25)) * 10e-2
            #adapt = 10e-4

            #print(f"Adapt: {adapt}")
            
        print(f"Acceptance ratio: {int(100*(num_acc/num_iterations))}%")

        return None    