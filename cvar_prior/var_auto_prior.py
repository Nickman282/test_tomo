import torch
import numpy as np
import scipy

from torch import nn
from torch.nn import functional as F

from einops import rearrange



def gauss_kernel_init():
    n= np.zeros((15, 15))
    n[7,7] = 1
    k = scipy.ndimage.gaussian_filter(n,sigma=2)
    k_torch = torch.tensor(k)   
    return k_torch



# Learned blurring kernel form image sharpening
class BlurKernel(nn.Module):
     #Initial kernel size 5x5
    def __init__(self, channel_in=2048, channel_out=128):
        super().__init__()
        self.bconv1 = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)
        self.bbn1 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.bconv2 = nn.ConvTranspose2d(channel_out, channel_out // 16, kernel_size=2, stride=1)
        self.bbn2 = nn.BatchNorm2d(channel_out // 16, eps=1e-4)
        self.bconv3 = nn.Conv2d(channel_out // 16, 1, 3, 1, 1)

        self.bact_fnc = nn.ELU()

    def forward(self, z):
        z = self.bact_fnc(self.bbn1(self.bconv1(z)))
        z = self.bact_fnc(self.bbn2(self.bconv2(z)))
        return self.bact_fnc(self.bconv3(z))

# 0 -> 1 to -1 -> 1 normalization
def normalize(img):
    return 2*img - 1

# -1 -> 1 to 0 -> 1 normalization
def unnormalize(img):
    return 2*(img + 1)
    
class Downsample(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))
    

class Upsample(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super().__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))

class Encoder(nn.Module):
    def __init__(self, channels, ch=32, latent_channels=2048):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, ch, 7, 1, 3) # Output: 256x256x32
        self.res_down_block1 = Downsample(ch, 2 * ch) # Output: 128x128x64
        self.res_down_block2 = Downsample(2 * ch, 4 * ch) # Output: 64x64x128
        self.res_down_block3 = Downsample(4 * ch, 8 * ch) # Output: 32x32x256
        self.res_down_block4 = Downsample(8 * ch, 16 * ch) # Output: 16x16x512
        self.res_down_block5 = Downsample(16 * ch, 32 * ch) # Output: 8x8x1024
        self.res_down_block6 = Downsample(32 * ch, 64 * ch) # Output: 4x4x2048
        self.res_down_block7 = Downsample(64 * ch, 128 * ch) # Output: 2x2x4096
        self.conv_mu = nn.Conv2d(128 * ch, latent_channels, 3, 1, 1) # Output: 2x2x2048
        self.conv_log_var = nn.Conv2d(128 * ch, latent_channels, 3, 1, 1) # Output: 2x2x2048
        self.act_fnc = nn.ELU()
        
    def forward(self, x):
        x = self.act_fnc(self.conv_in(x)) # 
        x = self.res_down_block1(x)  
        x = self.res_down_block2(x)  
        x = self.res_down_block3(x)  
        x = self.res_down_block4(x)  
        x = self.res_down_block5(x)  
        x = self.res_down_block6(x)
        x = self.res_down_block7(x)
        mu = self.conv_mu(x)  # 1
        log_var = self.conv_log_var(x)  # 1

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, channels, ch=32, latent_channels=2048):
        super().__init__()
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 64, 2, 2) # Output: 4x4x2048
        self.res_up_block1 = Upsample(ch * 64, ch * 32) # Output: 8x8x1024
        self.res_up_block2 = Upsample(ch * 32, ch * 16) # Output: 16x16x512
        self.res_up_block3 = Upsample(ch * 16, ch * 8) # Output: 32x32x256
        self.res_up_block4 = Upsample(ch * 8, ch * 4) # Output: 64x64x128
        self.res_up_block5 = Upsample(ch * 4, ch * 2) # Output: 128x128x64
        self.res_up_block6 = Upsample(ch * 2, ch) # Output: 256x256x32
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1) # Output: 256x256x1
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = self.res_up_block5(x) # 128
        x = self.res_up_block6(x)
        x = torch.tanh(self.conv_out(x))

        return x 

class DCVAE(nn.Module):
    def __init__(self, device, channel_in=1, ch=32, latent_channels=2048):
        super().__init__()

        self.device = device
        self.encoder = Encoder(channels=channel_in, ch=ch, latent_channels=latent_channels)
        self.decoder = Decoder(channels=channel_in, ch=ch, latent_channels=latent_channels)
        self.blur_kernel = BlurKernel(latent_channels)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x

    def loss_fn(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)

        # Blur Optimization
        kernel = gauss_kernel_init()

        
        FT_x = torch.fft.fft2(x)
        FT_recon_x = torch.fft.fft2(recon_x)
        FT_kernel = torch.fft.fft2(kernel, s=FT_recon_x.shape[-2, -1])

        C = 0.01

        loss_recon = F.mse_loss((torch.conj(FT_kernel)/(FT_kernel**2 + C))*(FT_recon_x - FT_x), reduction='sum')

        ## Compute Loss
        # Reconstruction loss function
        #loss_recon = F.mse_loss(recon_x, x, reduction='sum')

        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        logvar = torch.clamp(logvar, max = 10.0)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
        loss = loss_recon + loss_kl                  
        return loss, loss_recon, loss_kl