import torch
import numpy as np
import scipy

from torch import nn
from torch.nn import functional as F

from einops import rearrange
  

def covariance(x, y):
    x, y = rearrange(x, 'b c h w -> b c (h w)'), rearrange(y, 'b c h w -> b c (h w)')
    B, C, HW = x.size()
    x_mean, y_mean = torch.mean(x, dim=-1), torch.mean(y, dim=-1)
    cov = torch.empty(B, C, 1)
    for i in range(B):
        for j in range(C):
            cov[i, j, 0] = (1/(HW-1))*torch.sum((x[i, j, :] - x_mean)*(y[i, j, :] - y_mean))
    return cov

class scSE(nn.Module):
    """
    Spatial Squeeze and Channel Excitation Block
    """
    def __init__(self, ch_depth):
        super().__init__()
        #Spatial squeeze
        self.weights_1 = nn.Linear(ch_depth, ch_depth // 2)
        self.act_func = nn.ELU()
        self.weights_2 = nn.Linear(ch_depth // 2, ch_depth)
        self.sig_act = nn.Sigmoid()

        # Channel squeeze
        self.conv1 = nn.Conv2d(ch_depth, 1, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.sig_act(h)

        global_pool = nn.AvgPool2d(x.shape[-1])
        z = global_pool(x)
        z = rearrange(z, 'b c h w -> b (c h w)')
        z = self.weights_1(z)
        z = self.act_func(z)
        z = self.weights_2(z)
        z = self.sig_act(z)
        z = rearrange(z, 'b (c h w) -> b c h w', h=1, w=1)
        z = torch.add(h, z)
        return z

class Downsample(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding = 0)
        self.bn1 = nn.BatchNorm2d(channel_in, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding = 3 // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1, padding = 0)
        self.bn3 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.scse = scSE(channel_out)

        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=2, padding = 0)
        self.bn4 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.act_fnc(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x*self.scse(x)
        return self.act_fnc(self.bn4(x + skip))
    
class ResNet(nn.Module):
    """
    Residual block for the encoder
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn1 = nn.BatchNorm2d(channel_in, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1, padding = 0)
        self.bn3 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.scse = scSE(channel_out)

        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=2, padding = 0)
        self.bn4 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.act_fnc(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x*self.scse(x)
        return self.act_fnc(self.bn4(x + skip))
    
class Upsample(nn.Module):
    """
    Residual up-sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.up_nn = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding = 0)
        self.bn1 = nn.BatchNorm2d(channel_in, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 1)        
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(channel_out, eps=1e-4)
        
        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding = 0)
        self.bn4 = nn.BatchNorm2d(channel_out, eps=1e-4)

        #self.scse = scSE(channel_out)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv_skip(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.act_fnc(self.bn2(self.conv2(x)))
        x = self.act_fnc(self.bn3(self.conv3(x)))
        return self.act_fnc(self.bn4(x + skip))


class Encoder(nn.Module):
    def __init__(self, channels, ch=16, latent_channels=256):
        super().__init__()

        self.conv_in = nn.Conv2d(channels, ch, 3, 1, 1) # Output: 256x256x16
        self.res_down_block1 = Downsample(ch, ch * 2) # Output: 128x128x32
        self.res_down_block2 = Downsample(ch * 2, ch * 4) # Output: 64x64x64
        self.res_down_block3 = Downsample(ch * 4, ch * 8) # Output: 32x32x128
        self.res_down_block4 = Downsample(ch * 8, ch * 16) # Output: 16x16x256
        self.conv_mu = nn.Conv2d(ch * 16, latent_channels, 1, 1, 0) # Output: 16x16x256
        self.conv_log_var = nn.Conv2d(ch * 16, latent_channels, 1, 1, 0) # Output: 16x16x256
        self.act_fnc = nn.ELU()

    def forward(self, x):

        x = self.act_fnc(self.conv_in(x)) 
        x = self.res_down_block1(x)  
        x = self.res_down_block2(x)  
        x = self.res_down_block3(x)  
        x = self.res_down_block4(x) 

        mu = self.conv_mu(x) 
        log_var = self.conv_log_var(x)  #

        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, channels, ch=16, latent_channels=256):
        super().__init__()

        self.res_block_up1 = Upsample(latent_channels, ch * 8) # Output: 32x32x128
        self.res_block_up2 = Upsample(ch * 8, ch * 4) # Output: 64x64x64
        self.res_block_up3 = Upsample(ch * 4, ch * 2) # Output: 128x128x32
        self.res_block_up4 = Upsample(ch * 2, ch) # Output: 256x256x16
        self.conv_out = nn.Conv2d(ch, channels, 1, 1, 0) # Output: 256x256x1

    def forward(self, z):
        z = self.res_block_up1(z)  
        z = self.res_block_up2(z)  
        z = self.res_block_up3(z) 
        z = self.res_block_up4(z)     
        return torch.tanh(self.conv_out(z))    

class DCVAE2(nn.Module):
    def __init__(self, device, channel_in=1, ch=16, latent_channels=256):
        super().__init__()

        self.device = device
        self.encoder = Encoder(channels=channel_in, ch=ch, latent_channels=latent_channels)
        self.decoder = Decoder(channels=channel_in, ch=ch, latent_channels=latent_channels)

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

        ## Compute Loss
        # Reconstruction loss function
        loss_L2 = F.mse_loss(recon_x, x, reduction='sum')

        x_mean, x_var = torch.var_mean(x, dim=(2, 3), keepdim=True)
        recon_x_mean, recon_x_var = torch.var_mean(recon_x, dim=(2, 3), keepdim=True)
        cov = covariance(x, recon_x).to(self.device)

        #loss_SSIM = torch.sum(torch.squeeze((1 - ((2*x_mean*recon_x_mean + 0.01)*(2*cov + 0.03))/\
        #            ((recon_x_mean**2 + x_mean**2 + 0.01)*(x_var**2 + recon_x_var**2 + 0.03)))**2))

        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        logvar = torch.clamp(logvar, max = 10.0)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss_recon = loss_L2
                
        loss = loss_recon + loss_kl                  
        return loss, loss_L2, loss_kl      