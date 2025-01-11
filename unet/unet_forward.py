import torch
import torch.nn as nn

from torch.nn import functional as F

class ResBlock(nn.Module):
    """
    Residual flat block 
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn3 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding = 0)
        self.bn_skip = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.act_fnc = nn.LeakyReLU()

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.act_fnc(self.bn2(self.conv2(x)))
        x = self.act_fnc(self.bn3(self.conv3(x)))
        return self.act_fnc(self.bn_skip(x + skip))
    
class Downsample(nn.Module):
    """
    Residual down sampling block
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv_ds = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding = 3 // 2)
        self.bn_ds = nn.BatchNorm2d(channel_out, eps=1e-4)
        
        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding = 3 // 2)
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=2, padding = 0)
        self.bn_skip = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.act_fnc = nn.LeakyReLU()

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.act_fnc(self.bn_ds(self.conv_ds(x)))
        x = self.act_fnc(self.bn1(self.conv1(x)))
        return self.act_fnc(self.bn_skip(x + skip))
    
    
class Upsample(nn.Module):
    """
    Residual up-sampling block 
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.up_nn = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 1)        
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding = 1)        
        self.bn3 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.act_fnc = nn.LeakyReLU()

    def forward(self, x, prev):
        x = self.up_nn(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = torch.concat((x, prev), dim=1)
        x = self.act_fnc(self.bn2(self.conv2(x)))
        x = self.act_fnc(self.bn3(self.conv3(x)))
        return x




class UNet(nn.Module):
    """
    UNet module
    """
    def __init__(self, channel_in=1, channels=32):
        super().__init__()

        self.first_block = ResBlock(channel_in, channels) # Output: 256x256x32
        self.downsample_1 = Downsample(channels, channels*2) # Output: 128x128x64
        self.downsample_2 = Downsample(channels*2, channels*4) # Output: 64x64x128
        self.downsample_3 = Downsample(channels*4, channels*8) # Output: 32x32x256

        self.downsample_fin = Downsample(channels*8, channels*16) # Output: 16x16x512

        self.upsample_1 = Upsample(channels*16, channels*8) # Output: 32x32x256
        self.upsample_2 = Upsample(channels*8, channels*4) # Output: 64x64x128
        self.upsample_3 = Upsample(channels*4, channels*2) # Output: 128x128x64
        self.upsample_4 = Upsample(channels*2, channels) # Output: 256x256x32

        self.conv_fin_1 = nn.Conv2d(channels, channel_in, kernel_size=1, stride=1, padding=0)
        self.conv_fin_2 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)

        self.act_func = nn.Sigmoid()

    def forward(self, x):

        x0 = self.first_block(x)
        x1 = self.downsample_1(x0)
        x2 = self.downsample_2(x1)
        x3 = self.downsample_3(x2)

        x_fin = self.downsample_fin(x3)

        x_out = self.upsample_1(x_fin, x3)
        x_out = self.upsample_2(x_out, x2)
        x_out = self.upsample_3(x_out, x1)
        x_out = self.upsample_4(x_out, x0)

        x_out = self.conv_fin_1(x_out)

        return self.act_func(self.conv_fin_2(x_out + x))
    
    def loss_fn(self, x):
        recon_x = self.forward(x)

        ## Compute Loss
        # Reconstruction loss function
        loss = F.mse_loss(recon_x, x, reduction='sum')
                                  
        return loss     