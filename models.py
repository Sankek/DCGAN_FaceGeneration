import os

import gdown

import torch
import torch.nn as nn

from utils import load_models, save_folder


class ConvLayer(nn.Module):
    def __init__(self, mode, c_in, c_out, k_size=1, padding=0, 
                 stride=1, use_relu=True, ReLU_slope=0.2, 
                 padding_mode='zeros', use_dropout=False, 
                 dropout_p=0.5, use_batchnorm = True):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_relu = use_relu

        if self.use_batchnorm:
            self.bias = False
        else:
            self.bias = True

        if mode != 'conv' and mode != 'deconv':
            print(f"{mode} is not correct; correct modes: 'conv', 'deconv'")
            raise NameError

        if mode == 'conv':
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=k_size, 
                                  padding=padding, padding_mode=padding_mode, 
                                  stride=stride, bias=self.bias)
        else:
            self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, 
                                           padding=padding, padding_mode=padding_mode, 
                                           stride=stride, bias=self.bias)

        if use_dropout:
            self.dropout = nn.Dropout(dropout_p) 

        if use_batchnorm:
            self.bn = nn.BatchNorm2d(c_out)

        if use_relu:
            self.relu = nn.LeakyReLU(ReLU_slope)


    def forward(self, x):
        out = self.conv(x)

        if self.use_batchnorm:
            out = self.bn(out)

        if self.use_dropout:
            out = self.dropout(out)

        if self.use_relu:
            out = self.relu(out)
            
        return out


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            ConvLayer('conv', c_in=3, c_out=64, k_size=4, padding=1, stride=2), # 128x128 -> 64x64
            ConvLayer('conv', c_in=64, c_out=128, k_size=4, padding=1, stride=2), # 64x64 -> 32x32
            ConvLayer('conv', c_in=128, c_out=256, k_size=4, padding=1, stride=2),  # 32x32 -> 16x16
            ConvLayer('conv', c_in=256, c_out=512, k_size=3, padding=1, stride=1), # 16x16 -> 16x16
            ConvLayer('conv', c_in=512, c_out=1024, k_size=4, padding=1, stride=2), # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(1024*8*8, 1),
        )

        
    def forward(self, inp):
        return self.model(inp)
    
    
class GeneratorNet(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(self.latent_size, 8*8*1024),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1024, 8, 8)),
            ConvLayer('deconv', c_in=1024, c_out=512, k_size=4, padding=1, stride=2, dropout_p=0.1), # 8x8 -> 16x16
            ConvLayer('deconv', c_in=512, c_out=256, k_size=4, padding=1, stride=2, dropout_p=0.1), # 16x16 -> 32x32
            ConvLayer('deconv', c_in=256, c_out=128, k_size=4, padding=1, stride=2, dropout_p=0.1), # 32x32 -> 64x64
            ConvLayer('deconv', c_in=128, c_out=64, k_size=4, padding=1, stride=2, dropout_p=0.1), # 64x64 -> 128x128
            ConvLayer('deconv', c_in=64, c_out=3, k_size=3, padding=1, stride=1, use_relu=False, use_batchnorm=False), # 128x128 -> 128x128
            nn.Tanh()
        )
        

    def forward(self, inp):

        return self.model(inp)
    
    
# custom weights initialization called on generator and discriminator
# src: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvLayer':            
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
def load_final_state(device='cuda'):
    latent_size = 100
    models = {
        "discriminator": DiscriminatorNet(),
        "generator": GeneratorNet(latent_size)
    }

    optimizers = {
        "discriminator": torch.optim.Adam(models['discriminator'].parameters()),
        "generator": torch.optim.Adam(models['generator'].parameters())
    }
    
    save_name = '08_01_22_280ep'
    save_path = os.path.join(save_folder, save_name+'.pth')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    download_url = 'https://drive.google.com/uc?id=1-GX1wjgYDAuNaV-eILNW6f1T34SZADw4'
    if not os.path.exists(save_path):
        gdown.download(download_url, save_path, quiet=False)
    losses = load_models(models['discriminator'], models['generator'], 
                         optimizers['discriminator'], optimizers['generator'], save_name, device=device)
    
    return models, optimizers, losses
    
    
    