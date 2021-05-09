#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file contains the main models. Both the Generator and the Critic's structure are created here
"""
import torch
from torch import nn

class Generator(nn.Module):
    '''
    GAN's Generator
    '''
    def __init__(self, in_dim, in_chan=1):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        
        self.gen = nn.DataParallel(nn.Sequential(                
                # Conv Layer 1
                nn.ConvTranspose2d(in_dim, 1024, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                
                
                # Conv Layer 3
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Conv Layer 4
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Conv Layer 5
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Conv Layer 6
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # Conv Layer 7                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Conv Layer Output
                nn.ConvTranspose2d(64, in_chan, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            ))
        pass

    def forward(self, noise):
        x = noise.view(len(noise), self.in_dim, 1, 1)
        return self.gen(x)
    pass # EoC

class Critic(nn.Module):
    '''
    GAN's Critic
    '''
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        
        self.crit = nn.DataParallel(nn.Sequential(
            # Conv Layer 1    
            nn.utils.spectral_norm(nn.Conv2d(in_dim, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv Layer 2
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 3
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 4
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv Layer 5
            nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv Layer 6
            nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv Layer 8
            nn.utils.spectral_norm(nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0, bias=False)),
            
            nn.Flatten(),
            nn.Sigmoid()
            ))
        pass

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
    pass # EoC
    
def save_model(save_name, gen, gen_opt, crit, crit_opt):
    torch.save({
        'gen_state_dict'  : gen.state_dict(),
        'gen_optimiser'   : gen_opt.state_dict(),
        'crit_state_dict' : crit.state_dict(),
        'crit_optimiser'  : crit_opt.state_dict()
        }, save_name)
    print("Saved model to: {}".format(save_name))
    pass

def load_model(file_name):
    checkpoint = torch.load(file_name)
    gen_sd = checkpoint['gen_state_dict']
    gen_opt_sd = checkpoint['gen_optimiser']
    crit_sd = checkpoint['crit_state_dict']
    crit_opt_sd = checkpoint['crit_optimiser']
    print("Loaded model from {}".format(file_name))
    return (gen_sd, gen_opt_sd, crit_sd, crit_opt_sd)

def initialise_model_weights(model):
    return model.apply(weights_init)

def weights_init(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
        torch.nn.init.constant_(layer.bias, 0)