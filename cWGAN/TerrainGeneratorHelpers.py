#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file contains helper functions
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

def concat_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1).float()
    return combined

def save_tensor_images(image_tensor, num_images, fig_filename):
    '''
    Given a tensor of images and number of images, plots and saves the images in a grid.
    '''
    image_grid = make_grid(image_tensor[:num_images], nrow=5).detach().cpu().numpy()
    image_grid = (image_grid - np.min(image_grid))/np.ptp(image_grid) # normalize value to be between 0 and 1
    
    img = np.transpose(image_grid, (1, 2, 0))
    plt.imsave(fig_filename, img)
    pass

def save_image(image_tensor, file_name):
    img = image_tensor[:1].detach().cpu().numpy()
    img = ((img - np.min(img))/np.ptp(img) * 65536).astype(np.uint16)
    im = Image.fromarray(img[0])
    im.save(file_name)
    pass

def save_image_blurred(image_tensor, file_name):
    img = image_tensor[:1].detach().cpu().numpy()
    
    img = ((img - np.min(img))/np.ptp(img) * 65536).astype(np.uint16)
    blurred = gaussian_filter(img[0], sigma=3)
    im = Image.fromarray(blurred)
    
    im.save(file_name)
    pass



def show_tensor_images(image_tensor, num_images, fig_filename):
    '''
    Given a tensor of images and number of images, plots and shows the images in a grid.
    '''
    image_unflat = image_tensor
    image_grid = make_grid(image_unflat[:num_images], nrow=5).detach().cpu().numpy()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    pass
    
def calculate_generator_input_dimentions(z_dim, n_classes):
    return (z_dim + n_classes)
def calculate_critic_input_dimentions(image_n_layers, n_classes):
    return (image_n_layers + n_classes)

def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def get_noise(n_samples, z_dim, device):
    noise = torch.randn(n_samples, z_dim, device=device) 
    return noise