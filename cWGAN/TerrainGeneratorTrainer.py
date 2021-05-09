# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file is responsible for running the training of the cWGAN Terrain Generator model.

The GAN implementation of this code was inspired, (although heavily changed) by Coursera's course:
"Generative Adversarial Networks (GANs) Specialization"
https://www.coursera.org/specializations/generative-adversarial-networks-gans
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config

import TerrainGeneratorModels as models
import TerrainGeneratorHelpers as helpers
import model_config as m_config

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from skimage import io

import os
import glob
import random

torch.manual_seed(1988) # Set for testing purposes


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = []
        
        g_path = "{}/1/*.png".format(main_dir)
        all_imgs = glob.glob(g_path)
        print("Num images: {} in label 1".format(len(all_imgs)))
        for path in all_imgs:
            self.total_imgs.append((1, path))
        
        g_path = "{}/2/*.png".format(main_dir)
        all_imgs = glob.glob(g_path)
        print("Num images: {} in label 2".format(len(all_imgs)))
        for path in all_imgs:
            self.total_imgs.append((2, path))
        
        random.shuffle(self.total_imgs)
        print("Num images: {} total".format(len(self.total_imgs)))
        pass

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx][1])
        image = io.imread(img_loc) / 65536
        tensor_image = self.transform(image).type(torch.FloatTensor)
        return (tensor_image, self.total_imgs[idx][0]-1)
    pass #EoC


def get_gradient(crit, real, fake, epsilon, one_hot_channels):
    '''
    Calculates the gradient used for the gradient penalty
    It creates an intermediate image by mixing the real and the fake images together
    based on the epsilon value, then the critic evaluates it and 
    '''
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_images_with_labels = helpers.concat_vectors(mixed_images, one_hot_channels)

    mixed_scores = crit(mixed_images_with_labels)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Gradient Penalty used to help enforce the 1-L continuity
    '''
    gradient = gradient.view(len(gradient), -1) # flatten
    gradient_norm = gradient.norm(2, dim=1)     # magnitude of the gradient per image
    # calculate the mean squared distance of the magnitude relative to 1
    diff  = gradient_norm - 1
    penalty = torch.mean(torch.mul(diff,diff))
    return penalty

def get_gen_loss(crit_fake_pred):
    '''
    Calculates the Generator's loss
    '''
    return -(torch.mean(crit_fake_pred))

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Calculates the Critic's loss
    '''
    crit_fake_mean = torch.mean(crit_fake_pred)
    crit_real_mean = torch.mean(crit_real_pred)
    crit_loss = (crit_fake_mean-crit_real_mean)+(gp*c_lambda)
    return crit_loss

def train(dpath, start_epoch=-1):
    n_epochs = 100
    batch_size = 20
    
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 5

    # Create dataloader to load the training data
    DATA_PATH = dpath
    # transform to apply to the loaded data
    transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
    dataset = CustomDataSet(DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataset_size = dataset.__len__()

    # Create generator model
    gen = models.Generator(in_dim=helpers.calculate_generator_input_dimentions(m_config.z_dim, m_config.n_classes)).to(m_config.device)
    gen = models.initialise_model_weights(gen)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    # Create critic model
    crit = models.Critic(in_dim=helpers.calculate_critic_input_dimentions(m_config.input_image_shape[0], m_config.n_classes)).to(m_config.device) 
    crit = models.initialise_model_weights(crit)
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))    
    
    
    if start_epoch != -1:
        gen_sd, gen_opt_sd, crit_sd, crit_opt_sd = models.load_model("W_GAN_{}".format(start_epoch))
        gen.load_state_dict(gen_sd)
        gen_opt.load_state_dict(gen_opt_sd)
        crit.load_state_dict(crit_sd)
        crit_opt.load_state_dict(crit_opt_sd)
    
    # Track the losses
    generator_losses = []
    critic_losses    = []
    for epoch in range(start_epoch+1, n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            
            # CLASS LABEL ENCODING
            one_hot_labels = helpers.get_one_hot_labels(labels.to(m_config.device), m_config.n_classes) # Create one-hot encoded labels
            # create one-hot encoded labels as layers to be passed to the critic
            one_hot_channels = one_hot_labels[:,:,None,None]
            one_hot_channels = one_hot_channels.repeat(1,1, m_config.input_image_shape[1], m_config.input_image_shape[2])
            
            real = real.to(m_config.device)
            
            # For each iteration update the critic multiple times
            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                
                fake_noise = helpers.get_noise(cur_batch_size, m_config.z_dim, device=m_config.device)
                fake_noise_with_labels = helpers.concat_vectors(fake_noise, one_hot_labels)
                
                fake = gen(fake_noise_with_labels) # Generate fake height map
                
                fake_image_and_labels = helpers.concat_vectors(fake.detach(), one_hot_channels) # add one-hot labels to generated image
                real_image_and_labels = helpers.concat_vectors(real,          one_hot_channels) # add one-hot labels to real image
                
                # Do predictions on the critic, both for real and fake examples
                crit_fake_pred = crit(fake_image_and_labels)
                crit_real_pred = crit(real_image_and_labels)
                
                # calculate the gradient penalty
                epsilon = torch.rand(len(real), 1, 1, 1, device=m_config.device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon, one_hot_channels)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
    
                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]
    
            ### Update generator ###
            gen_opt.zero_grad()
            
            # Get random noise and add class labels
            fake_noise = helpers.get_noise(cur_batch_size, m_config.z_dim, device=m_config.device)
            fake_noise_and_labels = helpers.concat_vectors(fake_noise, one_hot_labels)
            
            # Generate image and add labels
            fake = gen(fake_noise_and_labels)
            fake_with_labels = helpers.concat_vectors(fake, one_hot_channels)
            
            # Predict using critic
            crit_fake_pred = crit(fake_with_labels)
            
            # Calculate the loss and backward pass
            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()
    
            # Update the weights
            gen_opt.step()
    
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
    
        ### Visualization code, do at the end of each epoch ###
        gen_mean = sum(generator_losses[-dataset_size:]) / dataset_size
        crit_mean = sum(critic_losses[-dataset_size:]) / dataset_size
        print("Epoch:{}, Generator loss: {}, critic loss: {}".format(epoch, gen_mean, crit_mean))
        
        # Save images to files
        helpers.save_tensor_images(fake, cur_batch_size, "fake_images_{}.png".format(epoch))
        helpers.save_tensor_images(real, cur_batch_size, "real_images_{}.png".format(epoch))
        
        # Save figure to file
        step_bins = 20
        num_examples = (len(generator_losses) // step_bins) * step_bins
        fig = plt.figure()
        plt.plot(
            range(num_examples // step_bins), 
            torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
            label="Generator Loss"
        )
        plt.plot(
            range(num_examples // step_bins), 
            torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
            label="Critic Loss"
        )
        plt.legend()
        plt.savefig("Loss_{}".format(epoch))
        fig.clear()
        plt.close(fig)
        
        # Save model to file
        models.save_model("W_GAN_{}".format(epoch), gen, gen_opt, crit, crit_opt)
                
if __name__ == "__main__" :
    training_folder = "rand_samples_of_specific_areas_256x256"
    data_path = "{}/{}/{}".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, training_folder)
    
    train(data_path)
