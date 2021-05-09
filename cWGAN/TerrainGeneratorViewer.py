#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file contains code that allows generation of heightmaps from a trained model.
By default this class creates an image of both classes used and images that interpolate between them.
"""
import TerrainGeneratorModels as models
import TerrainGeneratorHelpers as helpers
import model_config as m_config
import torch

class TerrainGeneratorViewer():
    def __init__(self, checkpoint_filename):
        
        # Load trained model
        print("Loading Trained Model")
        gen_sd, _, _, _ = models.load_model(checkpoint_filename)
        gen_input_dims = helpers.calculate_generator_input_dimentions(m_config.z_dim, m_config.n_classes)
        self.gen = (models.Generator(gen_input_dims)).to(m_config.device)
        self.gen.load_state_dict(gen_sd)
        
        self.gen.eval() # don't train
        pass
    
    def test_class_interpolation(self):
        """
        The interpolation function used here was taken from Coursera's course:
        "Generative Adversarial Networks (GANs) Specialization"
        https://www.coursera.org/specializations/generative-adversarial-networks-gans
        """
        print("Creating Interpolation Images")
        n_interpolations = 10
        interpolation_noise = helpers.get_noise(1, m_config.z_dim, m_config.device).repeat(n_interpolations, 1)
        
        one_hot_label_class_0 = helpers.get_one_hot_labels(torch.Tensor([0]).long(), m_config.n_classes)
        one_hot_label_class_1 = helpers.get_one_hot_labels(torch.Tensor([1]).long(), m_config.n_classes)
        
        percent_second_label = torch.linspace(0, 1, n_interpolations)[:, None]
        interpolation_labels = one_hot_label_class_0 * (1-percent_second_label) + one_hot_label_class_1 * percent_second_label
        
        noise_and_labels = helpers.concat_vectors(interpolation_noise, interpolation_labels.to(m_config.device))
        fake = self.gen(noise_and_labels)
        helpers.save_tensor_images(fake, n_interpolations, "Interpolated_Images.png")
        for i in range(n_interpolations):
            helpers.save_image(fake[i], "Interpolated_Image_No_Blur_{}.png".format(i))
            helpers.save_image_blurred(fake[i], "Interpolated_Image_{}.png".format(i))
        pass
    pass #EoC

if __name__ == "__main__" :
    # W_GAN_99 is the file name containing the pre-trained model's weigths
    viewer = TerrainGeneratorViewer("W_GAN_99")
    viewer.test_class_interpolation()
