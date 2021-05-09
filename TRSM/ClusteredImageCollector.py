# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file copies the images belonging to each cluster into separate folders. One for each cluster.
This is only used to create a visual inspection of the heightmaps belonging to each cluster
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config

import shutil
from os import walk

SUB_FOLDER_PATH = "rand_samples_of_all_areas_512x512"
CLUSTER_FOLDER_PATH = "{}\{}\{}".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, "Clusters")
CLUSTER_INFO_FOLDER = "Cluster_Info"

def extract_clustered_images():
    # Delete the folder and all files in it if already exists
    if os.path.isdir(CLUSTER_FOLDER_PATH):
        shutil.rmtree(CLUSTER_FOLDER_PATH)
        
    # Recreate folder
    os.mkdir(CLUSTER_FOLDER_PATH)
    
    # Cluster info files
    cluster_info_files = []
    for (dirpath, dirnames, filenames) in walk(CLUSTER_INFO_FOLDER):
        cluster_info_files.extend(filenames)
        
    
    for cluster_index in range(len(cluster_info_files)):
        cluster_info = cluster_info_files[cluster_index]
        # Create folder for cluster
        
        dest_path = "{}/{}".format(CLUSTER_FOLDER_PATH, cluster_index)
        os.mkdir(dest_path)
        dest_info_path = "{}/info".format(dest_path)
        os.mkdir(dest_info_path)
        
        with open("{}/{}".format(CLUSTER_INFO_FOLDER, cluster_info), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_filename = line.strip()[:-4]
                print("CLuster {} : Copying: {}".format(cluster_index, img_filename))        
                from_file_png = "{}/{}/{}/{}.png".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH, img_filename)
                shutil.copy(from_file_png, dest_path)
                
                from_file_txt = "{}/{}/{}/{}.txt".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH, img_filename)
                shutil.copy(from_file_txt, dest_info_path)
    print("Process Complete")
    
extract_clustered_images()