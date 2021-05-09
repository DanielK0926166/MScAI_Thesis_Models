# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This class randomly extracts a number of images from the training data to be used later for testing
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config

import glob
import shutil

import numpy as np

NUM_TEST_EXAMPLES = 100
SUB_FOLDER_PATH = "rand_samples_of_all_areas_512x512"

move_to_path = "{}\{}\{}_test".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH)
move_from_path = "{}\{}\{}".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH)

glob_path = "{}\srtm_[0-9][0-9]_[0-9][0-9]_[0-9].png".format(move_from_path)


image_list = glob.glob(glob_path)
random_indices = np.random.choice(len(image_list), NUM_TEST_EXAMPLES, replace=False)

if not os.path.isdir(move_to_path):
    os.mkdir(move_to_path)

for index in random_indices:
    filename = image_list[index].split('\\')[-1][:-4]
    print("Moving {}".format(filename))
    shutil.move("{}/{}.png".format(move_from_path, filename), move_to_path)
    shutil.move("{}/{}.txt".format(move_from_path, filename), move_to_path)
    shutil.move("{}/{}.png.aux.xml".format(move_from_path, filename), move_to_path)
