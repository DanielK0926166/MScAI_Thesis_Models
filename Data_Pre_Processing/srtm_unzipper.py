# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file is responsible for extracting the GeoTiff files from the SRTM zip file
"""

# This script is used to loop through a folder and extract the geoTiff file from each zipped file within that folder

import zipfile

from os import listdir
from os.path import isfile, join
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config

PATH = "{}/SRTMv4.1/6_5x5_TIFs".format(config.BASE_DATA_PATH)
EXTRACT_PATH = "{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER)

all_files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

for file in all_files:
    print("Extracting: {}".format(file))
    file_to_extract = file[:-4] + ".tif"
    
    with zipfile.ZipFile("{}/{}".format(PATH,file),'r') as zip_ref:
        zip_ref.extract(file_to_extract, EXTRACT_PATH)

print("Process Complete")