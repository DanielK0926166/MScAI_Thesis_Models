# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file contains static data used by multiple other files
"""

U16_MIN_VAL = 0     # min value that can be assigned for a u16. It will be used for scaling the pixels to use all bits
U16_MAX_VAL = 65536 # max value that can be assigned for a u16. It will be used for scaling the pixels to use all bits

TILE_SIZE = 6000                # size of the tiles, SRTM has 2 options, this corresponds to the 5x5 tile size version
BASE_DATA_PATH = "X:\Data"   # Folder location where data is found and results are stored
DATA_FOLDER = "SRTM_DATA"       # Data Folder, data is loaded from here
RESULT_FOLDER = "results"       # Results Folder, data is saved here

MAX_AVAILABLE_CPU_CORES = 6     # The maximum number of cores available to be used for multiprocessor operations

DEBUG = True

def PrintDebug(log):
    if DEBUG:
        print(log)
    pass