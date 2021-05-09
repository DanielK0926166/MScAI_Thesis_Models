# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file is responsible for gathering information about the GeoTiff data files.
It creates a list of dictionaries storing information about each tile of the GeoTiff data which
later can be used to find the right tile when searching for specific Longitude and Lattitude coordinates
"""
from osgeo import gdal
import glob
import json

DB_SAVE_FILE_NAME = "DEM_Data_Dict.txt"

def ReBuildCoordinateDB(glob_path):
    """
    Pass in a path containing the geoTIFF files. For example:
    glob_path = "X:\ML_Data\SRTM_DATA\srtm_[0-9][0-9]_[0-9][0-9].tif"
    
    It returns a list of dictionaries. Each element in the array has a dictionary with the following fields:
        - filename
        - file_coord
        - left_bottom_corner
        - left_top_corner
        - right_bottom_corner
        - right_top_corner
    This data is also cached into a file called DEM_Data_Dict.txt
    """
    demList = glob.glob(glob_path)
    
    demDB = []
    for file in demList:
        print("Working on: {}".format(file))
        file_parts = file.split('.')[0][-5:]
        part_nums = file_parts.split('_')
        f_x = int(part_nums[0])
        f_y = int(part_nums[1])
        
        data = {}
        data['filename'] = file.split('\\')[-1]
        data['file_coord'] = (f_x, f_y)
        
        tif = gdal.Open(file)
        gt = tif.GetGeoTransform()
        x_min = gt[0]
        x_size = gt[1]
        y_min = gt[3]
        y_size = gt[5]
        
        # coord in map units, as in question
        # 0   , 6000 is bottom left corner
        # 0   , 0    is top left corner
        # 6000, 6000 is bottom right corner
        # 6000, 0    is top right corner
        
        # Bottom left corner
        pixelX = 0
        pixelY = 6000
        px = pixelX * x_size + x_min #x pixel
        py = pixelY * y_size + y_min #y pixel
        data['left_bottom_corner'] = (py, px)
        # Top left corner
        pixelX = 0
        pixelY = 0
        px = pixelX * x_size + x_min #x pixel
        py = pixelY * y_size + y_min #y pixel
        data['left_top_corner'] = (py, px)
        # Bottom right corner
        pixelX = 6000
        pixelY = 6000
        px = pixelX * x_size + x_min #x pixel
        py = pixelY * y_size + y_min #y pixel
        data['right_bottom_corner'] = (py, px)
        # Top right corner
        pixelX = 6000
        pixelY = 0
        px = pixelX * x_size + x_min #x pixel
        py = pixelY * y_size + y_min #y pixel
        data['right_top_corner'] = (py, px)
        
        demDB.append(data)
    
    print("Database complete.")
    SaveCoordinateDB(demDB)
    return demDB

def SaveCoordinateDB(demDB):
    print("Saving database to: {}".format(DB_SAVE_FILE_NAME))
    with open(DB_SAVE_FILE_NAME, 'w') as f:
        json.dump(demDB, f)

def LoadCoordinateDB():
    demDB = None
    with open(DB_SAVE_FILE_NAME, "r") as f:
        demDB = json.load(f)
    return demDB
