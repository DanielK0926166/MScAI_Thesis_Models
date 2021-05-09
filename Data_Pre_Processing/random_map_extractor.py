# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This class is responsible for extracting the training data used by the TRSM model
"""

# This script creates training images by looping through the available data and randomly grabbing textures
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config

from osgeo import gdal
import random
random.seed(1988)

import json

import coordinate_DB_builder as demDB


class Random_Map_Extractor():
    def __init__(self, section_size, rebuild_dem_database = False):
        if section_size % 2 != 0:
            print("ERROR - size of {} is invalid. It has to be divisible by 2".format(section_size))
            return None
        self.section_size = section_size
        
        # create results folder if it doesn't exist
        self.RESULT_FOLDER = "{}/{}/rand_samples_of_all_areas_{}x{}".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, self.section_size, self.section_size)
        if not os.path.exists(self.RESULT_FOLDER):
            os.makedirs(self.RESULT_FOLDER)
        
        # rebuild the deb database if needed
        if rebuild_dem_database or not os.path.isfile("DEM_Data_Dict.txt"):
            demDB.ReBuildCoordinateDB("{}\{}\srtm_[0-9][0-9]_[0-9][0-9].tif".format(config.BASE_DATA_PATH, config.DATA_FOLDER));
        self.dem_database = demDB.LoadCoordinateDB()
        self.num_maps_in_database = len(self.dem_database)
        print("Map Database size: {}".format(self.num_maps_in_database))
    pass

    def find_info_for_filename(self, filename):
        """
        Finds info belonging to the filename
        """
        for data in self.dem_database:
            if data['filename'] == filename:
                return data
        return None
    
    def find_info_for_tile_coord(self, tile_coord):
        """
        Finds info with the tile coordinate
        """
        for data in self.dem_database:
            if data['file_coord'][0] == tile_coord[0] and data['file_coord'][1] == tile_coord[1]:
                return data
        return None
    
    def is_corner_point_valid(self, x, y, missing_tiles):
        """
        Checks if the random position picked is valid.
        If the position would cause overlap with a section that doesn't exist it's not a valid position
        """
        right_side = x + self.section_size
        bottom_side = y + self.section_size
        # check front section, if right side reaches over to the front section, it must exist
        if right_side >= config.TILE_SIZE and missing_tiles['front'] == True:
            return False
        # check bottom section, if bottom side reaches over to the bottom section, it must exist
        if bottom_side >= config.TILE_SIZE and missing_tiles['bottom'] == True:
            return False
        # check bottom_front section, if bottom side reaches over to the bottom section and right side reacher over to right section, it must exist
        if bottom_side >= config.TILE_SIZE and right_side >= config.TILE_SIZE and missing_tiles['front_bottom'] == True:
            return False
        
        # otherwise the coordinates are good
        return True
    
    def get_random_section_positions(self, missing_tiles):
        """
        Calculates a random corner position for a section
        """        
        random_x_corner_point = 0
        random_y_corner_point = 0
        
        loop_counter = 0
        corner_point_is_valid = False
        while corner_point_is_valid == False:
            if loop_counter > 100000:
                print("ERROR - Couldn't find a valid corner point after 100000 loops")
                return None
            # pick a random center position. Avoid the edges
            random_x_corner_point = random.randint(0, config.TILE_SIZE+self.section_size)
            random_y_corner_point = random.randint(0, config.TILE_SIZE+self.section_size)
            corner_point_is_valid = self.is_corner_point_valid(random_x_corner_point, random_y_corner_point, missing_tiles)
            if corner_point_is_valid == False:
                print("Invalid Corner-Point: x: {}, y: {}".format(random_x_corner_point, random_y_corner_point))
        
        print("Found Random Corner-Point: x: {}, y: {}".format(random_x_corner_point, random_y_corner_point))
        return random_x_corner_point, random_y_corner_point
    
    def build_vrt_for_tile(self, info):
        missing_tiles = {}
        files_to_merge = []
        # add the main file
        files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, info['filename']))
        main_tile_coord = info['file_coord']
        
        # add bottom
        bottom_coord = [main_tile_coord[0], main_tile_coord[1]+1]
        bottom_tile = self.find_info_for_tile_coord(bottom_coord)
        if bottom_tile != None:
            missing_tiles['bottom'] = False
            files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, bottom_tile['filename']))
        else:
            missing_tiles['bottom'] = True
        
        # add front
        front_coord = [main_tile_coord[0]+1, main_tile_coord[1]]
        front_tile = self.find_info_for_tile_coord(front_coord)
        if front_tile != None:
            missing_tiles['front'] = False
            files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, front_tile['filename']))
        else:
            missing_tiles['front'] = True
        
        # add front_bottom
        front_bottom_coord = [main_tile_coord[0]+1, main_tile_coord[1]+1]
        front_bottom_tile = self.find_info_for_tile_coord(front_bottom_coord)
        if front_bottom_tile != None:
            missing_tiles['front_bottom'] = False
            files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, front_bottom_tile['filename']))
        else:
            missing_tiles['front_bottom'] = True
        
        vrt = gdal.BuildVRT("merged_temp.vrt", files_to_merge)
        return vrt, missing_tiles
    
    def extract_a_map(self, srtm_file_name, num_to_generate):
        print("Processing: {}".format(srtm_file_name))
        info = self.find_info_for_filename(srtm_file_name)
        if info == None:
            print("ERROR - Filename '{}' cannot be find in the database".format(srtm_file_name))
            return
        
        vrt, missing_tiles = self.build_vrt_for_tile(info)
        
        # open the original geoTiff
        # loop the number of times we want a subsection of this image
        for i in range(num_to_generate):
            # get a random center position
            random_x_corner_point, random_y_corner_point = self.get_random_section_positions(missing_tiles)
            # cut out this sub section
            temp_section = gdal.Translate("temp_section.tif", vrt, srcWin=(random_x_corner_point, random_y_corner_point, self.section_size, self.section_size))
            # get some statistics of this sub section for scaling
            band = temp_section.GetRasterBand(1)
            stats = band.GetStatistics(True, True)
            if stats[3] > 50.0:
                # Write file info into txt
                file_info = {}
                file_info['Original_min_val']  = stats[0]
                file_info['Original_max_val']  = stats[1]
                file_info['Original_mean_val'] = stats[2]
                file_info['Original_std_val']  = stats[3]
                gt = temp_section.GetGeoTransform()
                file_info['Pixel_size_x'] = gt[1]
                file_info['Pixel_size_y'] = -gt[5]
                with open('{}/{}_{}.txt'.format(self.RESULT_FOLDER, srtm_file_name[:-4], i), 'w') as file:
                    file.write(json.dumps(file_info))          
                
                # create the final version, which is rescaled to utalize all 16 bits based on max and min values of this sub section
                gdal.Translate("{}/{}_{}.png".format(self.RESULT_FOLDER, srtm_file_name[:-4], i), temp_section, outputType=gdal.GDT_UInt16, scaleParams=[[stats[0], stats[1], config.U16_MIN_VAL, config.U16_MAX_VAL]])
            else:
                print("Extracted section is completely black: Minimum={:.2f}, Maximum={:.2f}, Mean={:.2f}, StdDev={:.2f}".format(stats[0], stats[1], stats[2], stats[3]))            
        pass
    
    def extract_all_maps(self, num_to_generate):
        for instance in self.dem_database:
            self.extract_a_map(instance['filename'], num_to_generate)
        print("Extraction of all maps is Complete")
        pass
    
    pass # EndofClass
        
NUM_FINAL_GENERATED_IMAGES = 20
FINAL_GENERATED_IMAGE_SIZE = 512
        
my_extractor = Random_Map_Extractor(FINAL_GENERATED_IMAGE_SIZE)
my_extractor.extract_all_maps(NUM_FINAL_GENERATED_IMAGES)

