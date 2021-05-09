# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This class is responsible for extracting the training data used by the cWGAN Terrain Generator model
"""

# This script creates training images by finding the required coordinate and randomly grabs images around those areas
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config
from PIL import Image 

import rasterio
from osgeo import gdal
import random
random.seed(1988)

import coordinate_DB_builder as demDB

class Specific_Map_Extractor():
    def __init__(self, section_size, cluster_index=0, rebuild_dem_database = False):
        if section_size % 2 != 0:
            print("ERROR - size of {} is invalid. It has to be divisible by 2".format(section_size))
            return None
        self.section_size = section_size
        
        self.MAX_DIST_FROM_ORIGINAL_COORD = 1000 # pixels
        
        # create results folder if it doesn't exist
        self.RESULT_FOLDER = "{}/{}/rand_samples_of_specific_areas_{}x{}/{}".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, self.section_size, self.section_size, cluster_index)
        if not os.path.exists(self.RESULT_FOLDER):
            os.makedirs(self.RESULT_FOLDER)
        
        # rebuild the deb database if needed
        if rebuild_dem_database or not os.path.isfile("DEM_Data_Dict.txt"):
            demDB.ReBuildCoordinateDB("{}\{}\srtm_[0-9][0-9]_[0-9][0-9].tif".format(config.BASE_DATA_PATH, config.DATA_FOLDER));
        self.dem_database = demDB.LoadCoordinateDB()
        self.num_maps_in_database = len(self.dem_database)
        print("Map Database size: {}".format(self.num_maps_in_database))
        pass
    
    def find_info_for_tile_coord(self, tile_coord):
        """
        Finds info with the tile coordinate
        """
        for data in self.dem_database:
            if data['file_coord'][0] == tile_coord[0] and data['file_coord'][1] == tile_coord[1]:
                return data
        return None
    
    def FindMapWithCoords(self, lat, long):
        for i in range(len(self.dem_database)):
            item = self.dem_database[i]
            if item['left_bottom_corner'][0] <= lat and item['right_top_corner'][0] > lat and item['left_bottom_corner'][1] <= long and item['right_top_corner'][1] > long:
                return item
        return None
        pass
    
    def get_pixel_for_coord(self, file_path, lat, long):
        """
        Returns the pixel coordinate of an input lat-long coordinate. file_path has to be the right tile
        """
        coords2pixels = None
        with rasterio.open("{}\{}\{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, file_path)) as map_layer:
            coords2pixels = map_layer.index(long, lat) #input lon,lat
        return coords2pixels
    
    def load_tiles_around_coords(self, info, pixel_coords):
        missing_tiles = {}
        files_to_merge = []
        # add the main file
        files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, info['filename']))
        main_tile_coord = info['file_coord']
        
        right_side  = pixel_coords[0] + self.MAX_DIST_FROM_ORIGINAL_COORD
        left_side   = pixel_coords[0] - self.MAX_DIST_FROM_ORIGINAL_COORD
        top_side    = pixel_coords[1] - self.MAX_DIST_FROM_ORIGINAL_COORD
        bottom_side = pixel_coords[1] + self.MAX_DIST_FROM_ORIGINAL_COORD
        
        # Checking each surrounding tile in a clockwise direction starting at 12 o'clock
        # check top
        missing_tiles['top'] = True
        if top_side < 0:
            top_coord = [main_tile_coord[0], main_tile_coord[1]-1]
            top_tile = self.find_info_for_tile_coord(top_coord)
            if top_tile != None:
                print("Adding top")
                missing_tiles['top'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, top_tile['filename']))
        # check front_top
        missing_tiles['front_top'] = True
        if right_side >= config.TILE_SIZE and top_side < 0:
            front_top_coord = [main_tile_coord[0]+1, main_tile_coord[1]-1]
            front_top_tile = self.find_info_for_tile_coord(front_top_coord)
            if front_top_tile != None:
                print("Adding front_top")
                missing_tiles['front_top'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, front_top_tile['filename']))
        # check front
        missing_tiles['front'] = True
        if right_side >= config.TILE_SIZE:
            front_coord = [main_tile_coord[0]+1, main_tile_coord[1]]
            front_tile = self.find_info_for_tile_coord(front_coord)
            if front_tile != None:
                print("Adding front")
                missing_tiles['front'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, front_tile['filename']))
        # check front_bottom
        missing_tiles['front_bottom'] = True
        if right_side >= config.TILE_SIZE and bottom_side >= config.TILE_SIZE:
            front_bottom_coord = [main_tile_coord[0]+1, main_tile_coord[1]+1]
            front_bottom_tile = self.find_info_for_tile_coord(front_bottom_coord)
            if front_bottom_tile != None:
                print("Adding front_bottom")
                missing_tiles['front_bottom'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, front_bottom_tile['filename']))
        # check bottom
        missing_tiles['bottom'] = True
        if bottom_side >= config.TILE_SIZE:
            bottom_coord = [main_tile_coord[0], main_tile_coord[1]+1]
            bottom_tile = self.find_info_for_tile_coord(bottom_coord)
            if bottom_tile != None:
                print("Adding bottom")
                missing_tiles['bottom'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, bottom_tile['filename']))
        # check back_bottom
        missing_tiles['back_bottom'] = True
        if  left_side < 0 and bottom_side >= config.TILE_SIZE:
            back_bottom_coord = [main_tile_coord[0]-1, main_tile_coord[1]+1]
            back_bottom_tile = self.find_info_for_tile_coord(back_bottom_coord)
            if back_bottom_tile != None:
                print("Adding back_bottom")
                missing_tiles['back_bottom'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, back_bottom_tile['filename']))
        # check back
        missing_tiles['back'] = True
        if left_side < 0:
            back_coord = [main_tile_coord[0]-1, main_tile_coord[1]]
            back_tile = self.find_info_for_tile_coord(back_coord)
            if back_tile != None:
                print("Adding back")
                missing_tiles['back'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, back_tile['filename']))
        # check back_top
        missing_tiles['back_top'] = True
        if left_side < 0 and top_side < 0:
            back_top_coord = [main_tile_coord[0]-1, main_tile_coord[1]-1]
            back_top_tile = self.find_info_for_tile_coord(back_top_coord)
            if back_top_tile != None:
                print("Adding back_top")
                missing_tiles['back_top'] = False
                files_to_merge.append("{}/{}/{}".format(config.BASE_DATA_PATH, config.DATA_FOLDER, back_top_tile['filename']))
        
        # build vrt
        vrt = gdal.BuildVRT("merged_temp.vrt", files_to_merge)
        
        return (vrt, missing_tiles)
    
    def is_centroid_valid(self, x, y, missing_tiles):
        """
        Checks if the random position picked is valid.
        If the position would cause overlap with a section that doesn't exist it's not a valid position
        """
        right_side  = x + self.section_size
        left_side   = x - self.section_size
        top_side    = y - self.section_size
        bottom_side = y + self.section_size

        # check each surrounding tiles if we need them and if they exist.        
        if top_side < 0 and missing_tiles['top'] == True:
            return False
        if top_side < 0 and right_side >= config.TILE_SIZE and missing_tiles['front_top'] == True:
            return False
        if right_side >= config.TILE_SIZE and missing_tiles['front'] == True:
            return False
        if right_side >= config.TILE_SIZE and bottom_side >= config.TILE_SIZE and missing_tiles['front_bottom'] == True:
            return False
        if bottom_side >= config.TILE_SIZE and missing_tiles['bottom'] == True:
            return False
        if bottom_side >= config.TILE_SIZE and left_side < 0 and missing_tiles['back_bottom'] == True:
            return False
        if left_side < 0 and missing_tiles['back'] == True:
            return False
        if left_side < 0 and top_side < 0 and missing_tiles['back_top'] == True:
            return False
        
        # otherwise the coordinates are good
        return True
    
    def check_extentions_of_vrt(self, missing_tiles):
        """
        Checks if the 0,0 coordinate of the original coordinate system has changed.
        If tiles were added on the left or top side than the original coords have to be adjusted
        """
        extended_to_top = False
        extended_to_left = False
        if missing_tiles['top'] == False or missing_tiles['front_top'] == False or ['back_top'] == False:
            extended_to_top = True
        if missing_tiles['back'] == False or missing_tiles['back_top'] == False or ['back_bottom'] == False:
            extended_to_left = True
        
        return (extended_to_top, extended_to_left)
    
    def get_random_section_positions(self, pixel_coords, missing_tiles):
        """
        Calculates a random centroid position to grab the section from
        """        
        random_x_centroid = 0
        random_y_centroid = 0
        
        loop_counter = 0
        centroids_are_valid = False
        while centroids_are_valid == False:
            if loop_counter > 100000:
                print("ERROR - Couldn't find a valid centroid after 100000 loops")
                return None
            # pick a random center position. Avoid the edges
            random_x_centroid = random.randint(pixel_coords[0]-self.MAX_DIST_FROM_ORIGINAL_COORD, pixel_coords[0]+self.MAX_DIST_FROM_ORIGINAL_COORD)
            random_y_centroid = random.randint(pixel_coords[1]-self.MAX_DIST_FROM_ORIGINAL_COORD, pixel_coords[1]+self.MAX_DIST_FROM_ORIGINAL_COORD)
            centroids_are_valid = self.is_centroid_valid(random_x_centroid, random_y_centroid, missing_tiles)
            if centroids_are_valid == False:
                print("Invalid centroid: x: {}, y: {}".format(random_x_centroid, random_y_centroid))
        
        print("Found Random Centroid: x: {}, y: {}".format(random_x_centroid, random_y_centroid))
        return random_x_centroid, random_y_centroid
    
    def extract_map_at_coordinate(self, lat, long, num_to_generate):
        """
        Creates sections randomly around the passed in coordnates
        """
        info = self.FindMapWithCoords(lat,long)
        if info != None:
            pixel_coords = self.get_pixel_for_coord(info['filename'], lat, long) # find the pixel position of the coordinate on the tile
            print("Pixel coords: {}".format(pixel_coords))
            vrt, missing_tiles = self.load_tiles_around_coords(info, pixel_coords)
            
            extended_to_top, extended_to_left = self.check_extentions_of_vrt(missing_tiles)
            
            for i in range(num_to_generate):
                # get a random center position
                random_x_centroid, random_y_centroid = self.get_random_section_positions(pixel_coords, missing_tiles)
                if extended_to_top:
                    random_y_centroid += config.TILE_SIZE
                if extended_to_left:
                    random_x_centroid += config.TILE_SIZE
                
                print("Actual centroid: x: {}, y: {}".format(random_x_centroid, random_y_centroid))
                # cut out this sub section
                temp_section = gdal.Translate("temp_sp_section.tif", vrt, srcWin=(random_x_centroid-(self.section_size/2), random_y_centroid-(self.section_size/2), self.section_size, self.section_size))
                # get some statistics of this sub section for scaling
                band = temp_section.GetRasterBand(1)
                stats = band.GetStatistics(True, True)
                
                save_file_name = "section_near_{}_{}___{}".format(lat, long, i)
                
                # Write file info into txt
#                file_info = {}
#                file_info['Original_min_val']  = stats[0]
#                file_info['Original_max_val']  = stats[1]
#                file_info['Original_mean_val'] = stats[2]
#                file_info['Original_std_val']  = stats[3]
#                gt = temp_section.GetGeoTransform()
#                file_info['Pixel_size_x'] = gt[1]
#                file_info['Pixel_size_y'] = -gt[5]
#                with open('{}/{}.txt'.format(self.RESULT_FOLDER, save_file_name), 'w') as file:
#                    file.write(json.dumps(file_info))                
                
                # create the final version, which is rescaled to utalize all 16 bits based on max and min values of this sub section
                gdal.Translate("{}/{}_{}.png".format(self.RESULT_FOLDER, save_file_name, 0), temp_section, outputType=gdal.GDT_UInt16, scaleParams=[[stats[0], stats[1], config.U16_MIN_VAL, config.U16_MAX_VAL]])
                
                original_img = Image.open("{}/{}_0.png".format(self.RESULT_FOLDER, save_file_name))
                original_img = original_img.rotate(90)
                original_img.save("{}/{}_1.png".format(self.RESULT_FOLDER, save_file_name))
                original_img = original_img.rotate(90)
                original_img.save("{}/{}_2.png".format(self.RESULT_FOLDER, save_file_name))
                original_img = original_img.rotate(90)
                original_img.save("{}/{}_3.png".format(self.RESULT_FOLDER, save_file_name))
                
        pass
    
    def extract_maps_at_many_coordinates(self, coords, num_to_generate):
        for coord in coords:
            print("Starting work on coords: {}  {}".format(coord[0], coord[1]))
            self.extract_map_at_coordinate(coord[0], coord[1], num_to_generate)
            
        print("Extraction of all maps is Complete")
        pass
    
    pass # EndofClass 
        



def extract_alps_heightmaps(image_size, num_images_per_coordinate):
    my_extractor = Specific_Map_Extractor(image_size, 1)
    all_coords = []
    all_coords.append((46.847446, 10.748950))
    all_coords.append((46.711426, 10.206196))
    all_coords.append((46.918482, 10.940646))
    all_coords.append((47.005177, 11.088459))
    all_coords.append((46.979971, 11.746693))
    all_coords.append((47.073436, 12.078117))
    all_coords.append((46.749905, 10.284943))
    all_coords.append((46.841584, 10.111831))
    all_coords.append((46.718564, 10.405794))
    all_coords.append((46.503164, 10.533178))
    all_coords.append((46.601995, 10.098766))
    all_coords.append((46.687204, 9.860330))
    all_coords.append((46.534630, 9.716614))
    all_coords.append((46.463231, 9.610654))
    all_coords.append((46.745642, 10.389664))
    all_coords.append((46.692919, 10.182297))
    all_coords.append((46.681614, 9.823868))
    all_coords.append((46.627885, 9.882920))
    all_coords.append((46.551441, 9.755204))
    all_coords.append((46.495692, 10.031235))
    all_coords.append((46.372071, 9.827427))
    all_coords.append((46.282755, 9.624596))
    all_coords.append((46.478260, 9.391340))
    all_coords.append((46.650623, 8.371117))
    all_coords.append((46.598222, 8.198900))
    all_coords.append((46.550269, 8.065922))
    all_coords.append((46.472255, 7.863185))
    all_coords.append((46.403149, 7.876265))
    all_coords.append((46.071429, 7.930764))
    all_coords.append((45.954771, 7.530644))
    all_coords.append((45.817625, 6.833281))
    all_coords.append((45.884436, 6.936978))
    all_coords.append((45.970989, 7.006974))
    all_coords.append((45.935152, 7.729646))
    all_coords.append((46.112253, 8.028301))
    all_coords.append((46.243575, 8.095880))
    all_coords.append((46.523282, 8.019581))
    all_coords.append((47.005708, 11.855261))
    all_coords.append((47.049838, 11.963163))
    all_coords.append((47.064540, 12.080315))
    all_coords.append((47.135893, 12.111144))
    all_coords.append((47.117015, 12.607496))
    all_coords.append((47.043536, 12.974366))
    all_coords.append((47.041435, 13.257995))
    all_coords.append((47.062440, 11.115356))
    all_coords.append((46.955228, 11.180098))
    all_coords.append((46.542797, 12.027930))
    all_coords.append((46.517401, 12.192936))
    all_coords.append((46.649708, 12.110435))
    all_coords.append((45.935240, 7.847860))
    all_coords.append((45.791611, 6.833092))
    all_coords.append((45.857471, 6.903632))
    all_coords.append((45.787755, 7.016170))
    all_coords.append((45.725998, 6.825310))
    all_coords.append((45.689343, 6.848695))
    all_coords.append((45.669128, 6.913421))
    all_coords.append((45.643687, 6.920530))
    all_coords.append((45.661146, 6.980271))
    all_coords.append((45.586903, 6.989355))
    all_coords.append((45.544727, 6.984056))
    all_coords.append((45.538234, 6.860820))
    all_coords.append((45.491526, 6.802715))
    all_coords.append((45.428119, 6.882014))
    all_coords.append((45.378044, 6.799457))
    all_coords.append((45.348036, 6.868192))
    all_coords.append((45.433290, 6.909900))
    all_coords.append((45.345550, 7.067390))
    all_coords.append((45.348616, 7.101752))
    all_coords.append((45.328763, 7.044043))
    all_coords.append((45.308567, 7.106352))
    all_coords.append((45.297647, 7.103171))
    all_coords.append((45.294580, 7.063670))
    all_coords.append((45.281962, 7.116083))
    all_coords.append((45.260554, 7.128403))
    all_coords.append((45.264025, 7.142226))
    all_coords.append((45.242163, 7.113609))
    all_coords.append((45.233886, 7.060620))
    all_coords.append((45.228014, 7.040677))
    all_coords.append((45.236909, 7.015994))
    all_coords.append((45.249073, 6.984010))
    all_coords.append((45.266783, 6.953267))
    all_coords.append((45.194332, 7.111871))
    all_coords.append((45.191820, 7.140799))
    all_coords.append((45.232021, 7.173623))
    all_coords.append((45.264402, 7.194569))
    all_coords.append((45.279954, 7.174081))
    all_coords.append((45.322814, 7.202772))
    all_coords.append((45.338147, 7.248870))
    all_coords.append((45.339644, 7.281346))
    all_coords.append((45.274292, 7.316008))
    all_coords.append((45.287782, 7.332029))
    all_coords.append((45.365585, 7.425663))
    all_coords.append((45.360201, 7.453724))
    all_coords.append((45.396353, 7.346468))
    all_coords.append((45.402520, 7.306047))
    all_coords.append((45.418033, 7.325583))
    all_coords.append((45.399572, 7.250693))
    all_coords.append((45.394384, 7.209892))
    my_extractor.extract_maps_at_many_coordinates(all_coords, num_images_per_coordinate)
    pass

def extract_sahara_heightmaps(image_size, num_images_per_coordinate):
    my_extractor = Specific_Map_Extractor(image_size, 2)
    all_coords = []
    all_coords.append((23.901129, -3.186349))
    all_coords.append((24.630498, -1.640935))
    all_coords.append((26.417823, -1.422440))
    all_coords.append((23.637195, 0.047135))
    all_coords.append((26.638182, -2.065037))
    all_coords.append((21.484287, -5.909085))
    all_coords.append((18.924550, -6.077877))
    all_coords.append((30.795660, 1.205584))
    all_coords.append((30.590930, 3.906988))
    all_coords.append((29.538282, 5.253487))
    all_coords.append((29.530426, 6.579794))
    all_coords.append((30.627379, 6.503282))
    all_coords.append((32.139119, 3.164351))
    all_coords.append((32.015740, -0.559793))
    all_coords.append((30.737387, 8.074115))
    all_coords.append((29.264283, 7.174847))
    all_coords.append((33.261349, 7.324793))
    all_coords.append((18.296635, -12.929083))
    all_coords.append((21.864509, -11.703974))
    all_coords.append((16.982215, -9.909914))
    all_coords.append((17.635137, -0.151721))
    all_coords.append((16.287570, 6.5176856))
    all_coords.append((15.846024, 8.338644))
    all_coords.append((14.926548, 11.180106))
    all_coords.append((15.968186, 11.632497))
    all_coords.append((15.075170, 15.335286))
    all_coords.append((14.719600, 16.799583))
    all_coords.append((18.055043, 17.909847))
    all_coords.append((20.865884, 22.013469))
    all_coords.append((20.483868, 28.725955))
    all_coords.append((16.416391, 27.387186))
    all_coords.append((19.915860, 25.061694))
    all_coords.append((26.017864, 25.845931))
    all_coords.append((15.369231, 31.094721))
    all_coords.append((22.801790, 30.639710))
    all_coords.append((27.314604, 25.763985))
    all_coords.append((23.662857, 25.978017))
    all_coords.append((27.668510, 9.386927))
    all_coords.append((28.191845, 0.872032))
    all_coords.append((29.577156, 24.628255))
    all_coords.append((25.495829, -9.589352))
    all_coords.append((24.247354, -11.023050))
    all_coords.append((24.050207, -8.204958))
    all_coords.append((24.566937, -5.869488))
    all_coords.append((26.032674, -3.867095))
    all_coords.append((23.442384, -5.104516))
    all_coords.append((20.377153, -1.487935))
    all_coords.append((18.502369, -0.595636))
    all_coords.append((18.408395, 3.679216))
    all_coords.append((22.981696, 5.662873))
    all_coords.append((25.636824, 5.293224))
    all_coords.append((22.977801, 5.657494))
    all_coords.append((22.974977, 5.670323))
    all_coords.append((22.964725, 5.656490))
    all_coords.append((22.969400, 5.637979))
    all_coords.append((22.977304, 5.623665))
    all_coords.append((22.997322, 5.638151))
    all_coords.append((22.983560, 5.596927))
    all_coords.append((23.029712, 5.660181))
    all_coords.append((23.059842, 5.718630))
    all_coords.append((23.048934, 5.823094))
    all_coords.append((22.995772, 5.910918))
    all_coords.append((22.946679, 5.846233))
    all_coords.append((22.475919, 3.850510))
    all_coords.append((22.478225, 3.243997))
    all_coords.append((21.710402, 3.694317))
    all_coords.append((22.079309, 2.438566))
    all_coords.append((21.338992, 4.485073))
    all_coords.append((21.636634, 1.815318))
    all_coords.append((19.969350, -4.011253))
    all_coords.append((19.597521, -5.751704))
    all_coords.append((19.320882, -9.327519))
    all_coords.append((15.735397, 7.912372))
    all_coords.append((18.647673, -13.201896))
    all_coords.append((16.702022, 18.269274))
    all_coords.append((19.732249, 11.451019))
    all_coords.append((20.391817, 28.313461))
    all_coords.append((19.468517, 20.909843))
    all_coords.append((14.586039, 24.686737))
    all_coords.append((20.629191, 10.419844))
    all_coords.append((16.111164, 10.269993))
    all_coords.append((19.777482, 14.554141))
    all_coords.append((26.897546, 23.738936))
    all_coords.append((25.585174, 20.092574))
    all_coords.append((29.116670, 21.374135))
    all_coords.append((23.838187, 26.998374))
    all_coords.append((24.150209, 11.134728))
    all_coords.append((20.408200, 27.914724))
    all_coords.append((22.296134, 1.516602))
    all_coords.append((26.999897, -0.493144))
    all_coords.append((30.503120, 1.986416))
    all_coords.append((29.901868, 6.093206))
    all_coords.append((28.024149, 8.320797))
    all_coords.append((21.929805, -7.463098))
    all_coords.append((18.849094, -8.282382))
    all_coords.append((28.020423, -5.429657))
    all_coords.append((24.511935, 12.711489))
    all_coords.append((17.398297, 15.921045))
    my_extractor.extract_maps_at_many_coordinates(all_coords, num_images_per_coordinate)
    pass


if __name__ == '__main__':
    NUM_IMAGES_PER_COORDINATE = 40
    FINAL_GENERATED_IMAGE_SIZE = 256
    
    extract_alps_heightmaps(FINAL_GENERATED_IMAGE_SIZE, NUM_IMAGES_PER_COORDINATE)
    extract_sahara_heightmaps(FINAL_GENERATED_IMAGE_SIZE, NUM_IMAGES_PER_COORDINATE)

