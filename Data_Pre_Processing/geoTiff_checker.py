# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

Helper file used to gather information about GeoTiff files
"""
from osgeo import gdal
import enum

# This script is used to get all kinds of information about a geoTiff.

class CORNER(enum.Enum):
    BOTTOM_LEFT     = 0
    BOTTOM_RIGHT    = 1
    TOP_LEFT        = 2
    TOP_RIGHT       = 3

corner_to_show = CORNER.BOTTOM_LEFT

# NOTE: srtm_37_04 is below srtm_37_03
# NOTE: srtm_36_04 is on the left of srtm_37_04

tif = gdal.Open("temp_section.tif")
gt = tif.GetGeoTransform()

x_min = gt[0]
x_size = gt[1]
y_min = gt[3]
y_size = gt[5]

pixelSizeX = gt[1]
pixelSizeY =-gt[5]
print("Pixel sizes: {} x {}".format(pixelSizeX, pixelSizeY))


cols = tif.RasterXSize #161190 
rows = tif.RasterYSize #104424
bands = tif.RasterCount #1
print("Cols: {}, Rows: {}, Bands: {}".format(cols, rows, bands))

#coord in map units, as in question.
# 0   , 6000 is bottom left corner
# 0   , 0    is top left corner
# 6000, 6000 is bottom right corner
# 6000, 0    is top right corner
pixelX = 0
pixelY = 0
if corner_to_show == CORNER.BOTTOM_LEFT:
    pixelX = 0
    pixelY = 6000
elif corner_to_show == CORNER.BOTTOM_RIGHT:
    pixelX = 6000
    pixelY = 6000
elif corner_to_show == CORNER.TOP_LEFT:
    pixelX = 0
    pixelY = 0
elif corner_to_show == CORNER.TOP_RIGHT:
    pixelX = 6000
    pixelY = 0

px = pixelX * x_size + x_min #x pixel
py = pixelY * y_size + y_min #y pixel
print("Lattitude: {}, Longitude: {}".format(py, px))


band = tif.GetRasterBand(1)
stats = band.GetStatistics(True, True)
print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (stats[0], stats[1], stats[2], stats[3]))