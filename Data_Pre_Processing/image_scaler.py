# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

Helper file to allow scaling of generated heightmap to be of size 512x512
"""


from PIL import Image

file_to_convert = "C:/Users/daniel.kiss/Documents/GitHub/AI_Masters_Thesis/AI Research Project/Code/ProceduralMapGenegator/Interpolated_Image_0.png"
converted_file = "C:/Users/daniel.kiss/Documents/GitHub/AI_Masters_Thesis/AI Research Project/Code/ProceduralMapGenegator/Interpolated_Image_0_conv.png"

im = Image.open("{}".format(file_to_convert))
im = im.resize((512, 512))
im.save("{}".format(converted_file))
print("{} Has Been Rescaled".format(converted_file))