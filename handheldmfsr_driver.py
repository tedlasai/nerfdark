import cv2
import numpy as np
import torch
import torchvision
from handheld_super_resolution.super_resolution import process
#from hdrplusdriver import load_jpeg_images
from glob import glob
import os
# Set the verbose level.
options = {'verbose': 3}

# Specify the scale (1 is demosaicking), the merging kernel, and if you want to postprocess the final image.
params = {
  "debug": True,
  "scale": 1,  # choose between 1 and 2 in practice.
  "merging": {"kernel": "handheld"},  # keep unchanged.
  "post processing": {"on": False},  # set it to False if you want to use your own ISP.
  "tuning": {},
  "grey method": "FFT",
  "mode": "bayer",
  'accumulated robustness denoiser' : {
                    'median':{'on':False, # post process median filter
                              'radius max':3,  # maximum radius of the median filter. Tt cannot be more than 14. for memory purpose
                              'max frame count': 8},
                    
                    'gauss':{'on':False, # post process gaussian filter
                             'sigma max' : 1.5, # std of the gaussian blur applied when only 1 frame is merged
                             'max frame count' : 8}, # number of merged frames above which no blur is applied
                    
                    'merge':{'on':True, # filter for the ref frame accumulation
                             'rad max': 2,# max radius of the accumulation neighborhod
                             'max multiplier': 8, # Multiplier of the covariance for single frame SR
                             'max frame count' : 8} # # number of merged frames above which no blur is applied
                    },
  }


image_path_dir = "/home/tedlasai/NerfDark/RawViz/lowlight_reconstruction/3_lux_capture_1210/iso204800_s1_00125/images"

if __name__ == "__main__":
    output_img, debug_dict = process(image_path_dir, options, params)
    robustness = debug_dict['accumulated robustness']
    #clip between 0 and 1
    robustness = (np.clip(robustness, 0, 1)*255).astype(np.uint8)
    #save the robustness image
    print("robustness image shape: ", robustness.shape)
    cv2.imwrite("robustness.png", robustness)

    print("output image shape: ", output_img.shape)
    print("mean of output image: ", np.mean(output_img))
    #multiply image so mean is 0.13
    output_img = output_img/np.mean(output_img)*0.18
    #gamma the image
    output_img = np.power(output_img, 1/2.2)
    output_img = (output_img*255).astype(np.uint8).clip(0, 255)
    

    #save the output image
    cv2.imwrite("output.png", output_img)
    print("output image saved")


# Run the algorithm.
