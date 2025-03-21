import numpy as np
#load all the dngs in a folder
from simple_camera_pipeline.python.pipeline import run_pipeline_v2

#load all dngs in a folder, run them through pipeline and output them as srgb .png

import os
import cv2

from simple_camera_pipeline.python.pipeline_utils import get_visible_raw_image, get_metadata


def load_dng(dng_path):
    raw_image = get_visible_raw_image(dng_path)
    metadata = get_metadata(dng_path)
    #print(metadata)
    params = {"input_stage": "raw", "output_stage": "gamma", "demosaic_type": ""}
    srgb_img = run_pipeline_v2(raw_image, params, metadata)

    return srgb_img

folder_path = "/Users/saitedla/Dropbox/Documents/School/UofT/lowlight_reconstruction/11050217"

dng_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dng')])

for i, dng_path in enumerate(dng_paths[::-1]):
    print(dng_path)
    srgb_img = load_dng(dng_path)
    srgb_img = (srgb_img*255).astype(np.uint8)
    cv2.imwrite(dng_path[:-4]+".png", srgb_img[:,:,::-1])