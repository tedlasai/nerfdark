from matplotlib import pyplot as plt
import numpy as np
import rawpy

#loop through directory
import os

from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_visible_raw_image

in_dir = "/home/tedlasai/NerfDark/RawViz/lowlight_reconstruction/3_lux_capture_1210/iso204800_s1_00125/images"
out_dir = "/home/tedlasai/NerfDark/RawViz/lowlight_reconstruction/3_lux_capture_1210/iso204800_s1_00125/npys"
os.makedirs(out_dir, exist_ok=True)
files  = sorted(os.listdir(in_dir))

for file in files:
    if file.endswith(".dng"):
        print(file)
        full_path = os.path.join(in_dir, file)
        #raw_image = get_visible_raw_image(full_path)
        params = {
            'input_stage': 'raw',
            'output_stage': 'raw',
            'demosaic_type': ''
        }
        raw_mosaic = run_pipeline_v2(full_path, params)
        #pack the mosaic
        raw_demosaiced = np.empty((raw_mosaic.shape[0]//2, raw_mosaic.shape[1]//2, 4), dtype=np.float32)
        raw_demosaiced[..., 0] = raw_mosaic[0::2, 0::2]
        raw_demosaiced[..., 1] = raw_mosaic[0::2, 1::2]
        raw_demosaiced[..., 2] = raw_mosaic[1::2, 0::2]
        raw_demosaiced[..., 3] = raw_mosaic[1::2, 1::2]

        #average the two green channels
        raw_demosaiced[..., 1] = (raw_demosaiced[..., 1] + raw_demosaiced[..., 2]) / 2

        #make RGB image
        raw_rgb = raw_demosaiced[..., [0, 1, 3]]
        raw_rgb = np.rot90(raw_rgb, 2)
        #raw_rgb = np.clip(raw_rgb, 0, 1)

        #store npy files
        file = file.split(".")[0]

        np.save(os.path.join(out_dir, file + ".npy"), raw_rgb)
