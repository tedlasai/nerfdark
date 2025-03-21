import os

import cv2
import numpy as np
#load all the dngs in a folder
from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_metadata, get_visible_raw_image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')

def load_dng(dng_path):
    bayer = get_visible_raw_image(dng_path)
    H, W = bayer.shape
    packed = np.zeros((H // 2, W // 2, 4), dtype=bayer.dtype)

    # Extract channels
    packed[:, :, 0] = bayer[0::2, 0::2]  # Top-left R
    packed[:, :, 1] = bayer[0::2, 1::2]  # Top-right G
    packed[:, :, 2] = bayer[1::2, 0::2]  # Bottom-left G
    packed[:, :, 3] = bayer[1::2, 1::2]  # Bottom-right B

    # average the two green channels
    packed[:, :, 1] = (packed[:, :, 1] + packed[:, :, 2]) / 2
    # make the image 3 channels
    packed[:, :, 2] = packed[:, :, 3]  # move blue to the third channel
    # drop the last channel
    packed = packed[:, :, :3]

    # downsample image (so its faster to run the pipeline) - its already demosaiced as its a pixelshift image
    raw_image = packed[::4, ::4, :]

    print(raw_image.shape)

    metadata = get_metadata(dng_path)

    params = {"input_stage": "raw", "output_stage": "normal"}
    normalized_img = run_pipeline_v2(raw_image, params, metadata)
    params = {"input_stage": "demosaic", "output_stage": "tone"}
    srgb_im = run_pipeline_v2(normalized_img, params, metadata)

    return raw_image, normalized_img, srgb_im, metadata

dng_path = "/Users/saitedla/Dropbox/Documents/School/UofT/Andrew/RawViz/average_raw_images_test/iso204800_s1_04000"
#get dng_paths that have .dng extension
dng_paths = sorted([os.path.join(dng_path, f) for f in os.listdir(dng_path) if f.endswith('.dng')])
dng_paths = dng_paths[0:100]
use_load = False
if not use_load:
    raw_images = []
    for i, im_path in enumerate(dng_paths):
        print(im_path)
        raw_image, normalized_img, srgb_im, metadata = load_dng(os.path.join(dng_path, im_path))
        raw_images.append(raw_image)

        #plot a histogram of the raw image
        plt.hist(raw_image.flatten(), bins=100)
        plt.show()
        plt.waitforbuttonpress()


    #turn the raw_images into a numpy array
    raw_images = np.array(raw_images)

    #average the burst
    average_image = np.mean(raw_images, axis=0)
    #write the average image to npy
    np.save(os.path.join(dng_path, "average_image.npy"), average_image)
else:
    average_image = np.load(os.path.join(dng_path, "average_image.npy"))
    metadata = get_metadata(dng_paths[0])
    print(metadata)

print("average raw", np.mean(average_image))

#process the average image
params = {"input_stage": "raw", "output_stage": "normal"}
normalized_img = run_pipeline_v2(average_image, params, metadata)

#scale the normalized_img so the mean is 0.18
normalized_img = normalized_img * 0.18/np.mean(normalized_img)

#store the normalized image as jpg
viz_normalized_img = (np.clip(normalized_img, 0, 1)*255).astype(np.uint8)
cv2.imwrite(os.path.join(dng_path, "average_image_normalized.jpg"), viz_normalized_img[:,:,::-1])


params = {"input_stage": "demosaic", "output_stage": "tone"}
srgb_im = run_pipeline_v2(normalized_img, params, metadata)

print("average srgb", np.mean(srgb_im))

#clip the srgb image to 1
srgb_im = np.clip(srgb_im, 0, 1)

cv2.imwrite(os.path.join(dng_path, "average_image_srgb.jpg"), (srgb_im[:,:,::-1]*255).astype(np.uint8))


