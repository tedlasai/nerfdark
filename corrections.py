import os

import cv2
import rawpy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import constants
from color import getMacbethGTColors
from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_metadata, get_visible_raw_image
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
    raw_image = packed[::2, ::2, :]

    print(raw_image.shape)

    metadata = get_metadata(dng_path)

    params = {"input_stage": "raw", "output_stage": "normal"}
    normalized_img = run_pipeline_v2(raw_image, params, metadata)
    params = {"input_stage": "demosaic", "output_stage": "tone"}
    srgb_im = run_pipeline_v2(normalized_img, params, metadata)

    return normalized_img, srgb_im, metadata


def get_patch_centers(top_left, bottom_right, rows=4, cols=6):
    # Compute step sizes for width and height
    patch_width = (bottom_right[0] - top_left[0]) / (cols - 1)
    patch_height = (bottom_right[1] - top_left[1]) / (rows - 1)

    # Compute center coordinates
    centers = []
    for r in range(rows):
        for c in range(cols):
            center_x = top_left[0] + (c) * patch_width
            center_y = top_left[1] + (r) * patch_height
            centers.append((int(center_x), int(center_y)))

    return centers

#image_path = "0127_sunlight_photometric_response/scene_02_dim/arw/iso006400_dng/_DSC6080.dng"
image_location = "0127_sunlight_photometric_response/scene_02_dim/arw/iso006400_dng"
out_location = "out/"+image_location
os.makedirs(out_location, exist_ok=True)
for file in os.listdir(image_location):

    image_path = os.path.join(image_location, file)


    raw, srgb, metadata = load_dng(image_path)


    bottom_right  = (323*2,306*2) #black patch
    top_left = (482*2,395*2) #brown patch
    patch_locations = get_patch_centers(top_left, bottom_right)
    print(get_patch_centers(top_left, bottom_right))

    #plot the patch centers on the image
    for center in get_patch_centers(top_left, bottom_right):
        plt.plot(center[0], center[1], 'ro')
    def getRawColorsFromLocations(raw, locations):
        colors = []
        for location in locations:
            # get the patch color with median or mean depending on the constant set
            if constants.USE_MEDIAN:
                color = np.median(
                    raw[location[1] - constants.OFFSET_COLOR_READ: location[1] + constants.OFFSET_COLOR_READ,
                    location[0] - constants.OFFSET_COLOR_READ: location[0] + constants.OFFSET_COLOR_READ, :], axis=(0, 1))
            else:
                color = np.mean(
                    raw[location[1] - constants.OFFSET_COLOR_READ: location[1] + constants.OFFSET_COLOR_READ,
                    location[0] - constants.OFFSET_COLOR_READ: location[0] + constants.OFFSET_COLOR_READ, :], axis=(0, 1))
            colors.append(color)
        return colors
    raw_chart = getRawColorsFromLocations(raw, patch_locations)
    xrite_chart = getMacbethGTColors()

    #convert both to np arrays
    raw_chart = np.array(raw_chart)
    xrite_chart = np.array(xrite_chart)

    #compute a least squares solution to the problem mapping raw to xrite chart
    T = np.linalg.lstsq(raw_chart, xrite_chart, rcond=None)[0]

    #apply the transformation to the raw image
    H, W, _ = raw.shape

    #reshape the raw image to be a 2d array
    raw_reshaped = raw.reshape((H*W, 3))

    #apply the transformation
    corrected_xyz = np.dot(raw_reshaped, T)

    #reshape the corrected raw image back to its original shape
    corrected_xyz = corrected_xyz.reshape((H, W, 3))

    corrected_xyz = np.clip(corrected_xyz, 0, 1)

    #convert the corrected raw image to sRGB
    params = {"input_stage": "xyz", "output_stage": "tone"}
    corrected_srgb = (run_pipeline_v2(corrected_xyz, params, metadata)*255).astype(np.uint8)

    #save the corrected sRGB image
    out_corrected_srgb_file_path = os.path.join(out_location, file.replace(".dng", ".jpg"))
    print(out_corrected_srgb_file_path)
    cv2.imwrite(out_corrected_srgb_file_path, corrected_srgb[:,:,::-1])


