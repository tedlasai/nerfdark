import os
import io
import requests
import zipfile
import torch
import torchvision
import numpy as np
from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_metadata
import align_hdrplus as align
import rawpy
import imageio
from glob import glob
import matplotlib.pyplot as plt


def load_raw_images(image_paths):
    """loads rgb pixels from jpeg images"""
    images = []
    for path in image_paths:
        images.append(get_rgb_values(path))

    # store the pixels in a tensor
    images = torch.stack(images)

    return images

def get_rgb_values(image_path, bayer_array=None, **kwargs):
    """using a raw file [and modified bayer pixels], get rgb pixels"""
    # open the raw image
    rgbs = np.load(image_path)
    print(rgbs.shape)
    rgbs = torch.tensor(rgbs).permute(2, 0, 1) 
    print("Max: ", rgbs.max())
    print("Min: ", rgbs.min())
    return rgbs

device = torch.device('cpu')

image_path_dir = "/home/tedlasai/NerfDark/RawViz/lowlight_reconstruction/3_lux_capture_1210/iso204800_s1_00125/npys"
all_image_paths = sorted(glob(os.path.join(image_path_dir, '*.npy')))
for i in range(len(all_image_paths)-5):
    if i!=168:
        continue
    image_paths = all_image_paths[i:i+5]
    images = load_raw_images(image_paths)
    print(f"Image i: {i}")

    params = {
        'input_stage': 'raw',
        'output_stage': 'normal',
        'demosaic_type': ''
    }


    merged_image = align.align_and_merge(images, device=device, ref_idx=2)
    dng_path = image_paths[0].replace("npys", "images").replace(".npy", ".dng")
    metadata = get_metadata(dng_path)

    #clip the merged image to 0 and 1
    #merged_image = torch.mean(images, dim=0)


    #visualize the merged image and the reference image Which is the 3rd image in the burst side by side

    merged_image_vis = merged_image.permute(1, 2, 0).numpy()
   
    merged_image_vis = run_pipeline_v2(merged_image_vis, params, metadata)


    ref_image_vis = images[2].permute(1, 2, 0).numpy()
    ref_image_vis = run_pipeline_v2(ref_image_vis, params, metadata)
    print("Max: ", ref_image_vis.max())
    print("Min: ", ref_image_vis.min())
    im_0_vis = images[0].permute(1, 2, 0).numpy()
    im_1_vis = images[1].permute(1, 2, 0).numpy()
    im_3_vis = images[3].permute(1, 2, 0).numpy()
    im_4_vis = images[4].permute(1, 2, 0).numpy()

    #find scale factor to make ref_image brightness of 0.18 
    scale_factor = 0.5 / ref_image_vis.mean()
    im_0_vis = np.clip(im_0_vis * scale_factor, 0, 1) ** (1/2.2)
    im_1_vis = np.clip(im_1_vis * scale_factor, 0, 1) ** (1/2.2)
    im_3_vis = np.clip(im_3_vis * scale_factor, 0, 1)
    im_4_vis = np.clip(im_4_vis * scale_factor, 0, 1)
    ref_image_vis = np.clip(ref_image_vis * scale_factor, 0, 1) 
    merged_image_vis = np.clip(merged_image_vis * scale_factor, 0, 1) 

    # figure
    font_size = 14
    fig, axs = plt.subplots(1,2, figsize=[12, 8])

    # reference image
    axs[0].imshow(ref_image_vis)
    axs[0].set_title('Reference image (Image 2)', fontsize=font_size)



    # merged burst
    axs[1].imshow(merged_image_vis)
    axs[1].set_title('Merged image', fontsize=font_size)

    # axs[1][0].imshow(im_0_vis)
    # axs[1][0].set_title('Image 0', fontsize=font_size)

    # axs[1][1].imshow(im_1_vis)
    # axs[1][1].set_title('Image 1', fontsize=font_size)

    # axs[2][0].imshow(im_3_vis)
    # axs[2][0].set_title('Image 3', fontsize=font_size)

    # axs[2][1].imshow(im_4_vis)
    # axs[2][1].set_title('Image 4', fontsize=font_size)
    #remove axes and ticks
    for ax in axs.flatten():
        ax.set_aspect(1)
        ax.axis('off')
    plt.tight_layout()



    output_dir = 'hdrplus_outputs_robust'
    os.makedirs(output_dir, exist_ok=True)
    print("Writing to ", f'{output_dir}/before_and_after_{i}.jpg')
    plt.savefig(f'{output_dir}/before_and_after_{i}.jpg', bbox_inches='tight')


# convert raw images to rgb images
#brigthness = 10
# ref_rgb = get_rgb_values(image_paths[0], no_auto_bright=True, bright=brigthness)
# merged_rgb = get_rgb_values(image_paths[0], merged_image[0], no_auto_bright=True, bright=brigthness)

# figure
# font_size = 14
# fig, axs = plt.subplots(1, 4, figsize=[12, 8])

# # crop
# crop_y = [1300, 1800]
# crop_x = [1800, 2300]

# reference image
# axs[0].imshow(ref_rgb)
# axs[0].set_title('Reference image (full)', fontsize=font_size)
# axs[1].imshow(ref_rgb[crop_y[0]:crop_y[1]:, crop_x[0]:crop_x[1], :])
# axs[1].set_title('Reference image (crop)', fontsize=font_size)

# # merged burst
# axs[2].imshow(merged_rgb)
# axs[2].set_title('Merged image (full)', fontsize=font_size)
# axs[3].imshow(merged_rgb[crop_y[0]:crop_y[1]:, crop_x[0]:crop_x[1], :])
# axs[3].set_title('Merged image (crop)', fontsize=font_size)

# for ax in axs:
#     ax.set_aspect(1)
#     ax.axis('off')
# plt.tight_layout()
# plt.savefig(f'before_and_after.jpg', bbox_inches='tight')
# plt.show()