import os
import io
import requests
import zipfile
import torch
import torchvision
import numpy as np
import align_hdrplus as align
import rawpy
import imageio
from glob import glob
import matplotlib.pyplot as plt


def load_jpeg_images(image_paths):
    """loads rgb pixels from jpeg images"""
    images = []
    for path in image_paths:
        image = torchvision.io.read_image(path)
        image = image.float() / 255
        #ungamma correct
        image = image ** 2.2
        images.append(image)

    # store the pixels in a tensor
    images = torch.stack(images)

    return images

def get_rgb_values(image_path, bayer_array=None, **kwargs):
    """using a raw file [and modified bayer pixels], get rgb pixels"""
    # open the raw image
    with rawpy.imread(image_path) as raw:
        # overwrite the original bayer array
        if bayer_array is not None:
            raw.raw_image[:] = bayer_array
        # get postprocessed rgb pixels
        rgb = raw.postprocess(**kwargs)
    return rgb


device = torch.device('cpu')

image_path_dir = "/home/tedlasai/NerfDark/RawViz/lowlight_reconstruction/3_lux_capture_1210/iso204800_s1_00125/downsampled_008/images"
all_image_paths = sorted(glob(os.path.join(image_path_dir, '*.png')))
for i in range(len(all_image_paths)-5):
    image_paths = all_image_paths[i:i+5]
    images = load_jpeg_images(image_paths)
    print(f"Image i: {i}")

    merged_image = align.align_and_merge(images, device=device, ref_idx=2)

    #clip the merged image to 0 and 1
    merged_image = torch.clamp(merged_image, 0, 1)


    #visualize the merged image and the reference image Which is the 3rd image in the burst side by side

    merged_image_vis = merged_image.permute(1, 2, 0).numpy()
    ref_image_vis = images[2].permute(1, 2, 0).numpy()

    # figure
    font_size = 14
    fig, axs = plt.subplots(1, 2, figsize=[12, 8])

    # reference image
    axs[0].imshow(ref_image_vis)
    axs[0].set_title('Reference image', fontsize=font_size)


    # merged burst
    axs[1].imshow(merged_image_vis)
    axs[1].set_title('Merged image', fontsize=font_size)

    for ax in axs:
        ax.set_aspect(1)
        ax.axis('off')
    plt.tight_layout()
    output_dir = 'hdrplus_outputs_robust'
    os.makedirs(output_dir, exist_ok=True)
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