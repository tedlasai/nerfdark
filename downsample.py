import os
import rawpy
import numpy as np
import cv2
import argparse

def center_crop(image, crop_width, crop_height):
    """Crop the center region of a 2D image."""
    h, w = image.shape
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    return image[start_y:start_y+crop_height, start_x:start_x+crop_width]

def stack_channels_demosaic(raw_image):
    """
    Convert a Bayer pattern raw image to a 3-channel image by:
    - taking every other pixel for red, green1, green2 and blue,
    - averaging the two green channels,
    - stacking channels to form an image.
    
    This operation downsamples the image by a factor of 2.
    """
    r = raw_image[0::2, 0::2]
    g1 = raw_image[0::2, 1::2]
    g2 = raw_image[1::2, 0::2]
    b = raw_image[1::2, 1::2]
    # Average the two green channels, using a higher-precision type then casting back
    g = ((g1.astype(np.uint32) + g2.astype(np.uint32)) // 2).astype(raw_image.dtype)
    return np.stack((r, g, b), axis=-1)

def process_arw_file(file_path, crop_width, crop_height, rotate=None):
    """Read the raw image from file, center crop, and demosaic (with downsampling)."""
    with rawpy.imread(file_path) as raw:
        raw_image = raw.raw_image_visible.copy()
    cropped = center_crop(raw_image, crop_width, crop_height)
    demosaiced = stack_channels_demosaic(cropped)

    if rotate == 180:
        demosaiced = np.rot90(demosaiced, 2)
    return demosaiced

def downsample_image(image, target_resolution):
    """
    Resize the given image to target_resolution using INTER_AREA interpolation.
    target_resolution should be given as a tuple (width, height).
    """
    if image.shape[0] == target_resolution[1] and image.shape[1] == target_resolution[0]:
        return image
    else:
        return cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)

def check_downsampled_outputs_exist(base_name, output_folder, downsample_resolutions):
    """
    Check if all expected downsampled outputs exist for a given base_name.
    Returns True if all expected files exist, otherwise False.
    """
    for label in downsample_resolutions:
        subfolder = os.path.join(output_folder, f"downsampled_{str(label).zfill(3)}", "images")
        output_path = os.path.join(subfolder, f"{base_name}.npy")
        if not os.path.exists(output_path):
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Process ARW files into NPY format with downsampling.")
    parser.add_argument("--input_folder", "-i", type=str, help="Folder containing ARW files")
    parser.add_argument("--output_folder", "-o", type=str, help="Base folder where NPY files will be saved")
    parser.add_argument("--start_index", type=int, default=0, help="Index of first file to process")
    parser.add_argument("--end_index", type=int, default=None, help="Index of last file to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    crop_width = 8192   # Adjust crop dimensions as needed
    crop_height = 5632

    # Define downsample resolutions.
    # "full" represents the full demosaiced image (already downsampled by 2 from raw crop)
    downsample_resolutions = {
        # "2": (4096, 2816),
        # "4": (2048, 1408),
        # "8": (1024, 704),
        "16": (512, 352),
        # "32": (256, 176),
        # "64": (128, 88),
        # "128": (64, 44),
    }

    # Process each ARW file in the input folder
    for i, filename in enumerate(sorted(os.listdir(input_folder))):
        if i < args.start_index:
            continue

        if args.end_index is not None and i >= args.end_index:
            break 

        if filename.lower().endswith(".arw"):
            file_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            # If not overwriting, check if all expected output files exist
            if not args.overwrite and check_downsampled_outputs_exist(base_name, output_folder, downsample_resolutions):
                print(f"Skipping {base_name} as all downsampled images already exist.")
                continue

            # Read and process the raw image (center crop + demosaic)
            demosaiced = process_arw_file(file_path, crop_width, crop_height, rotate=180)
            # Ensure the image is in uint16 format
            demosaiced = demosaiced.astype(np.uint16)

            # For each resolution version, save the image as an NPY file
            for label, res in downsample_resolutions.items():
                processed_image = downsample_image(demosaiced, res)
                subfolder = os.path.join(output_folder, f"downsampled_{str(label).zfill(3)}", "images")
                os.makedirs(subfolder, exist_ok=True)
                output_path = os.path.join(subfolder, f"{base_name}.npy")
                np.save(output_path, processed_image)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()