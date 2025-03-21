import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')

img_dir = "/Users/saitedla/Dropbox/Documents/School/UofT/lowlight_reconstruction/10550217"
#get all .png files in the directory
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

print("Number of images", len(img_files))

imgs_per_set = 50

image_sets = len(img_files)//imgs_per_set



global_mean = np.empty((0,3))
global_var =np.empty((0,3))
for set in range(image_sets):
    set_mean = 0
    set_mean_square = 0
    for im_path in img_files[set*imgs_per_set:(set+1)*imgs_per_set]:

        full_path = os.path.join("noise_model_iso3200_npy", im_path)
        #read in png image
        im = cv2.imread(os.path.join(img_dir, im_path))[...,::-1]/255 #convert to RGB

        #degamma
        im = im**2.2

        R = im[:,:,0]
        G = im[:,:,1]
        B = im[:,:,2]
        im_stack = np.stack([R,G,B], axis=2)

        print(im_stack.shape)
        set_mean += im_stack
        set_mean_square += im_stack ** 2
        print("IM_PATTH", im_path)

    mean = set_mean/imgs_per_set
    var = (set_mean_square/imgs_per_set) - (mean**2)

    mean = np.reshape(mean, (-1,3))
    var = np.reshape(var, (-1, 3))

    global_mean = np.concatenate((global_mean, mean))
    global_var = np.concatenate((global_var, var))


scale = 1
global_mean = global_mean*scale
global_var = global_var*(scale**2)

N = global_mean.shape[0]
idx = random.sample(range(N), 10000)



for ch in range(3):
    m, b = np.polyfit(global_mean[:,ch], global_var[:,ch], 1)
    print(f"CH{ch} M: {m} B:{b}")

    plt.scatter(global_mean[idx, ch], global_var[idx, ch])
    plt.plot(global_mean[idx, ch], m * global_mean[idx, ch] + b)
    plt.show()





