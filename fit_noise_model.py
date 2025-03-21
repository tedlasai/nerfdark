import os

import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')
image_sets = len(os.listdir("noise_model_iso3200_npy"))//30

paths = sorted(os.listdir("noise_model_iso3200_npy"))

global_mean = np.empty((0,4))
global_var =np.empty((0,4))
for set in range(image_sets):
    set_mean = 0
    set_mean_square = 0
    for im_path in paths[set*30:(set+1)*30]:

        full_path = os.path.join("noise_model_iso3200_npy", im_path)
        im = (np.load(full_path)-512)/2**14

        R = im[::2, ::2]
        G1 = im[1::2, ::2]
        G2 = im[::2, 1::2]
        B = im[1::2, 1::2]
        im_stack = np.stack([R,G1,G2,B], axis=2)

        print(im_stack.shape)
        set_mean += im_stack
        set_mean_square += im_stack ** 2
        print("IM_PATTH", im_path)

    mean = set_mean/30
    var = (set_mean_square/30) - (mean**2)

    mean = np.reshape(mean, (-1,4))
    var = np.reshape(var, (-1, 4))

    global_mean = np.concatenate((global_mean, mean))
    global_var = np.concatenate((global_var, var))


scale = 2**14
global_mean = global_mean*scale
global_var = global_var*(scale**2)

N = global_mean.shape[0]
idx = random.sample(range(N), 1000)



for ch in range(4):
    m, b = np.polyfit(global_mean[:,ch], global_var[:,ch], 1)
    print(f"CH{ch} M: {m} B:{b}")

    plt.scatter(global_mean[idx, ch], global_var[idx, ch])
    plt.plot(global_mean[idx, ch], m * global_mean[idx, ch] + b)
    plt.show()





