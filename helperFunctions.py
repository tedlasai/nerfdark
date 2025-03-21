import csv

import cv2
import numpy as np
import rawpy



def optimizationTrick(a,b):
    '''
    this function exists to do an optimization trick
    So currently we use euclidean distance for least squares
    However, we want something that respects the angular distance
    A way we can do this within the framework of the np lstsq method is to
    instead just scale each of the vectors to the same norm as the vector it is mapped to
    Thus the matrix will be like a rotation matrix because the vector remains the same length from
    input to output

    This creates the issue that we don't have scaling integrated into the matrix. Thus there has to be some other
    provisions required to make sure our output matrix has adequate scaling.
    :param a: a np array of n*3
    :param b: a np array of n*3
    :return: the vectors in a scaled to match the norms of vectors in b
    '''
    #optimization trick
    #scale input vectors in a to norm of output vectors in b
    aLength = np.linalg.norm(a, axis=1)
    bLength = np.linalg.norm(b, axis=1)
    normalizedA = a*bLength[:, None]/aLength[:, None]
    return normalizedA


def xyzTosRGB(xyz):
    '''
    :param xyz: xyz input scaled between 0-1
    :return: sRGB image scaled between 0-255
    '''
    originalShape = xyz.shape
    xyz = xyz.reshape((-1, 3))
    xyzTosRGB = np.array([[3.24, -1.54, -0.498], [-0.969, 1.875, 0.04], [0.055, -0.203, 1.05]])
    sRGB = np.dot(xyzTosRGB, xyz.T).T
    sRGB = np.float32(sRGB)
    sRGB = np.clip(sRGB, 0, 1)

    powers = np.ones(sRGB.shape) * (1/2.2)
    sRGB = np.power(sRGB, powers)
    #go back to original shape for visualization
    sRGB = sRGB.reshape(originalShape)
    sRGB = sRGB.astype("float32") * 255 #need to convert for cvtcolor method
    return sRGB

def rawToXYZ(raw, rawToXYZMat):
    '''
    :param raw: raw image or raw points
    :param rawToXYZMat: matrix to translate from raw to xyz
    :return: xyz image or xyz points
    '''
    return np.dot(raw, rawToXYZMat)

def sRGBToLab(sRGB):
    '''
    simple function to translate sRGB to Lab - the expected sRGB is to be 0-255 values
    :param sRGB: 0-255 scaled srgb values
    :return: Lab values
    '''
    sRGB = sRGB/255.0
    #conversion when input is just a single SRGB vector
    if sRGB.ndim == 1:
        # a neat little trick so I can use the opencv conversion method
        # basically create a psuedo image because my input only has 1 dim
        # then do the conversion and go back to the original input dimension
        return cv2.cvtColor(np.array([[sRGB]]), cv2.COLOR_RGB2LAB).reshape((3))
    else:
        return cv2.cvtColor(sRGB, cv2.COLOR_RGB2LAB)
    return sRGB


def getRGBAndRaw(im_path):
    '''
    simple function to return the rgb visualization associated with a raw image
    :param im_path: this is the image path of a raw image file like .dng
    :return: rgb and raw array of pixels for the image at im_path
    '''
    with rawpy.imread(im_path) as rawPyRead:
        rgb = rawPyRead.postprocess()
        #read in xrite image
        parameters = {'input_stage': 'raw', "output_stage": "vignetting_correction", 'demosaic_type':'menon2007'}
        raw = run_pipeline_v2(im_path, params=parameters)

        #the raw image needs to be changed in orientation so it matches with orientation of the postprocess from rawpy
        metadata = get_metadata(im_path)
        raw = fix_orientation(raw, metadata['orientation'])

        raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        return rgb, raw