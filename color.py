import csv
import numpy as np
import constants
import pandas as pd
import operator
from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_metadata, get_visible_raw_image

from helperFunctions import optimizationTrick, xyzTosRGB, sRGBToLab, rawToXYZ


class Color():
    '''
    A class to store information about specific colors in a raw image
    '''

    def __init__(self):
        self.name = None
        self.xyz = None
        self.rgb = None
        self.lab = None
        self.rawrgb = None


def avgColors(colors, newColorName):
    '''
    function that returns the average of a list of colors
    :param colors: list of colors to average
    :param newColorName: name of the color object that will be returned
    :return: a color object containing the average of the list of colros
    '''
    avgCol = Color()
    avgCol.name = newColorName
    if(colors[0].xyz is not None):
        avgCol.xyz = np.mean([col.xyz for col in colors], axis=0)
    if(colors[0].rgb is not None):
        avgCol.rgb = np.mean([col.rgb for col in colors], axis=0)
    if(colors[0].lab is not None):
        avgCol.lab = np.mean([col.lab for col in colors], axis=0)
    if(colors[0].rawrgb is not None):
        avgCol.rawrgb = np.mean([col.rawrgb for col in colors], axis=0)

    return avgCol


def stdColors(colors, newColorname):
    '''
    function that returns the std dev of a list of colors
    :param colors: list of colors to find std dev of
    :param newColorName: name of the color object that will be returned
    :return: a color object containing the std dev of the list of colors
    '''
    stdCol = Color()
    stdCol.name = newColorname
    stdCol.xyz = np.std([col.xyz for col in colors], axis=0)
    stdCol.rgb = np.std([col.rgb for col in colors], axis=0)
    stdCol.lab = np.std([col.lab for col in colors], axis=0)
    stdCol.rawrgb = np.std([col.rawrgb for col in colors], axis=0)

    return stdCol


def getRawColorFromLocation(raw, location):
    '''
    :param raw:
    :param location:
    :return: A color with the raw rgb value of the location input
    '''
    color = Color()
    color.name = location.name

    # get the patch color with median or mean depending on the constant set
    if constants.USE_MEDIAN:
        color.rawrgb = np.median(
            raw[location.row - constants.OFFSET_COLOR_READ: location.row + constants.OFFSET_COLOR_READ,
            location.col - constants.OFFSET_COLOR_READ: location.col + constants.OFFSET_COLOR_READ, :], axis=(0, 1))
    else:
        color.rawrgb = np.mean(
            raw[location.row - constants.OFFSET_COLOR_READ: location.row + constants.OFFSET_COLOR_READ,
            location.col - constants.OFFSET_COLOR_READ: location.col + constants.OFFSET_COLOR_READ, :], axis=(0, 1))
    return color


def getRawColorsFromLocations(raw, locations):
    '''

    :param raw:
    :param locations:
    :return: a list of colors with the raw rgb values at those locations given
    '''
    colors = []
    for location in locations:
        color = getRawColorFromLocation(raw, location)
        colors.append(color)
    return colors


def getMacbethGTColors():
    '''
    Get a list of color objects that have the macbeth ground truth xyz values in them
    :return:
    '''

    colors = []
    # read in xyY values from txt file

    #this is the one using the online values and converting
    manual = ([[0.11644218, 0.09954022, 0.0478699 ],
               [0.38900256, 0.33967091, 0.18542485],
               [0.15787956, 0.17591065, 0.25386739],
               [0.11469681, 0.13542278, 0.05299748],
               [0.23339898, 0.22611887, 0.32697735],
               [0.30115175, 0.41152943, 0.34519922],
               [0.42445043, 0.320002  , 0.05187697],
               [0.10744258, 0.10509278, 0.28584131],
               [0.29891628, 0.19321757, 0.09680321],
               [0.08080766, 0.06220937, 0.10360798],
               [0.35834424, 0.43702746, 0.08602798],
               [0.49842155, 0.43741672, 0.0616632 ],
               [0.06047982, 0.05408537, 0.20649443],
               [0.14626046, 0.22369682, 0.0743768 ],
               [0.22174789, 0.13172814, 0.0383473 ],
               [0.6171809 , 0.60875684, 0.07382304],
               [0.29858722, 0.19118349, 0.21980306],
               [0.14490446, 0.1919056 , 0.29172445],
               [0.84551602, 0.88091621, 0.69397144],
               [0.56659294, 0.58975643, 0.48289671],
               [0.34942191, 0.36465036, 0.30137383],
               [0.18354127, 0.1905525 , 0.15667815],
               [0.08437654, 0.08809643, 0.07391229],
               [0.03038259, 0.03148901, 0.02656261]]) #* 1.2

    #from Vivian
    manual = np.array([[11.22,	10.04,	5.18],
              [37.94,	33.95,	18.07],
              [16.12,	17.6,	25.01],
              [10.68,	13,	5.14],
              [23.29,	22.09,	31.29],
              [30.86,	42.47,	34.52],
              [40.99,	31.58,	5.24],
              [11.77,	10.74,	29.26],
              [29.82,	19.5,	9.86],
              [8.11,	5.98,	10.08],
              [34.43,	43.78,	8.62],
              [47.46,	42.69,	5.85],
              [6.96,	5.81,	21],
              [15.27,	23.86,	7.86],
              [21.29,	12.66,	3.86],
              [58.67,	59.88,	7.05],
              [30.29,	19.69,	22.68],
              [12.91,	18.33,	29.09],
              [88,	92.14,	72.72],
              [57.16,	60.05,	48.5],
              [35.64,	37.46,	30.53],
              [18.37,	19.2,	15.4],
              [8.84,	9.31,	7.63],
              [2.89,	3.01,	2.49]]) * 0.01
    with open('extra/macbethxyY.txt', newline='') as csvfile:
        colorsReader = csv.reader(csvfile, delimiter=',')
        colors = []
        for i, row in enumerate(colorsReader):
            if (i == 0):  # skip comment line at top
                continue
            x = float(row[1])
            y = float(row[2])
            Y = float(row[3])
            X = x * Y / y
            Z = (1 - x - y) * Y / y

            #color.name = row[0]
            #color.xyz = np.array(
               # [X, Y, Z]) / 100  # divide by 100 because the xyz values our system expects should be scaled between 0-1
            #color.xyz =
            colors.append(manual[i-1])

    return colors


# borrowed from https://github.com/aehaynes/IRLS/blob/master/irls.py
def IRLS(y, X, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    n, p = X.shape
    delta = np.array(np.repeat(d, n)).reshape(1, n)
    w = np.repeat(1, n)
    W = np.diag(w)
    B = np.dot(np.linalg.inv(X.T.dot(W).dot(X)),
               (X.T.dot(W).dot(y)))
    for _ in range(maxiter):
        _B = B
        _w = abs(y - X.dot(B)).T
        w = float(1) / np.maximum(delta, _w)
        W = np.diag(w[0])
        B = np.dot(np.linalg.inv(X.T.dot(W).dot(X)),
                   (X.T.dot(W).dot(y)))
        tol = sum(abs(B - _B))
        if tol < tolerance:
            return B
    return B


def getScanwellColorChartGTColors(gtcsvfile):
    '''
    :param gtcsvfile:
    :return: a list of colors containing the xyz values and names for the scanwell reference colors
    '''
    gtChart = pd.read_csv(gtcsvfile)
    colors = []
    for i, row in gtChart.iterrows():
        color = Color()
        color.name = row["Color"]
        color.xyz = np.array([row["X"], row["Y"], row["Z"]])
        colors.append(color)

    return colors


def getRawToXYZMat(rawColors, xyzColors, patchScaleName):
    '''
    this method is the brains of what we ware doing
    It takes in two list of colors,
    the raw colors are the raw colors of the colors in the image
    the xyz colors are the "gt" colors of the colors - what we expect
    then we find a least squares fit to convert from raw to xyz based off these colors
    the color  names must correspond from rawColors to xyzColors or this will throw an error
    :param rawColors:the raw colors are the raw colors of the colors in the image
    :param xyzColors:the "gt" colors of the colors - what we expect the color  xyz values to be
    :param patchScaleName:the color  used in our optimization trick for the scaling - usually should be a middle gray
    :return:return a matrix that converts from raw to xyz based off a least squares fit, and the sum of the residuals(an error metric of the fit)
    '''

    # first make sure the same colors are being used

    # check that the same number of colors are in both lists

    rawColorNames = []
    for color in rawColors:
        rawColorNames.append(color.name)

    xyzColorNames = []
    for color in xyzColors:
        xyzColorNames.append(color.name)

    # make sure names are the same
    # this needs to throw an error
    if (sorted(rawColorNames) != sorted(xyzColorNames)):
        print("Raw Names don't match XYZ Names")

    # make sure the patch we are scaling with actually exists in the names
    if patchScaleName not in rawColorNames:
        print("Patch Scale Name not in Raw Color Names")

    # sort the colors so we have them in name order
    rawColors = sorted(rawColors, key=operator.attrgetter("name"))
    xyzColors = sorted(xyzColors, key=operator.attrgetter("name"))

    # create matrices so we can do the least squares fit with the numpy

    if constants.USE_BIAS:
        rawMat = np.empty((0, 4))
    else:
        rawMat = np.empty((0, 3))

    rawPatchForScale = None  # hold the patch that corresponds to patchScaleName

    # loop through rawcolors and build up a rawMat matrix for the least squares fit
    # and also get the patch that corresponds to the patchScaleName

    #badColors = ["S24","S23","S23_L", "S24_L"]
    badColors = []
    for color in rawColors:
        print("RAW", color.name)
        if color.name in badColors:
            continue
        if (color.name == patchScaleName):
            rawPatchForScale = color
        if constants.USE_BIAS:
            rawColorWithBias = np.concatenate((color.rawrgb, [constants.BIAS]))
            rawMat = np.concatenate((rawMat, np.array([rawColorWithBias])))
        else:
            rawMat = np.concatenate((rawMat, np.array([color.rawrgb])))

    # do the same thing for the xyzPatches
    xyzMat = np.empty((0, 3))
    xyzPatchForScale = None
    for color in xyzColors:
        if color.name in badColors:
            continue
        print("XYZ", color.name)
        if (color.name == patchScaleName):
            xyzPatchForScale = color
        xyzMat = np.concatenate((xyzMat, np.array([color.xyz])))

    # do optimization trick and least squares fit
    a = rawMat
    b = xyzMat
    # a = optimizationTrick(a,b)
    if(patchScaleName == "Xrite"):

        #FIRST one is pixel il 100
        #SECOND one is NOte 10 D65-500
        #THIRD one is iphone 8 D65-500

        #change Xrite XYZs
        b = np.array([[ 0.13065572,  0.12085978,  0.05803146],

                    [ 0.42447287,  0.38208907,  0.21949033],
                    [ 0.19410459,  0.2165001 ,  0.26491431],
                      [ 0.1319272 ,  0.15681769,  0.07332194],
                      [ 0.29349163,  0.28029753,  0.35822017],
                      [ 0.41694939,  0.54929541,  0.48717846],
                      [ 0.40793272,  0.32815731,  0.01823575],
                      [ 0.13451968,  0.13184239,  0.31044952],
                      [ 0.3365077 ,  0.21649124,  0.12401949],
                    [ 0.10671911,  0.07794863,  0.12206422],
                    [ 0.43193921,  0.54908563,  0.12248641],
                    [ 0.5555936 ,  0.5417931 , -0.00989852],
                    [ 0.08155504,  0.07970599,  0.20742224],
                    [ 0.2021791 ,  0.28356519,  0.1250361 ],
                    [ 0.25492682,  0.13791286,  0.06214416],
                    [ 0.66773795,  0.70489767,  0.03411445],
                    [ 0.36953953,  0.2356448 ,  0.28758261],
                    [ 0.21751237,  0.27287078,  0.46134006],
                    [ 0.96520589,  1.05133006,  0.71789596],
                    [ 0.63808902,  0.68764711,  0.53530196],
                    [ 0.40905305,  0.45212277,  0.31916692],
                    [ 0.22913482,  0.24794905,  0.18195884],
                    [ 0.11800105,  0.13017148,  0.08287937],
                    [ 0.05015346,  0.05597906,  0.0454625 ],])

        b += np.array([[ 0.1461307 ,  0.13211464,  0.07641379],
        [ 0.47368184,  0.42346038,  0.23944776],
        [ 0.22428294,  0.24024138,  0.33597861],
                       [ 0.14483792,  0.17193905,  0.07967691],
                       [ 0.30816378,  0.29490257,  0.40246575],
                       [ 0.41364221,  0.5436048 ,  0.45675487],
                       [ 0.47117553,  0.35763427,  0.05730084],
                       [ 0.15888288,  0.14643081,  0.38015554],
                       [ 0.37359239,  0.24349686,  0.12683112],
                       [ 0.11167878,  0.08657316,  0.13104464],
                       [ 0.43631748,  0.54918824,  0.11511273],
                       [ 0.55344016,  0.49913372,  0.05974678],
                       [ 0.09674214,  0.08558581,  0.2583138 ],
                       [ 0.21236443,  0.31194168,  0.12231094],
                       [ 0.27245677,  0.15813283,  0.05692772],
                       [ 0.70037822,  0.71358983,  0.08219364],
                       [ 0.37711982,  0.24783656,  0.27515062],
                       [ 0.20911172,  0.26609383,  0.40940397],
                       [ 1.02464726,  1.07037884,  0.87864917],

                       [ 0.68934824,  0.7259345 ,  0.59190298],
                       [ 0.43866827,  0.46044204,  0.38541559],
                       [ 0.2325516 ,  0.24478998,  0.20534051],
                       [ 0.11910839,  0.12797293,  0.1053153 ],
                       [ 0.04975852,  0.05061882,  0.04452307]])

        b = b/2
        b += 0.02



    if constants.USE_IRLS_SOLVER:
        y = b
        x = a
        B0 = IRLS(y[:, 0], x, 50)
        B1 = IRLS(y[:, 1], x, 50)
        B2 = IRLS(y[:, 2], x, 50)

        rawToXYZMat = np.array([B0, B1, B2])
        rawToXYZMat = rawToXYZMat.T
    else:
        # get fit and residuals
        # residuals = Sums of squared residuals: Squared Euclidean 2-norm for each column in b - a @ x
        # residuals is the sum of squared residuals for X, Y, Z respectively  - 3x1 vector because
        # we have a column of residuals for each coordinate X, Y, and Z
        rawToXYZMat, residuals, rank, _ = np.linalg.lstsq(a, b, rcond=None)

    residuals = np.square(np.linalg.norm(b - np.dot(a, rawToXYZMat), axis=0))
    percentErrors = np.linalg.norm(b - np.dot(a, rawToXYZMat), axis = 1) / np.linalg.norm(b, axis = 1)

    fitError = np.mean(percentErrors) * 100
    #print(percentError)

    #fitError = np.sum(residuals)  # add up the residuals given by the lst sq fit for a simple error metric

    # calculate scaling Factor for the matrix
    # xyzPatchBeforeScale = rawToXYZ(rawPatchForScale.rawrgb, rawToXYZMat)
    # scaleFactor = np.linalg.norm(xyzPatchForScale.xyz)/np.linalg.norm(xyzPatchBeforeScale)

    # scale the matrix
    # rawToXYZMat = scaleFactor * rawToXYZMat

    return rawToXYZMat, fitError


def setCorrectedColors(colors, rawToXYZMat):
    '''
    Function to take in list of colors with raw colors and the conversion matrix
    Then the function gets the xyz, rgb, and labb values
    :param colors: list of colors with raw colors
    :param rawToXYZMat: conversion matrix for this raw image
    :return colors: colors with new color information
    '''
    for color in colors:
        # this should be moved into a setter somehow so the rgb and lab come with the xyz
        if constants.USE_BIAS:
            rawColorWithBias = np.concatenate((color.rawrgb, [constants.BIAS]))
            color.xyz = rawToXYZ(rawColorWithBias, rawToXYZMat)
        else:
            color.xyz = rawToXYZ(color.rawrgb, rawToXYZMat)
        color.rgb = xyzTosRGB(color.xyz)
        color.lab = sRGBToLab(color.rgb)
    return colors


def outputColors(outFileName, colors, gtScanwellColorChartCSV=None):
    '''
    Output color information to file
    :param outFileName: file to output to
    :param colors: list of colors
    :param gtScanwellColorChartCSV: path to ground truth csv for scanwell color chart
    :return:
    '''
    with open(outFileName, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["Color", "rR", "rG", "rB", "X", "Y", "Z", "R", "G", "B", "L", "a", "b"])
        for color in colors:
            combinedRow = np.concatenate([color.rawrgb, color.xyz, color.rgb, color.lab]).tolist()
            csvwriter.writerow([color.name] + combinedRow)

        if gtScanwellColorChartCSV is not None:
            csvwriter.writerow([])  # new line
            csvwriter.writerow(["Corrected with ", gtScanwellColorChartCSV])
