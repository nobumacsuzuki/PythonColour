import numpy as np
import cv2
import colour

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# it requires colour https://github.com/colour-science/colour
# from anaconda prompt, run
# conda install -c conda-forge colour-science

def PlotXYZ(arrayImageColorfRGB):
    # arrayImageColorfRGB is normalized one dimentional RGB data
    # [[R0, G0, B0], ... [RN-1, GN-1, BN-1]]
    # either R, G or B is equal to or grater than 0.0, and equal to or less than 1.0
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])
    chromatic_adaptation_transform = 'Bradford'
    RGB_to_XYZ_matrix = np.array(
        [[0.41240000, 0.35760000, 0.18050000],
        [0.21260000, 0.71520000, 0.07220000],
        [0.01930000, 0.11920000, 0.95050000]]
    )
    arrayImageXYZ = np.zeros((arrayImageColorfRGB.shape[0], 3), dtype=float)

    for (index, fRGB) in enumerate(arrayImageColorfRGB):
        fRGB = colour.cctf_decoding(fRGB, function = 'sRGB')
        arrayImageXYZ[index] = colour.RGB_to_XYZ(fRGB, illuminant_RGB, illuminant_XYZ, RGB_to_XYZ_matrix, chromatic_adaptation_transform)  
        if (index % 10 == 0):
            print(index)

    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color((0., 0., 0., 0.))

    for (index, (X, Y, Z)) in enumerate(arrayImageXYZ):
        pointColor = arrayImageColorfRGB[index]
        ax.scatter(X, Y, Z, color = pointColor)
        if (index % 10 == 0):
            print(index)
    plt.show()

def PlotXYZAndsXsY(arrayImageColorfRGB):
    # arrayImageColorfRGB is normalized one dimentional RGB data
    # [[R0, G0, B0], ... [RN-1, GN-1, BN-1]]
    # either R, G or B is equal to or grater than 0.0, and equal to or less than 1.0
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])
    chromatic_adaptation_transform = 'Bradford'
    RGB_to_XYZ_matrix = np.array(
        [[0.41240000, 0.35760000, 0.18050000],
        [0.21260000, 0.71520000, 0.07220000],
        [0.01930000, 0.11920000, 0.95050000]]
    )
    arrayImageXYZ = np.zeros((arrayImageColorfRGB.shape[0], 3), dtype=float)

    for (index, fRGB) in enumerate(arrayImageColorfRGB):
        fRGB = colour.cctf_decoding(fRGB, function = 'sRGB')
        arrayImageXYZ[index] = colour.RGB_to_XYZ(fRGB, illuminant_RGB, illuminant_XYZ, RGB_to_XYZ_matrix, chromatic_adaptation_transform)  
        if (index % 10 == 0):
            print(index)

    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color((0., 0., 0., 0.))


    for (index, (X, Y, Z)) in enumerate(arrayImageXYZ):
        pointColor = arrayImageColorfRGB[index]
        ax.scatter(X, Y, Z, color = pointColor)
        xs = X / (X + Y + Z)
        ys = Y / (X + Y + Z)
        zs = 1 - xs - ys
        ax.scatter(xs, ys, zs, color = pointColor)
        if (index % 10 == 0):
            print(index)
    X = np.arange(0, 2, 1)
    Y = np.arange(0, 2, 1)
    X, Y = np.meshgrid(X, Y)
    Z = 1 -X -Y
    #ax.plot_surface(X, Y, Z)
    plt.show()

def ConvertRGB2fRGB(arrayImageColor, isBGR):
    if (isBGR):
        arrayImageColorRGB = cv2.cvtColor(arrayImageColor, cv2.COLOR_BGR2RGB)
    else:
        arrayImageColorRGB = arrayImageColor
    print(arrayImageColorRGB.ndim)
    if (arrayImageColorRGB.ndim == 3):
        imageHeight = arrayImageColorRGB.shape[0]
        imageWidth = arrayImageColorRGB.shape[1]
        arrayImageColorRGB = arrayImageColorRGB.reshape(imageHeight * imageWidth, 3)
    arrayImageColorfRGB = arrayImageColorRGB.astype(np.float)
    arrayImageColorfRGB = arrayImageColorfRGB / 255.0
    return arrayImageColorfRGB

def main():
    # this is a test array for primaries, sub-primatries etc...
    arrayImageColorRGB = np.array(
        [[255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255,  255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 255]]
    )
    PlotXYZAndsXsY(ConvertRGB2fRGB(arrayImageColorRGB, isBGR = False))

    # visualize Lena image in CIE-XYZ
    #filename = "lena_512x512.bmp"
    #arrayImageColorBGR = cv2.imread(filename, cv2.IMREAD_COLOR)
    #PlotXYZ(ConvertRGB2fRGB(arrayImageColorBGR, isBGR = True)[::2000]) # take 512 x 512 / 2000, 131 points

if __name__ == "__main__":
    main()
