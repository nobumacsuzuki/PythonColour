import numpy as np
import cv2
import colour

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# it requires colour https://github.com/colour-science/colour
# from anaconda prompt
# conda install -c conda-forge colour-science

filename = "lena_512x512.bmp"
arrayImageColorBGR = cv2.imread(filename, cv2.IMREAD_COLOR) 
arrayImageColorRGB = cv2.cvtColor(arrayImageColorBGR, cv2.COLOR_BGR2RGB)
imageHeight = arrayImageColorRGB.shape[0]
imageWidth = arrayImageColorRGB.shape[1]
arrayImageColorRGB = arrayImageColorRGB.reshape(imageHeight * imageWidth, 3)
arrayImageColorfRGB = arrayImageColorRGB.astype(np.float)
arrayImageColorfRGB = arrayImageColorfRGB / 255.0

illuminant_RGB = np.array([0.31270, 0.32900])
illuminant_XYZ = np.array([0.34570, 0.35850])
chromatic_adaptation_transform = 'Bradford'
RGB_to_XYZ_matrix = np.array(
    [[0.41240000, 0.35760000, 0.18050000],
    [0.21260000, 0.71520000, 0.07220000],
    [0.01930000, 0.11920000, 0.95050000]]
)
arrayImageXYZ = np.zeros((imageHeight * imageWidth, 3), dtype=float)

#arrayImageColorfRGB = np.array(
#    [[1., 0., 0.],
#    [0., 1., 0.],
#    [0., 0., 1.],
#    [0.5, 0., 0.],
#    [0., 0.5, 0.],
#    [0., 0., 0.5],
#    [1.,  1., 0.],
#    [0., 1., 1.],
#    [1., 0., 1.],
#    [0.5, 0.5, 0.],
#    [0., 0.5, 0.5],
#    [0.5, 0., 0.5],
#    [1., 1., 1.]]
#)
#arrayImageXYZ = np.zeros((13, 3), dtype=float)
index = 0

for fRGB in arrayImageColorfRGB:
    fRGB = colour.cctf_decoding(fRGB, function = 'sRGB')
    arrayImageXYZ[index] = colour.RGB_to_XYZ(fRGB, illuminant_RGB, illuminant_XYZ, RGB_to_XYZ_matrix, chromatic_adaptation_transform)  
    if (index % 10000 == 0):
        print(index)
    index += 1

fig = plt.figure(facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
ax.w_zaxis.set_pane_color((0., 0., 0., 0.))

index = 0
for (X, Y, Z) in arrayImageXYZ[::1000]:
    pointColor = arrayImageColorfRGB[index * 1000]
    ax.scatter(X, Y, Z, color = pointColor)
    if (index % 10 == 0):
        print(index)
    index += 1
plt.show()

