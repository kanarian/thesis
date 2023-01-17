import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import cm
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

def generateRandomPixelValsTwoClasses(n):
    arrHighVals = np.random.randint(
        low=200,
        high=256,
        size=(n, 1, 2,),
        dtype=np.uint8
    )
    arrLowVals = np.random.randint(
        low=0,
        high=56,
        size=(n, 1, 2,),
        dtype=np.uint8
    )
    return arrLowVals, arrHighVals

low, high = generateRandomPixelValsTwoClasses(40)

fig, ax = plt.subplots()
ax.plot(low[:,0,0], low[:,0,1], 'o', color="black", fillstyle='none',label='Low')
ax.plot(high[:,0,0], high[:,0,1], 'o', color='black', label='High')

def ImgBoxPoint(point, xyBoxvals=(10,-90)):
    ex = Image.fromarray(point)
    imgBox = OffsetImage(ex, zoom=10,norm=plt.Normalize(0,255), cmap="Greys")
    imgBox.image.axes = ax
    ab = AnnotationBbox(imgBox, point[0],
                        xybox=xyBoxvals,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )
    ax.add_artist(ab)

ImgBoxPoint(high[0])
ImgBoxPoint(high[10], xyBoxvals=(10,-80))
ImgBoxPoint(low[0], xyBoxvals=(10,30))
ImgBoxPoint(low[10], xyBoxvals=(10,20))
ax.set_title("Plot of grayscale two-pixel values as points.")
ax.set_xlabel("Pixel value 1")
ax.set_ylabel("Pixel value 2")
fig.show()