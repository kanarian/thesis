from PIL import Image
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovForest as tbf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Opening the "cloud_dalle.png" image (1024x1024 dimension)
    thisImage = Image.open("cloud_dalle.png")
    # We want to crop out the middle (512x512) section of the image, because some images might not be
    (new_width, new_height) = (512, 512)
    width, height = thisImage.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    thisImage = thisImage.crop((left, top, right, bottom))


    # Convert to grayscale and normalize pixel values from range [0,255] -> [0,1]
    thisArray = np.asarray(thisImage.convert("L")) / 255

    # add some noise if you want

    thisArray = thisArray + np.random.normal(0, 0.1, thisArray.shape)
    # reconImage = Image.fromarray(thisArray * 255).show()


    # We need to get the wavelet decomposition of our normalized image
    # the function returns a smooth part and a detail part
    # for our approach, only the detail part matters
    # NOTE: For any non-Haar wavelet, we need to choose periodic "per" boundary continuation mode
    # otherwise the dimensions wont work.
    wavelet = "haar" # others are possible such as db2 https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
    mode = "smooth"

    hsm, hde = wt2d.get2DWaveletCoefficients(thisArray, wavelet, mode)

    # beta values need to be very close to 0.5 but just under it.
    beta = 0.49999
    # max_level is 8 for an image of size 512 by 512. (2^(8+1) = 512)
    # start_level refers to at which level the tree algorithm should start running
    besovForest = tbf.TwoDimBesovForest(hde, beta, max_level=8, start_level=4)

    # running the algorithm to get the new coefficients according to the algorithm
    newCoeffs = besovForest.getMinimizingPosteriorCoefficients()

    reconArray = wt2d.inverse2DDWT((hsm, newCoeffs), wavelet, mode)
    reconImage = Image.fromarray(reconArray*255)
    reconImage.show()


