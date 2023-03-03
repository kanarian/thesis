# let's first just do naive wavelet decomposition
# on the first 1000 images of 'data/test/Real' and 'data/test/DDPM'

import os
import numpy as np
from PIL import Image, ImageDraw
import pywt
import matplotlib.pyplot as plt

# read all filenames from a directory
def readFileNamesFromDir(dir):
    files = []
    for filename in os.listdir(dir):
        files.append(os.path.join(dir, filename))
    return files

def fileToImgArr(file):
    imgArr = np.asarray(Image.open(file).convert("L"))/255
    return imgArr

def waveletDecomp(imgArr, wavelet="haar"):
    coeffs = pywt.wavedec2(imgArr, wavelet, mode="periodic")
    return coeffs

def naiveWaveletDecomp(files, wavelets=["haar"]):
    res = {wavelet: [] for wavelet in wavelets}
    for file in files:
        imgArr = fileToImgArr(file)
        for wavelet in wavelets:
            coeffs = waveletDecomp(imgArr, wavelet)
            res[wavelet].append(coeffs)
    return res

def countZeroes(coeffs):
    count = 0
    totalLength = 0
    cA = coeffs[0]
    cD = coeffs[1:]
    for coeff in cD:
        totalLength += coeff[0].size*3
        count += np.count_nonzero(coeff[0])+np.count_nonzero(coeff[1])+np.count_nonzero(coeff[2])
    percentageZeroes = (totalLength-count)/totalLength
    return count, totalLength, percentageZeroes

def countZeroesInDir(dir, numFiles=10, wavelets=["haar"]):
    files = readFileNamesFromDir(dir)
    files = files[:numFiles]
    allCoefs = naiveWaveletDecomp(files, wavelets)
    zeroes = {wavelet: [] for wavelet in wavelets}
    for wavelet in wavelets:
        for coeffs in allCoefs[wavelet]:
            count, totalLength, percentageZeroes = countZeroes(coeffs)
            zeroes[wavelet].append(percentageZeroes)
    return zeroes

def findFileMostZeroes(dir, numFiles=10, wavelets=["haar"]):
    files = readFileNamesFromDir(dir)
    files = files[:numFiles]
    allCoefs = naiveWaveletDecomp(files, wavelets)
    d = {}
    for wavelet in wavelets:
        for idx, coeffs in enumerate(allCoefs[wavelet]):
            count, totalLength, percentageZeroes = countZeroes(coeffs)
            d[files[idx]] = percentageZeroes
    return d


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == "__main__":
    n = 500
    wavelets = ["db2"]
    reals = findFileMostZeroes("../../data/test/Real",n,wavelets)
    # ddpm = findFileMostZeroes("../../data/test/DDPM", n, wavelets)
    sortedReals = [(k,v) for k, v in sorted(reals.items(), key=lambda item: item[1])]
    # sortedDDPM = [(k,v) for k, v in sorted(ddpm.items(), key=lambda item: item[1])]
    z = []
    for (img, val) in sortedReals[0:9]:
        thisImg = Image.open(img)
        ImageDraw.Draw(thisImg).text((4, 0),f"{img.replace('../../data/test/','')}\n- percentageOfZeros: {val : .2f}",(0,0,0))
        z.append(thisImg)
    image_grid(z,3,3).show()