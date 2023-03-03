# let's first just do naive wavelet decomposition
# on the first 1000 images of 'data/test/Real' and 'data/test/DDPM'

import os
import numpy as np
from PIL import Image
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

if __name__ == "__main__":
    n = 500
    # wavelets=["haar","db2","db4","coif1","coif2","sym2", "sym4"]
    wavelets = ["haar"]
    reals = countZeroesInDir("../../data/test/Real",n,wavelets)
    ddpm = countZeroesInDir("../../data/test/DDPM",n,wavelets)
    # ldm = countZeroesInDir("../../data/test/LDM", n, wavelets)
    # adm = countZeroesInDir("../../data/test/ADM", n, wavelets)
    nRows = 2
    nCols = 4
    fig, axs = plt.subplots(nRows,nCols,figsize=(20,10))
    for i in range(0, len(wavelets)):
        axs[i//nCols][i%nCols].hist(reals[wavelets[i]], bins=20, alpha=0.2, label="Real")
        axs[i//nCols][i%nCols].hist(ddpm[wavelets[i]], bins=20, alpha=0.2, label="DDPM")
        # axs[i // nCols][i % nCols].plot(ldm[wavelets[i]], 'o', alpha=0.2, label="LDM")
        # axs[i // nCols][i % nCols].hist(adm[wavelets[i]], bins=20, alpha=0.2, label="ADM")
        axs[i//nCols][i%nCols].set_title(wavelets[i])
        axs[i//nCols][i%nCols].legend(loc='upper right')

    fig.suptitle(f"Percentage of zeroes in wavelet coefficients of {n} LSUN bedroom images")
    fig.supxlabel("Percentage of zeroes")
    fig.supylabel("Number of images")
    fig.show()
    # fig.savefig("naiveWaveletDecompScatter.png")