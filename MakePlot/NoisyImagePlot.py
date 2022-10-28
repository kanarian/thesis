from PIL import Image
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovForest as tbt
import matplotlib.pyplot as plt


def getNonZeroCoefficients(coeffDict):
    sum = 0
    for coeff in coeffDict.values():
        if coeff["cH"] > 0 or coeff["cV"] > 0 or coeff["cD"] > 0:
            sum += 1
    return sum

def calcMSE(img1, img2):
    return np.mean((img1 - img2) ** 2)

def imageToArray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img)/255

def arrayToImage(arr):
    return Image.fromarray(arr*255)

def findMSEBetaValue(beta, wt, hsm, y, wavelet, mode, saveImg=False, savePath=None):
    tree = tbt.TwoDimBesovForest(wt, beta, 6)
    new_coeffs = tree.getMinimizingPosteriorCoefficients()
    new_img = wt2d.inverse2DDWT((hsm, new_coeffs), wavelet, mode)
    mse = calcMSE(y, new_img)
    if saveImg:
        arrayToImage(new_img).convert("L").save(savePath)
    return mse

def analysis(beta_values, img_path, wavelet, mode):
    y = imageToArray(img_path)
    sd = 0.1
    y_noise = y + np.random.normal(0, sd, y.shape)
    Image.fromarray(y_noise*255).convert("L").show("Noisy Image")
    # Image.fromarray(y_noise*255).convert("L").save(f"../plots/NoisyKoala0.01/y_noise_{img_path.split('/')[-1]}")
    hsm, wt = wt2d.get2DWaveletCoefficients(y_noise, wavelet, mode)
    mses = []
    for beta in beta_values:
        mse = findMSEBetaValue(beta, wt, hsm, y,wavelet, mode, saveImg=False, savePath=f"../plots/NoisyKoala{sd}/y_recon_beta_{wavelet}_{mode}_{beta}_{img_path.split('/')[-1]}")
        mses.append(mse)
    plt.plot(beta_values, mses)
    plt.title(f"MSE for different beta values on a noisy version of {img_path}")
    plt.xlabel("Beta")
    plt.ylabel("MSE")
    plt.show()

if __name__ == "__main__":
    analysis([0.1,0.2,0.3,0.4,0.45,0.475,0.49,0.495,0.51,0.55,0.60], "../Koala.jpg", "db2", "per")