from PIL import Image
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovTree as tbt
import matplotlib.pyplot as plt
import cProfile
import pstats

def produceTree(wavelet="haar"):
    for i in range(0,1):
        img = Image.open("../Koala.jpg").convert("L")
        a = np.asarray(img)/255
        hsm, wt = wt2d.get2DWaveletCoefficients(a, wavelet)
        beta = 0.49
        tree = tbt.TwoDimBesovTree(wt, beta, 8)
        new_coeffs = tree.getMinimizingPosteriorCoefficients()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        produceTree("db1")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats("profile.prof")

