import pywt
import numpy as np
import math

def get2DWaveletCoefficients(signal, wavelet="haar", mode="smooth"):
    wavedec2 = pywt.wavedec2(signal, wavelet, mode=mode)
    hsm = wavedec2[0]
    wave_dec= wavedec2[1:]
    wave_dct = {}
    for idx_one, array_one in enumerate(wave_dec):
        [cH, cV, cD] = list(map(lambda x: x.flatten(), array_one))
        for idx_two, (this_cH, this_cV, this_cD) in enumerate(zip(cH, cV, cD)):
            wave_dct[(idx_one, idx_two)] = {"cH": this_cH, "cV": this_cV, "cD": this_cD}
    return hsm, wave_dct

def inverse2DDWT(coeffs, wavelet="haar", mode="smooth"):
    hsm, wave_dict = coeffs
    wave_levels = []
    number_of_roots = len(hsm.flatten())
    dim_of_root = hsm.shape[0]
    nodes_per_tree = (len(wave_dict) + number_of_roots) / (number_of_roots)
    # The number of nodes in a quad-tree of depth n is:
    # 1 + 4 + 4^2 + ... + 4^n = ( 4^(n+1) - 1 ) / 3
    # Thus the number of levels is log_4(number_of_nodes + 1)
    number_of_levels = int(math.log((3 * nodes_per_tree + 1), 4))

    # This sort step is actually not needed, as the wavelet coefficients are in the correct order
    # but we do is just to be safe
    sorted_wavedict = sorted(list(wave_dict.items()), key=lambda x: (x[0][0], x[0][1]))
    detail_levels = []
    prevStartIndex = 0
    for i in range(0, number_of_levels):
        startIndexThisLevel = prevStartIndex
        endIndexThisLevel = startIndexThisLevel + number_of_roots*4**i

        thisLevel = sorted_wavedict[startIndexThisLevel:endIndexThisLevel]
        thisLevelDetail_cH = np.fromiter(map(lambda x: x[1]['cH'], thisLevel),dtype=float).reshape(dim_of_root*2**i, dim_of_root*2**i)
        thisLevelDetail_cV = np.fromiter(map(lambda x: x[1]['cV'], thisLevel),dtype=float).reshape(dim_of_root*2**i, dim_of_root*2**i)
        thisLevelDetail_cD = np.fromiter(map(lambda x: x[1]['cD'], thisLevel),dtype=float).reshape(dim_of_root*2**i, dim_of_root*2**i)
        detail_level = ([thisLevelDetail_cH, thisLevelDetail_cV, thisLevelDetail_cD])
        detail_levels.append(detail_level)
        prevStartIndex = endIndexThisLevel

    wave_levels = detail_levels
    to_reconstruct = [hsm, *wave_levels]
    return pywt.waverec2(to_reconstruct, wavelet, mode=mode)


# hsm ,wave_dct = get2DWaveletCoefficients(np.arange(8*8).reshape((8, 8)), "haar", "smooth")
# inverse_signal = inverse2DDWT((hsm, wave_dct), "haar", "smooth")
# print("inverse signal", inverse_signal)