import math
import numpy as np
import pywt

def getWaveletCoefficients(signal,wavelet):
    # We zeropad the signal before getting the coefficients
    if math.log2(len(signal)) != int(math.log2(len(signal))):
        next_power_of_two = int(math.log2(len(signal))) + 1
        signal = np.append(signal, np.zeros((2 ** next_power_of_two - len(signal))))
    wave_dec = pywt.wavedec(signal, wavelet)
    print("hsm",len(wave_dec))
    for i, _ in enumerate(wave_dec[1:]):
        print(f"detail_coeffs lvl {i} = {len(_)}")
    hsm = wave_dec[0]
    wave_dct = {}
    for idx_one, array_one in enumerate(wave_dec[1:]):
        for idx_two, el_two in enumerate(array_one):
            wave_dct[(idx_one, idx_two)] = el_two
    return hsm, wave_dct

def inverseDWT(coeffs,wavelet):
    hsm, wave_dict = coeffs
    wave_levels = []
    number_of_levels = int(math.log2(len(wave_dict) + 1))
    for level_number in range(0, number_of_levels):
        level_number_of_elements = 2 ** level_number
        level = [None]*level_number_of_elements
        wave_levels.append(level)
    for (key, el) in wave_dict:
        wave_levels[key][el] = wave_dict[(key, el)]
    wave_levels = [np.array(wave_level) for wave_level in wave_levels]
    to_reconstruct = [hsm, *wave_levels]
    return pywt.waverec(to_reconstruct, wavelet)
