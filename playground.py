import pywt
import numpy as np

x = np.arange(16*16).reshape((16, 16))
wave_dec = pywt.wavedec2(x, 'db2')
hsm = wave_dec[0]
wave_coef = wave_dec[1:]
print(len(hsm))
print(hsm)