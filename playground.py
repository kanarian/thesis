import pywt
import numpy as np

x = np.arange(16*16*16*16).reshape((16*16, 16*16))
wave_dec = pywt.wavedec2(x, 'haar',level=2,mode="per")
hsm = wave_dec[0]
wave_coef = wave_dec[1:]
for idx, el in enumerate(wave_coef):
    print(f"Level {idx} horizontal differences {el[0].shape}")
    print(f"Level {idx} vertical differences {el[1].shape}")
    print(f"Level {idx} diagonal differences {el[2].shape}")