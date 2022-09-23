from unittest import TestCase
import pywt

from WaveletTransform.WaveletTransformer import getWaveletCoefficients


class Test(TestCase):
    def test_get_wavelet_daubechy(self):
        signal = [1,1,1,1,1,1]
        [hsm, hde] = pywt.dwt(signal, 'db4', 'smooth')
        # [hsm, hde] = getWaveletCoefficients(signal, 'db1')
        print(f"hsm= {hsm}\nhde={hde}")
        self.fail()
