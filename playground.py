import math
import pywt

low_pass = pywt.Wavelet('db2').dec_lo
print(low_pass)
x = [1,2,3,4,5,6,7,8]
# x = [0,1,0,0,0,1,0,0,0]

def convolve(x, h):
    y = [0] * len(x)
    for n in range(len(x)):
        print("n = ", n)
        for k in range(len(h)):
            print("k = ", k)
            print("x[n-k] = ", x[n-k])
            if n-k >= 0:
                y[n] += h[k] * x[n-k]
            # y[n] += x[n-k] * h[k]
    return y

def convolve_zero_padded(x, h):
    y = [0] * (len(x)+len(h)-1)
    for n in range(len(x)+len(h)-1):
        for k in range(len(h)):
            if n-k >= 0 and n-k < len(x):
                y[n] += h[k] * x[n-k]
            elif n-k >= len(x):
                y[n] += h[k] * 0
    return y

def convolve_circular(x, h):
    y = [0] * len(x)
    for n in range(len(x)):
        for k in range(len(h)):
            y[n] += h[k] * x[(n-k) % len(x)]
    return y

# print(f"convolved circular {convolve_circular(x, low_pass)}")
# self_convolved = convolve_zero_padded(x, low_pass)
# print("self convoled ",[float(f"{el: .2f}") for el in self_convolved])
# print("self convolved, even indexed ", self_convolved[::2])
# print("self convolved, odd indexed ", self_convolved[1::2])

def upsample_zeropad_matlab_esque(sm,de):
    recon_sgnl = 2*(len(sm) - 2 + 1)*[0]
    print(f"recon_sgnl {recon_sgnl} has length {len(recon_sgnl)}")


    # recon_sm = [0] * (len(sm) * 2)
    # recon_sm[::2] = sm
    # print("recon, when sm is inserted ", recon_sm)
    # recon_sm = convolve_zero_padded(recon_sm,pywt.Wavelet('db2').dec_lo)
    # print("smoothed upsampled ", recon_sm)
    # recon_de = [0] * (len(de) * 2)
    # recon_de[::2] = de
    # recon_de = convolve_zero_padded(recon_de, pywt.Wavelet('db2').dec_hi)
    # print("detail upsampled ", recon_de)

    # return [a+b for a,b in zip(recon_sm, recon_de)]

mode = "zero"
cA, cD = pywt.dwt(x,"db2",mode=mode)
# print("from pywt cA ",cA)
# print("from pywt cD ",cD)
# reconstructed = pywt.idwt(cA,cD,"db2",mode=mode)
# print("reconstructed ",reconstructed)
# modes = pywt.Modes.modes

upsampled = upsample_zeropad_matlab_esque(cA, cD)
# print("upsampled ", upsampled)

upsample_zeropad_matlab_esque([1,1,1,1,1],[0,0,0,0,0])

# db2_wavelet = pywt.Wavelet('db2')
# print(db2_wavelet.dec_lo)
# print(db2_wavelet.dec_hi)
# print(db2_wavelet.rec_lo)
# print(db2_wavelet.rec_hi)