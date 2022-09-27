import numpy as np
import matplotlib.pyplot as plt
import WaveletTransform.WaveletTransformer as wt
import Tree.BesovTree as bt
import math
from pydub import AudioSegment
from tempfile import mktemp
from scipy.io import wavfile


def sinus_plot():
    t = np.linspace(0, 2 * np.pi, 2 ** 9)
    # y = np.sin(4*t)+np.cos(3*t) + np.random.random(2**9)
    y = np.sin(t)
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"Plot of sin(4t)+cos(3t) with noise level 0.3\nDifferent values for beta")
    hsm, wave_coef = wt.getWaveletCoefficients(y, "db4")
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for idx, beta in enumerate(beta_values):
        besov_tree = bt.BesovTree(wave_coef, beta)
        transform_coeff = besov_tree.getMinimizingPosteriorCoefficients()
        transform_y = wt.inverseDWT((hsm, transform_coeff), "haar")
        axs[math.floor(idx / 3), idx % 3].set_title("Beta = " + str(beta))
        axs[math.floor(idx / 3), idx % 3].plot(t, y, label="Original")
        axs[math.floor(idx / 3), idx % 3].plot(t, transform_y[0:len(y)], label=f"Haar wavelet transform beta={beta}")
    plt.legend()
    plt.show()

def analyse_different_values_of_beta():
    t = np.linspace(0, 2 * np.pi, 2 ** 9)

    y_funcs = [np.sin(t), np.cos(t), np.sin(4*t)+np.sin(3*t),
               np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.3,size=2**9)
               ,np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.6,size=2**9),
               np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.9,size=2**9)]
    y_names = ["sin(t)","cos(t)","sin(4t)+sin(3t)","sin(4t)+cos(3t) + 0.3*N(0,1)","sin(4t)+cos(3t) + 0.6*N(0,1)","sin(4t)+cos(3t) + 0.9*N(0,1)"]

    for y,y_name in zip(y_funcs,y_names):
        beta_vals = np.linspace(0.01,0.99,100)
        number_of_zeroes = []
        hsm, wave_coef = wt.getWaveletCoefficients(y, "haar")
        for beta in beta_vals:
            besov_tree = bt.BesovTree(wave_coef, beta)
            transform_coeff = besov_tree.getMinimizingPosteriorCoefficients()
            number_of_zeroes.append(len([x for x in transform_coeff.values() if x == 0]) / 2**9)
        plt.plot(beta_vals,number_of_zeroes,label=y_name)
    plt.title("Haar-based Besov trees: percentage of zero coefficients\nin the wavelet coefficients as a function of beta")
    plt.ylabel("Percentage of zeroe coefficients")
    plt.xlabel("Beta")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    mp3_audio = AudioSegment.from_file('how_you_doing.mp3', format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file
    length = data.shape[0] / FS
    time = np.linspace(0., length, data.shape[0])

    data_w_noise = data + np.random.normal(0,1,len(data))
    # left_signal = data[:,0]
    beta_vals = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.9]
    for beta in beta_vals:
        hsm, sound_waveletcoeffs = wt.getWaveletCoefficients(data_w_noise,"haar")
        besov_tree = bt.BesovTree(sound_waveletcoeffs,beta)
        transform_coeff = besov_tree.getMinimizingPosteriorCoefficients()
        transformed_signal = wt.inverseDWT((hsm, transform_coeff), "haar")
        transformed_signal_cut = transformed_signal[0:len(data)]
        print(transformed_signal_cut)

        print(sum( (data - transformed_signal_cut)**2 ))
        wavfile.write(f"how_you_doing_transformed_beta_{beta}.wav",FS, np.asarray(transformed_signal_cut))