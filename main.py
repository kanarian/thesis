import numpy as np
import matplotlib.pyplot as plt
import WaveletTransform.WaveletTransformer as wt
import Tree.BesovForest as bf
import math
from MakePlot.MakeTreePlot import makePlotBasedOnWaveletCoefficients
import heapq

def sinus_plot(wavelet="haar", mode="smooth"):
    t = np.linspace(0, 2 * np.pi, 2 ** 9)
    # y = np.sin(4*t)+np.cos(3*t) + np.random.random(2**9)
    y = np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.3,size=2**9)
    # y = np.sin(t)
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"Besov Tree estimates based on {wavelet} wavelets plot of sin(4t)+cos(3t) with noise level 0.3\nDifferent values for beta")
    hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet, mode)
    beta_values = [0.00000000001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # beta_values = [0.0000000001]
    for idx, beta in enumerate(beta_values):
        besov_forest = bf.BesovForest(wave_coef, beta)
        transform_coeff = besov_forest.getMinimizingPosteriorCoefficients()
        transform_y = wt.inverseDWT((hsm, transform_coeff), wavelet,mode)
        axs[math.floor(idx / 3), idx % 3].set_title("Beta = " + str(beta))
        axs[math.floor(idx / 3), idx % 3].plot(t, y, label="Original")
        axs[math.floor(idx / 3), idx % 3].plot(t, transform_y[0:len(y)], label=f"Haar wavelet transform beta={beta}")
    plt.legend()
    plt.show()

def analyse_different_values_of_beta(wavelet="haar", mode="smooth"):
    t = np.linspace(0, 2 * np.pi, 2 ** 9)

    y_funcs = [np.sin(t), np.cos(t), np.sin(4*t)+np.sin(3*t),
               np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.3,size=2**9)
               ,np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.6,size=2**9),
               np.sin(4*t)+np.cos(3*t) + np.random.normal(0,0.9,size=2**9)]
    y_names = ["sin(t)","cos(t)","sin(4t)+sin(3t)","sin(4t)+cos(3t) + 0.3*N(0,1)","sin(4t)+cos(3t) + 0.6*N(0,1)","sin(4t)+cos(3t) + 0.9*N(0,1)"]

    for y,y_name in zip(y_funcs,y_names):
        beta_vals = np.linspace(0.01,0.99,100)
        number_of_zeroes = []
        hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet,mode)
        for beta in beta_vals:
            besov_tree = bf.BesovForest(wave_coef, beta)
            transform_coeff = besov_tree.getMinimizingPosteriorCoefficients()
            number_of_zeroes.append(len([x for x in transform_coeff.values() if x == 0]) / 2**9)
        plt.plot(beta_vals,number_of_zeroes,label=y_name)
    plt.title("Haar-based Besov trees: percentage of zero coefficients\nin the wavelet coefficients as a function of beta")
    plt.ylabel("Percentage of zeroe coefficients")
    plt.xlabel("Beta")
    plt.legend()
    plt.show()

def plot_function(t, y, wavelet, mode, beta, giveTreePlot=False, title=""):
    # fig.suptitle(f"Besov Tree estimates based on {wavelet} wavelets plot of sin(4t)+cos(3t) with noise level 0.3\nDifferent values for beta")
    hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet, mode)
    besov_forest = bf.BesovForest(wave_coef, beta)
    transform_coeff = besov_forest.getMinimizingPosteriorCoefficients()
    transform_y = wt.inverseDWT((hsm, transform_coeff), wavelet,mode)
    plt.plot(t, y, label="Original")
    plt.title(title)
    plt.plot(t, transform_y[0:len(y)], label=f"{wavelet} wavelet with mode {mode} transform beta={beta}")
    plt.legend()
    plt.show()
    if giveTreePlot:
        makePlotBasedOnWaveletCoefficients(transform_coeff, wavelet, mode, beta)


def set_percentage_of_coefficients_to_zero(coefficients, percentage):
    new_coefficients = coefficients.copy()
    number_of_coefficients_to_set_to_zero = int(len(new_coefficients) * percentage)
    coefficient_s_sorted_by_value = sorted(new_coefficients.items(), key=lambda x: abs(x[1]))
    for i in range(number_of_coefficients_to_set_to_zero):
        new_coefficients[coefficient_s_sorted_by_value[i][0]] = 0
    return new_coefficients

def set_percentage_of_coefficients_to_non_zero(coefficients, percentage):
    new_coefficients = coefficients.copy()
    number_of_coefficients_to_set_to_zero = len(coefficients) - int(len(new_coefficients) * percentage)
    coefficient_s_sorted_by_value = sorted(new_coefficients.items(), key=lambda x: abs(x[1]))
    for i in range(number_of_coefficients_to_set_to_zero):
        new_coefficients[coefficient_s_sorted_by_value[i][0]] = 0
    return new_coefficients

def MSE_wavelet_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys(), "The two dictionaries do not have the same keys"
    sum = 0
    for key in dict1.keys():
        sum += (dict1[key] - dict2[key]) ** 2
    return sum/len(dict1)

def compare_MSE():
    t = np.linspace(0, 2 * np.pi, 2 ** 9)
    y_real = 1/(t+1)
    # y_real = np.sin(4 * t) + np.cos(8 * t)
    y = y_real + np.random.normal(0, 0.3, size=2 ** 9)
    wavelet = "db2"
    mode = "per"
    hsm, wave_coef_real = wt.getWaveletCoefficients(y_real, wavelet, mode)
    hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet, mode)
    mse_set_perc_to_zero = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle(
        "MSE between original wavelet coefficients and wavelet coefficients\nFor sin(4t)+cos(8t) with noise level 0.3")

    for i in np.arange(0, 1.01, 0.01):
        new_coeffs = set_percentage_of_coefficients_to_non_zero(coefficients=wave_coef, percentage=i)
        x = MSE_wavelet_dicts(wave_coef_real, new_coeffs)
        mse_set_perc_to_zero.append(x)
    axs[0].plot(np.arange(0, 1.01, 0.01), mse_set_perc_to_zero)
    axs[0].set_title("Set smallest $i$ coeffs to zero")
    axs[0].set_xlabel("Percentage of coefficients set to zero")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("MSE")

    mse_beta = []
    beta_range = np.arange(0.01, 0.6, 0.001)
    for i in beta_range:
        besov_forest = bf.BesovForest(wave_coef, beta=i)
        transform_coeff = besov_forest.getMinimizingPosteriorCoefficients()
        x = MSE_wavelet_dicts(wave_coef_real, transform_coeff)
        mse_beta.append(x)
    axs[1].plot(beta_range, mse_beta)
    axs[1].set_title("Beta $beta$ different values")
    axs[1].set_xlabel("Beta")
    fig.show()


def compare_MSE_given_wavelet_coeffs(wave_coef,wave_coef_real):
    mse_set_perc_to_zero = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle(
        "MSE between original wavelet coefficients and wavelet coefficients")

    for i in np.arange(0, 1.01, 0.01):
        new_coeffs = set_percentage_of_coefficients_to_non_zero(coefficients=wave_coef, percentage=i)
        x = MSE_wavelet_dicts(wave_coef_real, new_coeffs)
        mse_set_perc_to_zero.append(x)
    axs[0].plot(np.arange(0, 1.01, 0.01), mse_set_perc_to_zero)
    axs[0].set_title("Set smallest $i$ coeffs to zero")
    axs[0].set_xlabel("Percentage of coefficients set to zero")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("MSE")

    mse_beta = []
    beta_range = np.arange(0.01, 0.6, 0.001)
    for i in beta_range:
        besov_forest = bf.BesovForest(wave_coef, beta=i)
        transform_coeff = besov_forest.getMinimizingPosteriorCoefficients()
        x = MSE_wavelet_dicts(wave_coef_real, transform_coeff)
        mse_beta.append(x)
    axs[1].plot(beta_range, mse_beta)
    axs[1].set_title("Beta $beta$ different values")
    axs[1].set_xlabel("Beta")
    fig.show()

def tree_bottom_analysis():
    t = np.linspace(0, 2 * np.pi, 2 ** 9)
    y_real = np.sin(4 * t) + np.cos(8 * t)
    y = y_real + np.random.normal(0, 0.3, size=2 ** 9)
    wavelet = "db2"
    mode = "per"
    hsm, wave_coef_real = wt.getWaveletCoefficients(y_real, wavelet, mode)
    hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet, mode)

    new_coefficients = {}
    coefficient_s_sorted_by_value = sorted(wave_coef.items(), key=lambda x: x[0][0])
    for idx, coeff in enumerate(coefficient_s_sorted_by_value):
        if coeff[0][0] != 6:
            new_coefficients[coeff[0]] = 0
        else:
            new_coefficients[coeff[0]] = np.sin(coeff[0][1])

    new_real_coefficients = {}
    coefficient_s_sorted_by_value = sorted(wave_coef_real.items(), key=lambda x: x[0][0])
    for idx, coeff in enumerate(coefficient_s_sorted_by_value):
        if coeff[0][0] != 6:
            new_real_coefficients[coeff[0]] = 0
        else:
            new_real_coefficients[coeff[0]] = coeff[1]

    transform_y = wt.inverseDWT((hsm, new_coefficients), wavelet, mode)
    transform_real_y = wt.inverseDWT((hsm, new_real_coefficients), wavelet, mode)
    plt.plot(t, y, label="Original")
    plt.plot(t, transform_real_y, label="Real")
    plt.plot(t, transform_y[0:len(y)], label=f"{wavelet} wavelet with mode {mode} keeping only bottom of tree")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sinus_plot()