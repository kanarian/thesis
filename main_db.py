import numpy as np
import WaveletTransform.WaveletTransformer as wt

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