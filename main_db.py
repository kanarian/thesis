import numpy as np
import WaveletTransform.WaveletTransformer as wt
import Tree.BesovTree as bt
import matplotlib.pyplot as plt

def plot():
    t = np.linspace(0, 2 * np.pi, 2 ** 9)
    y = 255*np.sin(t)
    hsm, wave_coef = wt.getWaveletCoefficients(y, "haar")
    plt.hist(y, bins=100)
    plt.show()
    beta_values = np.arange(0.001,0.6,0.01)
    plt.title(f"Plot of y=255*sin(t) with beta={beta_values}")
    plt.plot(t,y,label="original")
    for idx, beta in enumerate(beta_values):
        besov_tree = bt.BesovTree(wave_coef, beta)
        transform_coeff = besov_tree.getMinimizingPosteriorCoefficients()
        transform_y = wt.inverseDWT((hsm, transform_coeff), "haar")
        plt.plot(t,transform_y,label=f"{beta=}")
        print(list(besov_tree.createConnectedTree().values()).count(1))
    plt.legend()
    plt.show()



if __name__ == "__main__":
    plot()