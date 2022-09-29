import graphviz
import queue
import numpy as np
import WaveletTransform.WaveletTransformer as wt
import Tree.BesovForest as bf

def makePlotBasedOnWaveletCoefficients(waveletCoefficientsDict, beta, wavelet, smooth):
    dot = graphviz.Digraph('g',comment='Wavelet Coefficients')
    dot.graph_attr['label'] = f'Wavelet Coefficients for beta = {beta} using {wavelet} wavelets and {smooth} extrapolation mode'
    nodeQueue = queue.Queue()
    for key, value in waveletCoefficientsDict.items():
        if value != 0:
            dot.node(str(key), str(key) + ": " + "{:.2e}".format(value))
            nodeQueue.put(key)
    while not nodeQueue.empty():
        currKey = nodeQueue.get()
        if (currKey[0]+1,2*currKey[1]) in waveletCoefficientsDict and waveletCoefficientsDict[(currKey[0]+1,2*currKey[1])] != 0:
            dot.edge(str(currKey), str((currKey[0] + 1, 2*currKey[1])) )
        if (currKey[0]+1,2*currKey[1] + 1) in waveletCoefficientsDict and waveletCoefficientsDict[(currKey[0] + 1, 2*currKey[1] + 1)] != 0:
            dot.edge(str(currKey), str((currKey[0] + 1, 2*currKey[1] + 1)))
    dot.render('waveletCoefficientsTemp', view=True)

# wavelet="db4"
# mode="per"
# beta_vals = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
#
# t = np.linspace(0, 2 * np.pi, 2 ** 9)
# y = np.sin(4*t)+np.cos(3*t) + np.random.random(2**9)
# hsm, wave_coef = wt.getWaveletCoefficients(y, wavelet, mode)
# for beta in beta_vals:
#     besov_forest = bf.BesovForest(wave_coef, beta)
#     transform_coeff = besov_forest.getMinimizingPosteriorCoefficients()
#     makePlotBasedOnWaveletCoefficients(transform_coeff,beta, wavelet, mode)