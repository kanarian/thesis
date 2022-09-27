from dataclasses import dataclass
import pywt
from Tree.BesovTree import BesovTree
from WaveletTransform.WaveletTransformer import getWaveletCoefficients, inverseDWT

@dataclass
class BesovForest:
    """"One or multiple Besov Trees"""
    wavelet_coefficients: dict
    beta: float

    def initializeForest(self):
        forest = {}
        for i in self.wavelet_coefficients:
            # all nodes in the first level are the root of a subtree
            if i[0] == 0:
                forest[i[1]] = {i: self.wavelet_coefficients[i]}
            # not first level
            else:
                # one of the trees in forest should contain the parent
                for j in forest:
                    if (i[0] - 1, i[1] // 2) in forest[j]:
                        forest[j][i] = self.wavelet_coefficients[i]
                        break
        return forest

    def transformIndicesOfTreesInForest(self, forest):
        invertedForest = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] + 2 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            invertedForest[el] = newSubtree
        return invertedForest

    def inverseTransformIndicesOfTreesInForest(self,forest):
        forest_corrected_Indices = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] - 2 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            forest_corrected_Indices[el] = newSubtree
        return forest_corrected_Indices

    def runBesovTreeAlgorithmPerTree(self,forest):
        new_coeffs = {}
        for el in forest:
            subTree = forest[el]
            subBesovTree = BesovTree(subTree, 0.2)
            thisCoefficients = subBesovTree.getMinimizingPosteriorCoefficients()
            new_coeffs[el] = thisCoefficients
        return new_coeffs

    def flattenDict(dict):
        new_dict = {}
        for el in dict:
            new_dict = {**new_dict, **dict[el]}
        return new_dict

    def getMinimizingCoefficients(self):
        forest = self.initializeForest()
        indicesTransformedForest = self.transformIndicesOfTreesInForest(forest)
        new_coeffs = self.runBesovTreeAlgorithmPerTree(indicesTransformedForest)
        invertedIndicesForest = self.inverseTransformIndicesOfTreesInForest(new_coeffs)
        flattenedForest = self.flattenDict(invertedIndicesForest)
        return flattenedForest
