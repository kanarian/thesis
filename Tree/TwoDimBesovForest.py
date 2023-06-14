from dataclasses import dataclass
import Tree.TwoDimBesovTree as tbt


class TwoDimBesovForest:
    """"One or multiple Two Dim Besov Trees"""
    def __init__(self, wavelet_coefficients, beta, max_level, start_level=0):
        self.wavelet_coefficients = wavelet_coefficients
        self.beta = beta
        self.max_level = max_level
        self.start_level = start_level

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
                    if (i[0] - 1, i[1] // 4) in forest[j]:
                        forest[j][i] = self.wavelet_coefficients[i]
                        break
        return forest

    # This function causes all elements of the forest to be rooted at (0,0), thereby setting all
    # subnodes to the usual conventions wrt parent-children indexing
    def setRootTo0_0ForAllSubtrees(self, forest):
        forestCorrectedRoots = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] - 4 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            forestCorrectedRoots[el] = newSubtree
        return forestCorrectedRoots

    def unsetRootFrom0_0ForAllSubtrees(self, forest):
        invertedForest = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] + 4 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            invertedForest[el] = newSubtree
        return invertedForest

    def runTwoDimBesovTreeAlgorithmPerTree(self,forest):
        new_coeffs = {}
        for el in forest:
            subTree = forest[el]
            subBesovTree = tbt.TwoDimBesovTree(subTree, self.beta, self.max_level, start_level=self.start_level)
            thisCoefficients = subBesovTree.getMinimizingPosteriorCoefficients()
            new_coeffs[el] = thisCoefficients
        return new_coeffs

    def flattenDict(self, dict):
        new_dict = {}
        for el in dict:
            new_dict = {**new_dict, **dict[el]}
        return new_dict

    def getMinimizingPosteriorCoefficients(self):
        forest = self.initializeForest()
        indicesTransformedForest = self.setRootTo0_0ForAllSubtrees(forest)
        new_coeffs = self.runTwoDimBesovTreeAlgorithmPerTree(indicesTransformedForest)
        invertedIndicesForest = self.unsetRootFrom0_0ForAllSubtrees(new_coeffs)
        flattenedForest = self.flattenDict(invertedIndicesForest)
        return flattenedForest