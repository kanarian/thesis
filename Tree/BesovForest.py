from dataclasses import dataclass
from Tree.BesovTree import BesovTree

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

    def unsetRootFrom0_0ForAllSubtrees(self, forest):
        invertedForest = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] + 2 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            invertedForest[el] = newSubtree
        return invertedForest

    # This function causes all elements of the forest to be rooted at (0,0), thereby setting all
    # subnodes to the usual conventions wrt parent-children indexing
    def setRootTo0_0ForAllSubtrees(self, forest):
        forestCorrectedRoots = {}
        for el in forest:
            subTree = forest[el]
            newSubtree = {}
            for index in subTree:
                new_index = index[1] - 2 ** (index[0]) * el
                newSubtree[(index[0], new_index)] = subTree[index]
            forestCorrectedRoots[el] = newSubtree
        return forestCorrectedRoots

    def runBesovTreeAlgorithmPerTree(self,forest):
        new_coeffs = {}
        for el in forest:
            subTree = forest[el]
            subBesovTree = BesovTree(subTree, self.beta)
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
        new_coeffs = self.runBesovTreeAlgorithmPerTree(indicesTransformedForest)
        invertedIndicesForest = self.unsetRootFrom0_0ForAllSubtrees(new_coeffs)
        flattenedForest = self.flattenDict(invertedIndicesForest)
        return flattenedForest
