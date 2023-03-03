from dataclasses import dataclass
import math
import pywt
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import WaveletTransform.TwoDWaveletTransformer as wt2d
from collections import defaultdict
from collections import deque
from numba import jit


@dataclass
class TwoDimBesovTree:
    """2D Besov Tree"""
    wavelet_coefficients: dict
    beta: float
    max_depth: int
    start_level: int = 0
    mForSubtreeAllDetailsCache = {}

    def __post_init__(self):
        # self.j_max = self.getMaxDepth()
        self.t = {}
        self.F = {}

    def getParentIndex(self, j, k):
        assert j > self.start_level, f"Node on level {j} has no parent"
        return j - 1, math.floor(k / 4)

    def getXthChildIndex(self, j, k, i):
        assert j + 1 <= self.max_depth, "Node on max_depth has no child"
        assert i >= 0 and i <= 3, "i must be between 0 and 3"
        return j + 1, 4 * k + i

    # should return {"cV":10, "cH": 10, "cD": 10}
    def mForSubtreeAllDetails(self, j, k):
        sum = {"cV": 0, "cH": 0, "cD": 0}

        if (j, k) in self.mForSubtreeAllDetailsCache:
            return self.mForSubtreeAllDetailsCache[j, k]
        thisQueue = deque()
        thisQueue.append((j, k))

        while thisQueue:
            thisSum = {"cV": 0, "cH": 0, "cD": 0}
            currJ, currK = thisQueue.pop()
            if (currJ, currK) in self.mForSubtreeAllDetailsCache:
                sum["cV"] += .5*self.mForSubtreeAllDetailsCache[currJ, currK]["cV"]
                sum["cH"] += .5*self.mForSubtreeAllDetailsCache[currJ, currK]["cH"]
                sum["cD"] += .5*self.mForSubtreeAllDetailsCache[currJ, currK]["cD"]
                thisSum["cV"] = .5*self.wavelet_coefficients[currJ, currK]["cV"]**2
                thisSum["cH"] = .5*self.wavelet_coefficients[currJ, currK]["cH"]**2
                thisSum["cD"] = .5*self.wavelet_coefficients[currJ, currK]["cD"]**2
                continue

            self.mForSubtreeAllDetailsCache[currJ, currK] = thisSum

            sum["cV"] += thisSum["cV"]
            sum["cH"] += thisSum["cH"]
            sum["cD"] += thisSum["cD"]
            if currJ + 1 <= self.max_depth:
                for i in range(0, 4):
                    thisQueue.append(self.getXthChildIndex(currJ, currK, i))
        self.mForSubtreeAllDetailsCache[j, k] = sum
        return sum

    # def calcF(self, j, k, detail):
    #     assert detail in ["cH", "cV", "cD"], "detail must be one of cH, cV, cD"
    #     return .25*self.wavelet_coefficients[j, k][detail]**2

    def initializeLeafs(self):
        for k in range(0, 4 ** self.max_depth):
            for detail in ["cH", "cV", "cD"]:
                self.F[(self.max_depth, k, detail)] = .25*self.wavelet_coefficients[self.max_depth, k][detail]**2

    def getNumberOfNodesInSubtree(self, j):
        levelsBelow = self.max_depth - j
        # number of nodes in a subtree of depth j is (4^j - 1) / 3
        # this can be seen by the geometric series formula
        return (4 ** (levelsBelow + 1) - 1) / 3

    def considerSubTree(self, j, k, child_index):
        # assert j + 1 <= self.max_depth, "Node on max_depth has no child"
        # assert child_index in [0, 1, 2, 3], "child_index must be 0, 1, 2 or 3"
        s = (4 ** (j + 1 + 1) - 1) / 3
        j_c, k_c = j + 1, 4 * k + child_index

        # subTreeVals = self.mForSubtreeAllDetails(j_c, k_c)
        subTreeVals= {"cV": 0, "cH": 0, "cD": 0}
        thisBeta = self.beta
        mathLogOneMinBeta = math.log(1 - thisBeta)
        mathLogBeta = math.log(thisBeta)


        for detail in ["cH", "cV", "cD"]:
            thisDetail = subTreeVals[detail]
            F_jkd = self.F[j, k, detail]
            F_jkcd = self.F[j_c, k_c, detail]
            if F_jkcd - math.log(thisBeta) < thisDetail - s * mathLogOneMinBeta:
                self.t[j_c, k_c, detail] = 1
                self.F[j, k, detail] = F_jkd + F_jkcd - mathLogBeta
            else:
                self.t[j_c, k_c, detail] = 0
                self.F[j, k, detail] = F_jkd + thisDetail - s * mathLogOneMinBeta

    def createConnectedTree(self):
        t_tilde = {}

        for j in range(0, self.start_level + 1):
            for k in range(0, 4 ** j - 1 + 1):
                t_tilde[j, k] = 1
        for j in range(self.start_level + 1, self.max_depth + 1):
            for k in range(0, 4 ** j - 1 + 1):
                j_p, k_p = self.getParentIndex(j, k)
                # to_add = any([self.t[j, k, "cH"],self.t[j, k, "cV"],self.t[j, k, "cD"]])
                to_add = all([self.t[j, k, "cH"], self.t[j, k, "cV"], self.t[j, k, "cD"]])
                t_tilde[(j, k)] = t_tilde[j_p, k_p] if to_add else 0
        return t_tilde

    def copyWaveletCoefficientsZero(self):
        g = {}
        for key in self.wavelet_coefficients.keys():
            g[key] = {"cH": 0, "cV": 0, "cD": 0}
        return g


    def getMinimizingCoefficients(self, t_tilde):
        g = self.copyWaveletCoefficientsZero()
        for j in range(0, self.max_depth + 1):
            for k in range(0, 4 ** j - 1 + 1):
                if abs(t_tilde[j, k] - 1) < 1e-10:
                    for detail in ["cH", "cV", "cD"]:
                        # g[j, k][detail] = self.wavelet_coefficients[j, k][detail]*.5
                        g[j, k][detail] = self.wavelet_coefficients[j, k][detail]
                elif abs(t_tilde[j, k] - 1) > 1e-10:
                    for detail in ["cH", "cV", "cD"]:
                        g[j,k][detail] = 0
        return g

    def getMinimizingPosteriorCoefficients(self):
        self.initializeLeafs()

        j = self.max_depth - 1
        while j >= 0:
            for k in range(0, 4 ** j):
                for detail in ["cH", "cV", "cD"]:
                    self.F[j, k, detail] = .25*self.wavelet_coefficients[j, k][detail]**2
                self.considerSubTree(j, k, 0)
                self.considerSubTree(j, k, 1)
                self.considerSubTree(j, k, 2)
                self.considerSubTree(j, k, 3)
            j = j - 1

        t_tilde = self.createConnectedTree()
        g = self.getMinimizingCoefficients(t_tilde)
        return g
