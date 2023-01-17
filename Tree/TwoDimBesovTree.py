from dataclasses import dataclass
from queue import Queue
import math
import pywt
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
from collections import defaultdict
import copy



@dataclass
class TwoDimBesovTree:
    """2D Besov Tree"""
    wavelet_coefficients: dict
    beta: float
    max_depth: int
    start_level: int = 0
    mForSubtreeCache = {}
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

    def mForSubtree(self, j, k, detail):
        assert detail in ["cH", "cV", "cD"], "detail must be one of cH, cV, cD"
        sum = 0
        if (j, k, detail) in self.mForSubtreeCache:
            return self.mForSubtreeCache[j, k, detail]
        thisQueue = Queue()
        thisQueue.put((j, k))
        while not thisQueue.empty():
            thisSum = 0
            currJ, currK = thisQueue.get()
            if (currJ, currK, detail) in self.mForSubtreeCache:
                sum += self.mForSubtreeCache[currJ, currK, detail]
                continue
            thisSum += abs(self.wavelet_coefficients[currJ, currK][detail])**2
            self.mForSubtreeCache[currJ, currK, detail] = thisSum
            sum += thisSum
            # sum += abs(self.wavelet_coefficients[currJ, currK][detail])**2
            if currJ + 1 <= self.max_depth:
                for i in range(0, 4):
                    thisQueue.put(self.getXthChildIndex(currJ, currK, i))
        self.mForSubtreeCache[j, k, detail] = sum
        return sum

    # should return {"cV":10, "cH": 10, "cD": 10}
    def mForSubtreeAllDetails(self, j, k):
        sum = {"cV": 0, "cH": 0, "cD": 0}

        if (j, k) in self.mForSubtreeAllDetailsCache:
            return self.mForSubtreeAllDetailsCache[j, k]
        thisQueue = Queue()
        thisQueue.put((j, k))

        while not thisQueue.empty():
            thisSum = {"cV": 0, "cH": 0, "cD": 0}
            currJ, currK = thisQueue.get()
            if (currJ, currK) in self.mForSubtreeAllDetailsCache:
                sum["cV"] += self.mForSubtreeAllDetailsCache[currJ, currK]["cV"]
                sum["cH"] += self.mForSubtreeAllDetailsCache[currJ, currK]["cH"]
                sum["cD"] += self.mForSubtreeAllDetailsCache[currJ, currK]["cD"]
                thisSum["cV"] = abs(self.wavelet_coefficients[currJ, currK]["cV"])**2
                thisSum["cH"] = abs(self.wavelet_coefficients[currJ, currK]["cH"])**2
                thisSum["cD"] = abs(self.wavelet_coefficients[currJ, currK]["cD"])**2
                continue

            self.mForSubtreeAllDetailsCache[currJ, currK] = thisSum

            sum["cV"] += thisSum["cV"]
            sum["cH"] += thisSum["cH"]
            sum["cD"] += thisSum["cD"]
            if currJ + 1 <= self.max_depth:
                for i in range(0, 4):

                    thisQueue.put(self.getXthChildIndex(currJ, currK, i))
        self.mForSubtreeAllDetailsCache[j, k] = sum
        return sum

    def calcF(self, j, k, detail):
        assert detail in ["cH", "cV", "cD"], "detail must be one of cH, cV, cD"
        return .25 * abs(self.wavelet_coefficients[j, k][detail]) ** 2

    def initializeLeafs(self):
        for k in range(0, 4 ** self.max_depth):
            for detail in ["cH", "cV", "cD"]:
                self.F[(self.max_depth, k, detail)] = self.calcF(self.max_depth, k, detail)

    def getNumberOfNodesInSubtree(self, j):
        levelsBelow = self.max_depth - j
        # number of nodes in a subtree of depth j is (4^j - 1) / 3
        # this can be seen by the geometric series formula
        return (4 ** (levelsBelow + 1) - 1) / 3

    def considerSubTree(self, j, k, child_index):
        assert j + 1 <= self.max_depth, "Node on max_depth has no child"
        assert child_index in [0, 1, 2, 3], "child_index must be 0, 1, 2 or 3"
        s = self.getNumberOfNodesInSubtree(j + 1)
        # print(s)
        j_c, k_c = self.getXthChildIndex(j, k, child_index)

        subTreeVals = {key: value*.5 for key, value in self.mForSubtreeAllDetails(j_c, k_c).items()}
        for detail in ["cH", "cV", "cD"]:
            if self.F[j_c, k_c, detail] - math.log(self.beta) < subTreeVals[detail] - s * math.log(1 - self.beta):
                self.t[j_c, k_c, detail] = 1
                self.F[j, k, detail] = self.F[j, k, detail] + self.F[j_c, k_c, detail] - math.log(self.beta)
            else:
                self.t[j_c, k_c, detail] = 0
                self.F[j, k, detail] = self.F[j, k, detail] + subTreeVals[detail] - s * math.log(1 - self.beta)

    def createConnectedTree(self):
        t_tilde = {}

        for j in range(0, self.start_level + 1):
            for k in range(0, 4 ** j - 1 + 1):
                t_tilde[j, k] = 1
        for j in range(self.start_level + 1, self.max_depth + 1):
            for k in range(0, 4 ** j - 1 + 1):
                j_p, k_p = self.getParentIndex(j, k)
                to_add = any([self.t[j, k, "cH"],self.t[j, k, "cV"],self.t[j, k, "cD"]])
                # to_add = all([self.t[j, k, "cH"], self.t[j, k, "cV"], self.t[j, k, "cD"]])
                # print(to_add)
                t_tilde[(j, k)] = t_tilde[j_p, k_p] if to_add else 0
        return t_tilde

    def copyWaveletCoefficientsZero(self):
        g = copy.deepcopy(self.wavelet_coefficients)
        for keys in g.keys():
            for detail in ["cH", "cV", "cD"]:
                g[keys][detail] = 0
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
                    self.F[j, k, detail] = self.calcF(j, k, detail)
                self.considerSubTree(j, k, 0)
                self.considerSubTree(j, k, 1)
                self.considerSubTree(j, k, 2)
                self.considerSubTree(j, k, 3)
            j = j - 1

        t_tilde = self.createConnectedTree()
        g = self.getMinimizingCoefficients(t_tilde)
        return g
