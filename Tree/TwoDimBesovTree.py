from dataclasses import dataclass
from queue import Queue
import math
import pywt
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
from collections import defaultdict



@dataclass
class TwoDimBesovTree:
    """2D Besov Tree"""
    wavelet_coefficients: dict
    beta: float
    max_depth: int

    def __post_init__(self):
        # self.j_max = self.getMaxDepth()
        self.t = {}
        self.F = {}

    def getParentIndex(self, j, k):
        assert j > 0, f"Node on level {j} has no parent"
        return j - 1, math.floor(k / 4)

    def getXthChildIndex(self, j, k, i):
        assert j + 1 <= self.max_depth, "Node on max_depth has no child"
        assert i >= 0 and i <= 3, "i must be between 0 and 3"
        return j + 1, 4 * k + i

    def mForSubtree(self, j, k, detail):
        assert detail in ["cH", "cV", "cD"], "detail must be one of cH, cV, cD"
        sum = 0
        thisQueue = Queue()
        thisQueue.put((j, k))
        while not thisQueue.empty():
            currJ, currK = thisQueue.get()
            sum += abs(self.wavelet_coefficients[currJ, currK][detail])**2
            if currJ + 1 <= self.max_depth:
                for i in range(0, 4):
                    thisQueue.put(self.getXthChildIndex(currJ, currK, i))
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
        j_c, k_c = self.getXthChildIndex(j, k, child_index)

        for detail in ["cH", "cV", "cD"]:
            subTreeVal = .5 * self.mForSubtree(j_c, k_c, detail)
            if self.F[j_c, k_c, detail] - math.log(self.beta) < subTreeVal - s * math.log(1 - self.beta):
                self.t[j_c, k_c, detail] = 1
                self.F[j, k, detail] = self.F[j, k, detail] + self.F[j_c, k_c, detail] - math.log(self.beta)
            else:
                self.t[j_c, k_c, detail] = 0
                self.F[j, k, detail] = self.F[j, k, detail] + subTreeVal - s * math.log(1 - self.beta)

    def createConnectedTree(self):
        t_tilde = {}

        t_tilde[(0, 0)] = 1
        for j in range(1, self.max_depth + 1):
            for k in range(0, 4 ** j - 1 + 1):
                j_p, k_p = self.getParentIndex(j, k)
                to_add = any([self.t[j, k, "cH"],self.t[j, k, "cV"],self.t[j, k, "cD"]])
                t_tilde[(j, k)] = t_tilde[j_p, k_p] if to_add else 0
        return t_tilde

    def getMinimizingCoefficients(self, t_tilde):
        g = self.wavelet_coefficients.copy()

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
                    self.F[j, k, detail] = self.calcF(j, k,detail)
                self.considerSubTree(j, k, 0)
                self.considerSubTree(j, k, 1)
                self.considerSubTree(j, k, 2)
                self.considerSubTree(j, k, 3)
            j = j - 1

        t_tilde = self.createConnectedTree()
        g = self.getMinimizingCoefficients(t_tilde)
        return g


x = np.arange(16*16*16*16).reshape((16*16, 16*16))
# x = np.arange(8*8).reshape((8, 8))
[hsm, wave_levels] = wt2d.get2DWaveletCoefficients(x, 'haar',"zero")
z = wt2d.inverse2DDWT((hsm, wave_levels), 'haar', "zero")

# g = t_bt.getMinimizingPosteriorCoefficients()
# inverse = wt2d.inverse2DDWT((hsm, g), 'haar',"per")
