from dataclasses import dataclass
from queue import Queue
import math

@dataclass
class BesovTree:
    """Random Tree"""
    wavelet_coefficients: dict
    beta: float

    def getParentIndex(self, j, k):
        assert j > 0, f"Node on level {j} has no parent"
        return j - 1, math.floor(k / 2)

    def getLeftIndex(self, j, k, max_depth):
        assert j + 1 <= max_depth, "Node on max_depth has no left child"
        return j + 1, 2 * k

    def getRightIndex(self, j, k, max_depth):
        assert j + 1 <= max_depth, "Node on max_depth has no right child"
        return j + 1, 2 * k + 1

    def mForSubtree(self, j, k, max_depth):
        sum = 0
        thisQueue = Queue()
        thisQueue.put((j, k))
        while not thisQueue.empty():
            currJ, currK = thisQueue.get()
            sum += self.wavelet_coefficients[currJ, currK]
            if currJ + 1 <= max_depth:
                thisQueue.put(self.getLeftIndex(currJ, currK, max_depth))
                thisQueue.put(self.getRightIndex(currJ, currK, max_depth))
        return sum

    def __post_init__(self):
        j_max = int(math.log2(len(self.wavelet_coefficients)))

        t = {}
        F = {}
        for k in range(0, 2 ** j_max):
            F[(j_max, k)] = .25 * abs(self.wavelet_coefficients[j_max, k]) ** 2

        j = j_max - 1
        while j >= 0:
            s = 2 ** (j_max - j + 1) - 1
            for k in range(0, 2 ** j):
                F[(j, k)] = .25 * abs(self.wavelet_coefficients[j, k]) ** 2
                j_l, k_l = self.getLeftIndex(j, k, max_depth=j_max)
                if F[j_l, k_l] - math.log(self.beta) < .5 * abs(
                        self.mForSubtree(j_l, k_l, max_depth=j_max)) ** 2 - s * math.log(1 - self.beta):
                    t[j_l, k_l] = 1
                    F[j, k] = F[j, k] + F[j_l, k_l] - math.log(self.beta)
                else:
                    t[j_l, k_l] = 0
                    F[j, k] = F[j, k] + .5 * abs(self.mForSubtree(j_l, k_l, max_depth=j_max)) ** 2 - s * math.log(
                        1 - self.beta)

                j_r, k_r = self.getRightIndex(j, k, max_depth=j_max)
                if F[j_r, k_r] - math.log(self.beta) < .5 * abs(
                        self.mForSubtree(j_r, k_r, max_depth=j_max)) ** 2 - s * math.log(
                        1 - self.beta):
                    t[j_r, k_r] = 1
                    F[j, k] = F[j, k] + F[j_r, k_r] - math.log(self.beta)
                else:
                    t[j_r, k_r] = 0
                    F[j, k] = F[j, k] + .5 * abs(self.mForSubtree(j_r, k_r, max_depth=j_max)) ** 2 - s * math.log(
                        1 - self.beta)
            j = j - 1

        t_tilde = {}
        t_tilde[(0, 0)] = 1
        for j in range(1, j_max + 1):
            for k in range(0, 2 ** j - 1 + 1):
                j_p, k_p = self.getParentIndex(j, k)
                t_tilde[(j, k)] = t[j, k] * t_tilde[j_p, k_p]

        g = {}
        for j in range(0, j_max + 1):
            for k in range(0, 2 ** j - 1 + 1):
                if abs(t_tilde[j, k] - 1) < 1e-10:
                    g[j,k] = self.wavelet_coefficients[j,k]/2
                    g[j, k] = self.wavelet_coefficients[j, k]
                else:
                    g[j, k] = 0
        self.transformedValues = g

    def getTransformedCoefficients(self):
        return self.transformedValues
