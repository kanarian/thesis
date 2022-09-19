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

    def getLeftIndex(self, j, k):
        assert j + 1 <= self.j_max, "Node on max_depth has no left child"
        return j + 1, 2 * k

    def getRightIndex(self, j, k):
        assert j + 1 <= self.j_max, "Node on max_depth has no right child"
        return j + 1, 2 * k + 1

    def mForSubtree(self, j, k):
        sum = 0
        thisQueue = Queue()
        thisQueue.put((j, k))
        while not thisQueue.empty():
            currJ, currK = thisQueue.get()
            sum += abs(self.wavelet_coefficients[currJ, currK])**2
            if currJ + 1 <= self.j_max:
                thisQueue.put(self.getLeftIndex(currJ, currK))
                thisQueue.put(self.getRightIndex(currJ, currK))
        return sum

    def initializeLeafs(self):
        for k in range(0, 2 ** self.j_max):
            self.F[(self.j_max, k)] = self.calcF(self.j_max, k)

    def calcF(self, j, k):
        return .25 * abs(self.wavelet_coefficients[j, k]) ** 2

    def getNumberOfNodesInSubtree(self, j):
        return 2 ** (self.j_max - j + 1) - 1

    def considerSubTree(self, j, k, child_index):
        assert child_index == 0 or child_index == 1, "child_index must be 0 or 1"
        s = self.getNumberOfNodesInSubtree(j + 1)
        if child_index == 0:
            j_c, k_c = self.getLeftIndex(j, k)
        else:
            j_c, k_c = self.getRightIndex(j, k)
        subTreeVal = .5 * self.mForSubtree(j_c, k_c)
        if self.F[j_c, k_c] - math.log(self.beta) < subTreeVal - s * math.log(1 - self.beta):
            self.t[j_c, k_c] = 1
            self.F[j, k] = self.F[j, k] + self.F[j_c, k_c] - math.log(self.beta)
        else:
            self.t[j_c, k_c] = 0
            self.F[j, k] = self.F[j, k] + subTreeVal - s * math.log(1 - self.beta)

    def createConnectedTree(self):
        t_tilde = {}

        t_tilde[(0, 0)] = 1
        for j in range(1, self.j_max + 1):
            for k in range(0, 2 ** j - 1 + 1):
                j_p, k_p = self.getParentIndex(j, k)
                t_tilde[(j, k)] = t_tilde[j_p, k_p] if self.t[(j, k)] == 1 else 0
        return t_tilde

    def getMinimizingCoefficients(self, t_tilde):
        g = {}
        for j in range(0, self.j_max + 1):
            for k in range(0, 2 ** j - 1 + 1):
                if abs(t_tilde[j, k] - 1) < 1e-10:
                    # g[j,k] = self.wavelet_coefficients[j,k]*.5
                    g[j, k] = self.wavelet_coefficients[j, k]
                else:
                    g[j, k] = 0
        return g

    def getMaxDepth(self):
        return int(math.log2(len(self.wavelet_coefficients)))

    def getMinimizingPosteriorCoefficients(self):
        self.initializeLeafs()

        j = self.j_max - 1
        while j >= 0:
            for k in range(0, 2 ** j):
                self.F[(j, k)] = self.calcF(j, k)
                self.considerSubTree(j, k, 0)
                self.considerSubTree(j, k, 1)
            j = j - 1

        t_tilde = self.createConnectedTree()
        g = self.getMinimizingCoefficients(t_tilde)
        return g

    def __post_init__(self):
        self.j_max = self.getMaxDepth()
        self.t = {}
        self.F = {}
