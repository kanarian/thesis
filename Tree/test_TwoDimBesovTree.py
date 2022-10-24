from unittest import TestCase
from TwoDimBesovTree import TwoDimBesovTree
import math

def createWaveletDictAllOnes(levels):
    wavelet_dict = {}
    for level in range(0, levels):
        for j in range(0, 4**level):
            wavelet_dict[(level, j)] = {"cH": 1, "cV": 1, "cD": 1}
    return wavelet_dict

class TestTwoDimBesovTree(TestCase):
    def setUp(self):
        self.wavelet_dict = {(0, 0): {"cH": .2, "cV": .3, "cD": .4},
                        (1, 0): {"cH": 0, "cV": .3, "cD": .4},
                        (1, 1): {"cH": 0, "cV": .5, "cD": 0},
                        (1, 2): {"cH": .1, "cV": 3, "cD": .2},
                        (1, 3): {"cH": .1, "cV": .1, "cD": .1}}
        self.beta = 0.3
        self.max_depth = 1
        self.tbt = TwoDimBesovTree(self.wavelet_dict, self.beta, self.max_depth)

    def test_initialization(self):
        self.assertEqual(self.tbt.wavelet_coefficients, self.wavelet_dict)
        self.assertEqual(self.tbt.beta, self.beta)
        self.assertEqual(self.tbt.max_depth, self.max_depth)

    def test_getParentIndex(self):
        for i in range(0, 4):
            self.assertEqual(self.tbt.getParentIndex(1, i), (0, 0))

    def test_getIthChildIndex(self):
        z = createWaveletDictAllOnes(3)
        larger_tree = TwoDimBesovTree(z, self.beta, 3)
        for i in range(0,4):
            self.assertEqual(larger_tree.getXthChildIndex(1, 3, i), (2, 12 + i))

    def test_mForSubtree(self):
        self.assertAlmostEqual(self.tbt.mForSubtree(0, 0, "cH"),
                               0.2**2 + .1**2+ .1**2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.mForSubtree(0, 0, "cV"),
                               2 * (0.3 ** 2) + .1 ** 2 + .5**2 + 3**2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.mForSubtree(0, 0, "cD"),
                               2*(.4 ** 2) + .2 ** 2 + .1 ** 2, delta=0.000001)

    def test_mForSubtree_leaf(self):
        self.assertAlmostEqual(self.tbt.mForSubtree(1, 0, "cH"),
                               0, delta=0.000001)
        self.assertAlmostEqual(self.tbt.mForSubtree(1, 0, "cV"),
                               .3**2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.mForSubtree(1, 0, "cD"),
                               .4**2, delta=0.000001)

    def test_calcF(self):
        self.assertAlmostEqual(self.tbt.calcF(0, 0, "cH"), .25*0.2**2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.calcF(0, 0, "cV"), .25*0.3**2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.calcF(0, 0, "cD"), .25*0.4**2, delta=0.000001)

    def test_initializeLeafsCorrectLength(self):
        self.tbt.initializeLeafs()
        self.assertEqual(len(self.tbt.F), 4*3)

    def test_initializeLeafsCorrectValues(self):
        self.tbt.initializeLeafs()
        self.assertAlmostEqual(self.tbt.F[1, 0,"cH"], .25 * 0 ** 2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[1, 0, "cV"], .25 * .3 ** 2, delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[1, 0, "cD"], .25 * .4 ** 2, delta=0.000001)

    def test_getNumberOfNodesInSubtreeRoot(self):
        self.assertEqual(self.tbt.getNumberOfNodesInSubtree(0), 5)

    def test_getNumberOfNodesInSubtreeLeaf(self):
        self.assertEqual(self.tbt.getNumberOfNodesInSubtree(1), 1)

    def test_consider0thSubtree(self):
        self.tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            self.tbt.F[0, 0, detail] = self.tbt.calcF(0, 0, detail)
        self.tbt.considerSubTree(0, 0, 0)
        self.assertAlmostEqual(self.tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 0.5*0**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cV"], 0.25 * 0.3 ** 2 + 0.5*0.3**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cD"], 0.25 * 0.4 ** 2 + 0.5*0.4**2 - math.log(1-self.beta), delta=0.000001)

    def test_consider1thSubtree(self):
        self.tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            self.tbt.F[0, 0, detail] = self.tbt.calcF(0, 0, detail)
        self.tbt.considerSubTree(0, 0, 1)

        self.assertAlmostEqual(self.tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 0.5*0**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cV"], 0.25 * 0.3 ** 2 + 0.5*0.5**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cD"], 0.25 * 0.4 ** 2 + 0.5*0**2 - math.log(1-self.beta), delta=0.000001)

    def test_consider2thSubtree(self):
        self.tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            self.tbt.F[0, 0, detail] = self.tbt.calcF(0, 0, detail)
        self.tbt.considerSubTree(0, 0, 2)
        self.assertAlmostEqual(self.tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 0.5*0.1**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cV"], 0.25 * 0.3 ** 2 + .25*3**2 - math.log(self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cD"], 0.25 * 0.4 ** 2 + 0.5*0.2**2 - math.log(1-self.beta), delta=0.000001)

    def test_consider3thSubtree(self):
        self.tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            self.tbt.F[0, 0, detail] = self.tbt.calcF(0, 0, detail)
        self.tbt.considerSubTree(0, 0, 3)
        self.assertAlmostEqual(self.tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 0.5*0.1**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cV"], 0.25 * 0.3 ** 2 + 0.5*0.1**2 - math.log(1-self.beta), delta=0.000001)
        self.assertAlmostEqual(self.tbt.F[0,0, "cD"], 0.25 * 0.4 ** 2 + 0.5*0.1**2 - math.log(1-self.beta), delta=0.000001)

    def test_considerAddingAllSubtrees(self):
        beta = 0.9
        tbt = TwoDimBesovTree(self.wavelet_dict,beta, self.max_depth)
        tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            tbt.F[0,0, detail] = tbt.calcF(0, 0, detail)
        for i in range(4):
            tbt.considerSubTree(0, 0, i)
        self.assertAlmostEqual(tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 2*0.25*0.1**2 - 4*math.log(beta), delta=0.000001)
        for detail, i in zip(["cH", "cV", "cD"],range(0,4)):
            self.assertEqual(tbt.t[1, i, detail], 1)

    def test_considerAddingNoSubtrees(self):
        beta = 0.001
        tbt = TwoDimBesovTree(self.wavelet_dict,beta, self.max_depth)
        tbt.initializeLeafs()
        for detail in ["cH", "cV", "cD"]:
            tbt.F[0,0, detail] = tbt.calcF(0, 0, detail)
        for i in range(4):
            tbt.considerSubTree(0, 0, i)
        self.assertAlmostEqual(tbt.F[0,0, "cH"], 0.25 * 0.2 ** 2 + 2*0.5*0.1**2 - 4*math.log(1-beta), delta=0.000001)
        for detail, i in zip(["cH", "cV", "cD"],range(0,4)):
            self.assertEqual(tbt.t[1, i, detail], 0)
