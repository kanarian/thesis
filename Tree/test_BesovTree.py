from unittest import TestCase
from BesovTree import BesovTree
import math


class TestBesovTree(TestCase):
    def test_initialization(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.wavelet_coefficients == wavelet_dict)
        assert (besov_tree.beta == beta)

    def test_get_parent_index(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.getParentIndex(1, 0) == (0, 0))
        assert(besov_tree.getParentIndex(1, 1) == (0, 0))

    def test_get_parent_index_throws(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        with self.assertRaises(AssertionError):
            besov_tree.getParentIndex(0, 0)

    def test_get_left_index(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.getLeftIndex(0, 0) == (1, 0))

    def test_get_left_index_throws(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        with self.assertRaises(AssertionError):
            besov_tree.getLeftIndex(1, 1)

    def test_get_right_index(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.getRightIndex(0, 0) == (1, 1))

    def test_get_right_index_throws(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        with self.assertRaises(AssertionError):
            besov_tree.getRightIndex(1, 1)

    def test_m_for_subtree(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): -0.4}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        self.assertAlmostEqual(besov_tree.mForSubtree(0, 0), (1.2**2+0.3**2+0.4**2) ,delta=0.0001)

    def test_get_max_depth(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.getMaxDepth() == 1)

    def test_calcF(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        self.assertAlmostEqual(besov_tree.calcF(0, 0),0.25,delta=0.0001)

    def test_initializeLeafsCorrectLength(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        besov_tree.initializeLeafs()
        assert(len(besov_tree.F) == 2)

    def test_initializeLeafsCorrectValues(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        besov_tree.initializeLeafs()
        assert(besov_tree.F[(1, 0)] == .25)
        assert(besov_tree.F[(1, 1)] == .25)

    def test_considerLeftSubtree(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.2
        besov_tree = BesovTree(wavelet_dict, beta)
        # initializing leafs:
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 0)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.5*0.3**2 - math.log(1-beta), delta=0.0001)
        # left has not been chosen so the left child should have t[(1,0)] = 0
        self.assertEqual(besov_tree.t[(1, 0)], 0)

    def test_considerRightSubtree(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.2
        besov_tree = BesovTree(wavelet_dict, beta)
        # initializing leafs:
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 1)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.5*0.4**2 - math.log(1-beta), delta=0.0001)
        # left has not been chosen so the left child should have t[(1,0)] = 0
        self.assertEqual(besov_tree.t[(1, 1)], 0)

    def test_considerBothSubtrees(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.2
        besov_tree = BesovTree(wavelet_dict, beta)
        # initializing leafs:
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 0)
        besov_tree.considerSubTree(0, 0, 1)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.5*0.4**2 - math.log(1-beta) + 0.5*0.3**2 - math.log(1-beta), delta=0.0001)

        #none of the leafs were chosen
        self.assertEqual(besov_tree.t[(1, 1)], 0)
        self.assertEqual(besov_tree.t[(1, 0)], 0)

    def test_considerSubtreeLeftAdd(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.9
        besov_tree = BesovTree(wavelet_dict, beta)
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 0)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.25*0.3**2 - math.log(beta), delta=0.0001)
        self.assertEqual(besov_tree.t[(1, 0)], 1)

    def test_considerSubtreeRightAdd(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.9
        besov_tree = BesovTree(wavelet_dict, beta)
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 1)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.25*0.4**2 - math.log(beta), delta=0.0001)
        self.assertEqual(besov_tree.t[(1, 1)], 1)

    def test_considerSubtreeBothAdd(self):
        wavelet_dict = {(0, 0): 1.2, (1, 0): 0.3, (1, 1): 0.4}
        beta = 0.9
        besov_tree = BesovTree(wavelet_dict, beta)
        besov_tree.initializeLeafs()
        besov_tree.F[(0, 0)] = besov_tree.calcF(0, 0)
        besov_tree.considerSubTree(0, 0, 0)
        besov_tree.considerSubTree(0, 0, 1)
        self.assertAlmostEqual(besov_tree.F[(0, 0)], 0.25*1.2**2 + 0.25*0.4**2 - math.log(beta) + + 0.25*0.3**2 - math.log(beta), delta=0.0001)
        self.assertEqual(besov_tree.t[(1, 1)], 1)
        self.assertEqual(besov_tree.t[(1, 0)], 1)