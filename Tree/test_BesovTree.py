from unittest import TestCase
from BesovTree import BesovTree


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
        assert(besov_tree.getLeftIndex(0, 0, 1) == (1, 0))

    def test_get_left_index_throws(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        with self.assertRaises(AssertionError):
            besov_tree.getLeftIndex(1, 1, 1)

    def test_get_right_index(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.getRightIndex(0, 0, 1) == (1, 1))

    def test_get_right_index_throws(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        with self.assertRaises(AssertionError):
            besov_tree.getRightIndex(1, 1, 1)

    def test_m_for_subtree(self):
        wavelet_dict = {(0, 0): 1, (1, 0): 1, (1, 1): 1}
        beta = 0.5
        besov_tree = BesovTree(wavelet_dict, beta)
        assert(besov_tree.mForSubtree(0, 0,1) == 3)

    def test_get_transformed_coefficients(self):
        self.fail()
