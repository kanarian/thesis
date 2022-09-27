from unittest import TestCase

from Tree.BesovTree import BesovTree
from Tree.BesovForest import BesovForest


class TestBesovForest(TestCase):
    def test_instantiation_one_tree(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1}
        beta = 0.5
        besov_forest = BesovForest(wavelet_dict,beta)
        self.assertEqual(besov_forest.wavelet_coefficients, wavelet_dict)
        self.assertEqual(besov_forest.beta, beta)

    def test_instantiation_mult_trees(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1,(0,1): 1, (1,2): 1, (1,3): 1}
        beta = 0.5
        besov_forest = BesovForest(wavelet_dict,beta)
        self.assertEqual(besov_forest.wavelet_coefficients, wavelet_dict)
        self.assertEqual(besov_forest.beta, beta)

    def test_initialization_forest_multiple_trees(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1,(0,1): 1, (1,2): 1, (1,3): 1}
        expected_res = {0: {(0, 0): 1, (1, 0): 1, (1, 1): 1}, 1: {(0, 1): 1, (1, 2): 1, (1, 3): 1}}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        self.assertEqual(init_forest, expected_res)

    def test_initialization_forest_one_trees(self):
        wavelet_dict = {(0,0): 1, (1,0): 1, (1,1): 1}
        expected_res = {0: {(0, 0): 1, (1, 0): 1, (1, 1): 1}}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        self.assertEqual(init_forest, expected_res)

    def test_setRootTo0_0ForAllASubtrees(self):
        wavelet_dict = {(0,0): 1, (1,0): 2, (1,1): 3,(0,1): 4, (1,2): 5, (1,3): 6}
        expected_res = {0: {(0, 0): 1, (1, 0): 2, (1, 1): 3}, 1: {(0, 0): 4, (1, 0): 5, (1, 1): 6}}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        transformed_indices = bf.setRootTo0_0ForAllSubtrees(init_forest)
        self.assertEqual(transformed_indices, expected_res)

    def test_runBesovTreePerAlgorithm(self):
        wavelet_dict = {(0,0): 1, (1,0): 2, (1,1): 3,(0,1): 4, (1,2): 5, (1,3): 6}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        transformed_indices = bf.setRootTo0_0ForAllSubtrees(init_forest)
        new_coeffs = bf.runBesovTreeAlgorithmPerTree(transformed_indices)

        wavelet_dict_tree_one = {(0,0): 1, (1,0): 2, (1,1): 3}
        wavelet_dict_tree_two = {(0,0): 4, (1,0): 5, (1,1): 6}
        bt_one_coeffs = BesovTree(wavelet_dict_tree_one,beta).getMinimizingPosteriorCoefficients()
        bt_two_coeffs = BesovTree(wavelet_dict_tree_two, beta).getMinimizingPosteriorCoefficients()
        exp_res = {0: bt_one_coeffs, 1: bt_two_coeffs}

        self.assertEqual(new_coeffs, exp_res)

    def test_unsetRootFrom0_0ForAllSubtrees(self):
        wavelet_dict = {(0,0): 1, (1,0): 2, (1,1): 3,(0,1): 4, (1,2): 5, (1,3): 6}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        transformed_indices = bf.setRootTo0_0ForAllSubtrees(init_forest)
        new_coeffs = bf.runBesovTreeAlgorithmPerTree(transformed_indices)
        transformed_back = bf.unsetRootFrom0_0ForAllSubtrees(new_coeffs)
        self.assertEqual(list(transformed_back[0].keys()), [(0,0), (1,0), (1,1)])
        self.assertEqual(list(transformed_back[1].keys()), [(0,1), (1,2), (1,3)])

    def test_flattenDict(self):
        wavelet_dict = {(0,0): 1, (1,0): 2, (1,1): 3,(0,1): 4, (1,2): 5, (1,3): 6}
        beta = 0.5
        bf = BesovForest(wavelet_dict,beta)
        init_forest = bf.initializeForest()
        transformed_indices = bf.setRootTo0_0ForAllSubtrees(init_forest)
        new_coeffs = bf.runBesovTreeAlgorithmPerTree(transformed_indices)
        transformed_back = bf.unsetRootFrom0_0ForAllSubtrees(new_coeffs)
        flattened = bf.flattenDict(transformed_back)
        self.assertCountEqual(list(flattened.keys()), [(1,0), (0,0), (1,1), (0,1), (1,2), (1,3)])