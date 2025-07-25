import unittest

import numpy as np
import pandas as pd

from wtv.similarity import (
    calculate_solo_compound_combination_score,
    dot_product_distance,
)


class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.solo_compound = pd.DataFrame(
            {"Compound1": [20.0, 30.0], "ion": [204.09, 300.0]}, index=[204.09, 300.0]
        )

    def test_valid_vectors(self):
        # Test with valid vectors
        p = np.array([1, 2, 3])
        q = np.array([4, 5, 6])
        result = dot_product_distance(p, q)
        expected = (np.sum(p * q) ** 2) / (np.sum(p**2) * np.sum(q**2))
        self.assertAlmostEqual(result, expected, places=6)

    def test_is_normalized(self):
        p = np.array([1, 2, 3])
        q = np.array([2, 4, 6])
        assert dot_product_distance(p, q) == 1

    def test_zero_vector_p(self):
        # Test with p as a zero vector
        p = np.array([0, 0, 0])
        q = np.array([4, 5, 6])
        result = dot_product_distance(p, q)
        self.assertEqual(result, 0)

    def test_zero_vector_q(self):
        # Test with q as a zero vector
        p = np.array([1, 2, 3])
        q = np.array([0, 0, 0])
        result = dot_product_distance(p, q)
        self.assertEqual(result, 0)

    def test_both_zero_vectors(self):
        # Test with both p and q as zero vectors
        p = np.array([0, 0, 0])
        q = np.array([0, 0, 0])
        result = dot_product_distance(p, q)
        self.assertEqual(result, 0)

    def test_different_lengths(self):
        # Test with vectors of different lengths
        p = np.array([1, 2, 3])
        q = np.array([4, 5])
        with self.assertRaises(ValueError):
            dot_product_distance(p, q)

    def test_calculate_solo_compound_combination_score(self):
        actual = calculate_solo_compound_combination_score(self.solo_compound, 250)
        expected = pd.DataFrame(
            {
                "Compound1": [30.0, 20.0],
                "ion": [300.0, 1.0],
                "com_score": [1.478851e08, 4.472136],
            },
            index=[300.0, 204.09],
        )

        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
