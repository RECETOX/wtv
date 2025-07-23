import unittest
import numpy as np
from wtv.similarity import dot_product_distance


class TestDotProductDistance(unittest.TestCase):
    def test_valid_vectors(self):
        # Test with valid vectors
        p = np.array([1, 2, 3])
        q = np.array([4, 5, 6])
        result = dot_product_distance(p, q)
        expected = (np.sum(p * q) ** 2) / (np.sum(p**2) * np.sum(q**2))
        self.assertAlmostEqual(result, expected, places=6)

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


if __name__ == "__main__":
    unittest.main()
