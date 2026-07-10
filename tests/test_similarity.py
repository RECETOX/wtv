"""Tests for wtv similarity module."""

import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum

from wtv.similarity import (
    WTVWeightedDotProduct,
    calculate_average_score_and_difference_count,
    calculate_combination_score,
    calculate_solo_compound_combination_score,
    dot_product_distance,
    weighted_dot_product_distance,
)


@pytest.fixture
def solo_compound():
    """Fixture for solo compound test data."""
    return pd.DataFrame(
        {"Compound1": [20.0, 30.0], "ion": [204.09, 300.0]}, index=[204.09, 300.0]
    )


class TestDotProductDistance:
    """Tests for dot product distance function."""

    def test_valid_vectors(self):
        """Test with valid vectors."""
        p = np.array([1, 2, 3])
        q = np.array([4, 5, 6])
        result = dot_product_distance(p, q)
        expected = (np.sum(p * q) ** 2) / (np.sum(p**2) * np.sum(q**2))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_is_normalized(self):
        """Test that parallel vectors give score of 1."""
        p = np.array([1, 2, 3])
        q = np.array([2, 4, 6])
        assert dot_product_distance(p, q) == 1

    def test_zero_vector_p(self):
        """Test with p as a zero vector."""
        p = np.array([0, 0, 0])
        q = np.array([4, 5, 6])
        result = dot_product_distance(p, q)
        assert result == 0

    def test_zero_vector_q(self):
        """Test with q as a zero vector."""
        p = np.array([1, 2, 3])
        q = np.array([0, 0, 0])
        result = dot_product_distance(p, q)
        assert result == 0

    def test_both_zero_vectors(self):
        """Test with both p and q as zero vectors."""
        p = np.array([0, 0, 0])
        q = np.array([0, 0, 0])
        result = dot_product_distance(p, q)
        assert result == 0

    def test_different_lengths_raises(self):
        """Test with vectors of different lengths raises error."""
        p = np.array([1, 2, 3])
        q = np.array([4, 5])
        with pytest.raises((ValueError, RuntimeError)):
            dot_product_distance(p, q)


class TestWeightedDotProductDistance:
    """Tests for weighted dot product distance function."""

    def test_weighted_dot_product_basic(self):
        """Test basic weighted dot product calculation."""
        compare_df = pd.DataFrame(
            {"spec1": [1.0, 2.0, 3.0], "spec2": [1.5, 2.5, 3.5]},
            index=[100.0, 200.0, 300.0],
        )
        result = weighted_dot_product_distance(compare_df, fr_factor=2.0)
        assert isinstance(result, float)
        assert result >= 0

    def test_weighted_dot_product_no_shared_peaks(self):
        """Test weighted dot product with no shared peaks."""
        compare_df = pd.DataFrame(
            {"spec1": [1.0, 0.0, 3.0], "spec2": [0.0, 5.0, 0.0]},
            index=[100.0, 200.0, 300.0],
        )
        result = weighted_dot_product_distance(compare_df, fr_factor=2.0)
        assert isinstance(result, float)


class TestWTVWeightedDotProduct:
    """Tests for WTVWeightedDotProduct class."""

    def test_pair_method(self):
        """Test pair method of WTVWeightedDotProduct."""
        similarity = WTVWeightedDotProduct(fr_factor=2.0)
        reference = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([1.0, 2.0, 3.0]),
        )
        target = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([1.5, 2.5, 3.5]),
        )
        result = similarity.pair(reference, target)
        assert isinstance(result, np.ndarray)
        assert result[()] >= 0

    def test_tolerance_property(self):
        """Test tolerance property getter/setter."""
        similarity = WTVWeightedDotProduct()
        similarity.tolerance = 0.5
        assert similarity.tolerance == 0.5

    def test_repr(self):
        """Test string representation."""
        similarity = WTVWeightedDotProduct(fr_factor=2.0, tolerance=0.1)
        repr_str = repr(similarity)
        assert "WTVWeightedDotProduct" in repr_str

    def test_custom_weights(self):
        """Test with custom intensity and m/z weights."""
        similarity = WTVWeightedDotProduct(
            fr_factor=2.0, intensity_weight=0.3, mz_weight=1.5
        )
        reference = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([1.0, 2.0, 3.0]),
        )
        target = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([1.5, 2.5, 3.5]),
        )
        result = similarity.pair(reference, target)
        assert isinstance(result, np.ndarray)


class TestCalculateSoloCompoundCombinationScore:
    """Tests for calculate_solo_compound_combination_score function."""

    def test_calculate_solo_compound_combination_score(self, solo_compound):
        """Test solo compound combination score calculation."""
        actual = calculate_solo_compound_combination_score(solo_compound, 250)
        expected = pd.DataFrame(
            {
                "Compound1": [30.0, 20.0],
                "ion": [300.0, 1.0],
                "com_score": [1.478851e08, 4.472136],
            },
            index=[300.0, 204.09],
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestCalculateAverageScoreAndDifferenceCount:
    """Tests for calculate_average_score_and_difference_count function."""

    def test_calculate_average_score_basic(self):
        """Test basic average score calculation."""
        # DataFrame with compounds as rows, m/z values as columns
        df = pd.DataFrame(
            {
                100.0: [10.0, 15.0, 20.0],
                200.0: [20.0, 25.0, 30.0],
                300.0: [30.0, 35.0, 40.0],
            },
            index=["target", "compound1", "compound2"],
        )
        ion_combination = [[100.0], [200.0]]

        result = calculate_average_score_and_difference_count(
            targeted_compound="target",
            ion_combination=ion_combination,
            df=df,
            similarity_threshold=0.5,
            fr_factor=2.0,
        )

        assert "Diff_Count" in result.columns
        assert "Similar_Compound_Ave_Score" in result.columns


class TestCalculateCombinationScore:
    """Tests for calculate_combination_score function."""

    def test_calculate_combination_score_basic(self):
        """Test basic combination score calculation."""
        # DataFrame with compounds as rows, m/z values as columns
        combination_df = pd.DataFrame(
            {"score": [1.0, 2.0]}, index=["[100.0, 200.0]", "[300.0]"]
        )
        temp_df = pd.DataFrame(
            {
                100.0: [10.0, 15.0, 20.0],
                200.0: [20.0, 25.0, 30.0],
                300.0: [30.0, 35.0, 40.0],
            },
            index=["target", "compound1", "compound2"],
        )

        result = calculate_combination_score(
            combination_df=combination_df,
            targeted_compound="target",
            temp_df=temp_df,
            prefer_mz_threshold=150,
        )

        assert "com_score" in result.columns
