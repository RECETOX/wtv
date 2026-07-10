"""Custom similarity metrics for wtv.

This module implements the original WTV similarity metrics while providing
a matchms-compatible interface through inheritance from matchms' BaseSimilarity.
"""

import re
from typing import List

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity

from wtv.similarity.core import _weighted_dot_product_distance


class WTVWeightedDotProduct(BaseSimilarity):
    """Weighted dot product similarity with fragment ratio calculation.

    This is the core WTV similarity metric that combines:
    - Weighted dot product of peak intensities (weighted by m/z and intensity)
    - Fragment Ratio (FR) calculation for peak pattern similarity

    Inherits from matchms' BaseSimilarity for compatibility with matchms infrastructure.
    """

    # Class attributes required by matchms
    is_commutative = True
    score_datatype = np.float64

    def __init__(
        self,
        fr_factor: float = 2.0,
        intensity_weight: float = 0.5,
        mz_weight: float = 2.0,
        tolerance: float = 0.1,
    ):
        """Initialize the weighted dot product calculator.

        Args:
            fr_factor: Factor used in fragment ratio calculation. Defaults to 2.0.
            intensity_weight: Exponent for intensity weighting. Defaults to 0.5.
            mz_weight: Exponent for m/z weighting. Defaults to 2.0.
            tolerance: Tolerance for peak matching in m/z units. Defaults to 0.1.
        """
        super().__init__()
        self._tolerance = tolerance
        self.fr_factor = fr_factor
        self.intensity_weight = intensity_weight
        self.mz_weight = mz_weight

    @property
    def tolerance(self) -> float:
        """Tolerance for peak matching in m/z units."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self._tolerance = value

    def pair(self, reference: Spectrum, query: Spectrum) -> np.ndarray:
        """Calculate weighted dot product similarity between two spectra.

        Args:
            reference: Reference spectrum.
            query: Query spectrum.

        Returns:
            Similarity score as numpy array.
        """
        # Convert spectra to DataFrames for the existing calculation
        ref_df = pd.DataFrame(
            {"intensities": reference.peaks.intensities},
            index=reference.peaks.mz,
        )
        query_df = pd.DataFrame(
            {"intensities": query.peaks.intensities},
            index=query.peaks.mz,
        )

        compare_df = pd.concat([ref_df, query_df], axis=1)
        compare_df = compare_df.astype(float)

        score = _weighted_dot_product_distance(
            compare_df,
            fr_factor=self.fr_factor,
            intensity_weight=self.intensity_weight,
            mz_weight=self.mz_weight,
        )

        return np.asarray(score, dtype=self.score_datatype)

    def matrix(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        array_type: str = "numpy",
        is_symmetric: bool = False,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """Calculate similarity matrix for multiple spectra.

        Uses default implementation from BaseSimilarity which iterates over pairs.

        Args:
            references: List of reference spectra.
            queries: List of query spectra.
            array_type: Output array type ("numpy" or "sparse").
            is_symmetric: Whether input is symmetric.
            progress_bar: Show progress bar.

        Returns:
            Similarity matrix.
        """
        # Use the default matrix implementation from BaseSimilarity
        return super().matrix(
            references,
            queries,
            array_type=array_type,
            is_symmetric=is_symmetric,
            progress_bar=progress_bar,
        )


def get_ion_list(index: str) -> list[float]:
    """Parse ion m/z values from a string representation.

    Args:
        index: String containing ion m/z values (e.g., "[300.0, 204.09]" or "300.0,204.09").

    Returns:
        List of float m/z values.
    """
    ion_list = re.findall(r"\d+\.?\d*", str(index))
    return [float(x) for x in ion_list]


def calculate_solo_compound_combination_score(
    matrix: pd.DataFrame, prefer_mz_threshold: float
) -> pd.DataFrame:
    """Calculate combination scores for solo compound ions.

    Args:
        matrix: DataFrame with ion data including 'ion' column and intensity column.
        prefer_mz_threshold: Threshold below which m/z values are treated as 1.

    Returns:
        DataFrame sorted by combination score (descending).
    """
    solo_scores = matrix.copy()
    solo_scores["ion"] = matrix["ion"].apply(
        lambda x: 1 if x < prefer_mz_threshold else x
    )
    solo_scores["com_score"] = solo_scores.apply(
        lambda row: pow(row.iloc[0], 0.5) * pow(row.iloc[1], 3), axis=1
    )
    return solo_scores.sort_values(by="com_score", ascending=False)


def calculate_average_score_and_difference_count(
    targeted_compound: str,
    ion_combination: list,
    df: pd.DataFrame,
    similarity_threshold: float,
    fr_factor: float,
) -> pd.DataFrame:
    """Calculate average similarity score and difference count for ion combinations.

    Args:
        targeted_compound: Name of the target compound.
        ion_combination: List of ion combinations to evaluate.
        df: DataFrame containing compound spectral data.
        similarity_threshold: Threshold for considering compounds as similar.
        fr_factor: Factor used in weighted dot product calculation.

    Returns:
        DataFrame with difference counts and average similarity scores.
    """
    difference_count_df = pd.DataFrame(
        columns=["Diff_Count", "Similar_Compound_Ave_Score"]
    )

    for ions in ion_combination:
        temp_df = df[ions]
        result_df = calculate_similarity(targeted_compound, temp_df, fr_factor)
        result_below = result_df[result_df["Score"] < similarity_threshold]
        count = len(result_below)
        difference_count_df.loc[str(ions), "Diff_Count"] = count

        result_above = result_df[result_df["Score"] >= similarity_threshold]
        if result_above.shape[0] > 0:
            ave_score = np.average(result_above, axis=0)[0]
        else:
            ave_score = 1.0
        difference_count_df.loc[str(ions), "Similar_Compound_Ave_Score"] = ave_score

    difference_count_df.sort_values(by="Diff_Count", inplace=True, ascending=False)
    return difference_count_df


def calculate_combination_score(
    combination_df: pd.DataFrame,
    targeted_compound: str,
    temp_df: pd.DataFrame,
    prefer_mz_threshold: float,
) -> pd.DataFrame:
    """Calculate combination scores for ion combinations.

    Args:
        combination_df: DataFrame containing ion combinations.
        targeted_compound: Name of the target compound.
        temp_df: Temporary DataFrame with compound spectral data.
        prefer_mz_threshold: Threshold below which m/z values are treated as 1.

    Returns:
        DataFrame with combination scores added.
    """
    for index in combination_df.index:
        ion_list = get_ion_list(index)
        new_temp_df = temp_df.loc[str(targeted_compound), ion_list].to_frame()
        new_temp_df["ion"] = new_temp_df.index.tolist()
        new_temp_df["ion"] = new_temp_df["ion"].astype(float)
        new_temp_df["ion"] = np.where(
            new_temp_df["ion"] < prefer_mz_threshold, 1.0, new_temp_df["ion"]
        )
        new_temp_df["score"] = (pow(new_temp_df["ion"], 3)) * (
            pow(new_temp_df[str(targeted_compound)], 0.5)
        )
        combination_df.loc[index, "com_score"] = new_temp_df["score"].sum()

    return combination_df


def calculate_similarity(
    target_name: str, df: pd.DataFrame, fr_factor: float
) -> pd.DataFrame:
    """Calculate similarity scores between target and all compounds in DataFrame.

    Args:
        target_name: Name of the target compound.
        df: DataFrame containing compound spectral data (compounds as rows).
        fr_factor: Factor used in weighted dot product calculation.

    Returns:
        DataFrame with similarity scores for each compound.
    """
    from wtv.similarity.core import _weighted_dot_product_distance

    result_df = pd.DataFrame(columns=["Score"])
    first_col = df.loc[target_name]

    for compound in df.index.values:
        if compound != target_name:
            second_col = df.loc[compound]
            compare_df = pd.concat([first_col, second_col], axis=1)
            compare_df = compare_df.astype(float)
            score = _weighted_dot_product_distance(compare_df, fr_factor)
            result_df.loc[compound, "Score"] = score

    return result_df


# Legacy function names for backward compatibility
def weighted_dot_product_distance(
    compare_df: pd.DataFrame, fr_factor: float
) -> float:
    """Legacy wrapper for _weighted_dot_product_distance."""
    return _weighted_dot_product_distance(compare_df, fr_factor)


def dot_product_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Legacy wrapper for core dot product distance."""
    from wtv.similarity.core import _dot_product_distance
    return _dot_product_distance(p, q)
