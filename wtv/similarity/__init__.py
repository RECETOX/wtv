"""Similarity calculation modules for wtv.

This module provides custom similarity metrics for mass spectrometry data,
compatible with matchms' BaseSimilarity interface.
"""

from wtv.similarity.custom import (
    WTVWeightedDotProduct,
    calculate_average_score_and_difference_count,
    calculate_combination_score,
    calculate_similarity,
    calculate_solo_compound_combination_score,
    dot_product_distance,
    get_ion_list,
    weighted_dot_product_distance,
)

__all__ = [
    "WTVWeightedDotProduct",
    "dot_product_distance",
    "weighted_dot_product_distance",
    "calculate_similarity",
    "get_ion_list",
    "calculate_average_score_and_difference_count",
    "calculate_combination_score",
    "calculate_solo_compound_combination_score",
]
