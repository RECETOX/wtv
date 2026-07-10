"""Core similarity calculation functions.

This module contains the core weighted dot product calculation that is used
by both the custom metrics and the score functions, avoiding circular imports.
"""

import numpy as np
import pandas as pd


def _dot_product_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the dot product distance between two vectors.

    Args:
        p: First vector.
        q: Second vector.

    Returns:
        Dot product distance between the two vectors (0 if either is empty).
    """
    if np.sum(p) == 0 or np.sum(q) == 0:
        return 0.0
    return np.power(np.sum(q * p), 2) / (
        np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
    )


def _weighted_dot_product_distance(
    compare_df: pd.DataFrame,
    fr_factor: float,
    intensity_weight: float = 0.5,
    mz_weight: float = 2.0,
) -> float:
    """Calculate weighted dot product distance with fragment ratio.

    Args:
        compare_df: DataFrame with two columns representing spectra to compare.
        fr_factor: Factor used in fragment ratio calculation.
        intensity_weight: Exponent for intensity weighting.
        mz_weight: Exponent for m/z weighting.

    Returns:
        Composite score based on weighted dot product and fragment ratio.
    """
    m_q = pd.Series(compare_df.index).astype(float)
    i_q = np.array(compare_df.iloc[:, 0])
    i_r = np.array(compare_df.iloc[:, 1])

    # Apply weights to intensities and m/z values
    w_q = np.power(i_q, intensity_weight) * np.power(m_q, mz_weight)
    w_r = np.power(i_r, intensity_weight) * np.power(m_q, mz_weight)

    ss = _dot_product_distance(w_q, w_r)

    # Calculate Fragment Ratio (FR) for shared peaks
    shared_spec = np.vstack((i_q, i_r))
    shared_spec = pd.DataFrame(shared_spec)
    shared_spec = shared_spec.loc[:, (shared_spec != 0).all(axis=0)]
    m = shared_spec.shape[1]

    if m >= fr_factor:
        FR = 0.0
        for i in range(1, m):
            s = (shared_spec.iat[0, i] / shared_spec.iat[0, i - 1]) * (
                shared_spec.iat[1, i - 1] / shared_spec.iat[1, i]
            )
            if s > 1:
                s = 1 / s
            FR = FR + s
        ave_FR = FR / (m - 1)
        NU = len(compare_df)
        composite_score = ((NU * ss) + (m * ave_FR)) / (NU + m)
    else:
        composite_score = ss

    return composite_score
