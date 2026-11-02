"""Utility functions for wtv package.

This module provides I/O operations, data loading, and helper functions
for mass spectrometry data processing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.exporting.metadata_export import get_metadata_as_array
from matchms.importing import load_from_msp

logger = logging.getLogger(__name__)

# Retention metadata fields understood by get_rt_data.
RETENTION_FIELDS = ("retention_time", "retention_index")

# Starting point for rt_window when retention indices are used. Retention indices
# span hundreds to thousands of units, so a window carried over from a retention
# time in minutes would match almost no neighbouring compounds. Tune this for your
# own gradient and index variation.
RT_INDEX_WINDOW_DEFAULT = 20.0


def resolve_rt_window(
    rt_window: float | None, retention_field: str, rt_default: float
) -> float:
    """Pick the retention window to use when the caller left it unset.

    Retention times and retention indices live on very different scales, so the
    window cannot have a single scale-free default. An explicitly supplied
    ``rt_window`` is always honoured unchanged.

    Args:
        rt_window: User supplied window, or None to select a default.
        retention_field: Field used for retention data.
        rt_default: Default for retention times; differs between the CLI and the
            library API for backwards compatibility.

    Returns:
        The window to apply, in the units of ``retention_field``.
    """
    if rt_window is not None:
        return rt_window
    if retention_field == "retention_index":
        logger.warning(
            "No rt_window given; using %.1f retention index units. Retention indices "
            "are not on a minute scale, so review this value against the index spread "
            "of your data.",
            RT_INDEX_WINDOW_DEFAULT,
        )
        return RT_INDEX_WINDOW_DEFAULT
    return rt_default


def normalize_array(array: np.ndarray, desired_max: float = 100.0) -> np.ndarray:
    """Normalize an array to a desired maximum value.

    Args:
        array: Input array to normalize.
        desired_max: Desired maximum value after normalization. Defaults to 100.

    Returns:
        Normalized array with values scaled to desired_max.
    """
    actual_max = np.max(array)
    if actual_max == 0:
        return array
    zero_to_one_array = array / actual_max
    return zero_to_one_array * desired_max


def read_msp(
    msp_file_path: Path,
    retention_field: str = "retention_time",
    mz_min: float = 0,
    mz_max: float = np.inf,
) -> Tuple[Dict[str, Dict[float, int]], pd.DataFrame]:
    """Read data from an MSP file using matchms.

    Args:
        msp_file_path: Path to the MSP file.
        retention_field: Field name for retention data. Use "retention_time" or
            "retention_index". Defaults to "retention_time".
        mz_min: Minimum m/z value to include. Defaults to 0.
        mz_max: Maximum m/z value to include. Defaults to infinity.

    Returns:
        Tuple containing:
            - Dictionary of compound names to ion intensity dictionaries.
            - DataFrame with compound names and retention values.
    """
    spectra = list(load_from_msp(msp_file_path, metadata_harmonization=False))
    meta = get_ion_dict(spectra, mz_min, mz_max)
    df = get_rt_data(retention_field, spectra)
    return meta, df


class MissingRetentionDataError(ValueError):
    """Raised when the selected retention field carries no usable values."""


def missing_retention_error(
    retention_field: str, available_fields: pd.Index
) -> MissingRetentionDataError:
    """Build an actionable error for unusable retention data.

    Args:
        retention_field: Field the user asked to use for retention data.
        available_fields: Metadata fields actually present in the spectra.

    Returns:
        MissingRetentionDataError naming the problem and, when possible, the field
        to switch to.
    """
    available = set(available_fields)
    alternatives = [
        field
        for field in RETENTION_FIELDS
        if field != retention_field and field in available
    ]
    message = f"No usable '{retention_field}' value found in any spectrum."
    if alternatives:
        suggestion = alternatives[0]
        message += (
            f" Found '{suggestion}' instead - GC-MS libraries frequently provide a "
            f"retention index rather than a retention time. Rerun with "
            f"'--retention_field {suggestion}' (or retention_field='{suggestion}' "
            f"when calling the library directly) to use it."
        )
    else:
        message += (
            f" No alternative retention field is available; metadata fields present: "
            f"{sorted(available)}."
        )
    return MissingRetentionDataError(message)


def get_rt_data(retention_field: str, spectra: list[Spectrum]) -> pd.DataFrame:
    """Extract retention data from spectra.

    Args:
        retention_field: Field name for retention data (e.g., "retention_time",
            "retention_index").
        spectra: List of Spectrum objects.

    Returns:
        DataFrame with compound names as index and retention values.

    Raises:
        MissingRetentionDataError: If the requested retention field is absent from
            the metadata or holds no numeric value for any spectrum.
        ValueError: If the spectra carry no 'compound_name' metadata.
    """
    spectra_md, _ = get_metadata_as_array(spectra)
    df = pd.DataFrame(spectra_md)
    if retention_field not in df.columns:
        raise missing_retention_error(retention_field, df.columns)
    df = df.rename(columns={"compound_name": "Name", retention_field: "RT"})
    if "Name" not in df.columns:
        raise ValueError(
            "No 'compound_name' metadata found in spectra; retention data is "
            "matched to compounds by name."
        )
    df = df[["Name", "RT"]]
    df.set_index("Name", inplace=True)
    # Convert RT column to float, handling string values
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    if not df["RT"].notna().any():
        raise missing_retention_error(retention_field, df.columns)
    return df


def get_ion_dict(
    spectra: list[Spectrum], mz_min: float = 0, mz_max: float = np.inf
) -> Dict[str, Dict[float, int]]:
    """Extract ion intensities from spectra as a dictionary.

    Args:
        spectra: List of Spectrum objects.
        mz_min: Minimum m/z value to include. Defaults to 0.
        mz_max: Maximum m/z value to include. Defaults to infinity.

    Returns:
        Dictionary mapping compound names to ion intensity dictionaries.
    """
    meta: Dict[str, Dict[float, int]] = {}

    for spectrum in spectra:
        if spectrum is None:
            continue

        name = spectrum.metadata.get("compound_name")
        if name is None:
            continue

        ion_intens_dic: Dict[float, int] = {}

        # Filter by m/z range
        mz_mask = (spectrum.peaks.mz >= mz_min) & (spectrum.peaks.mz <= mz_max)
        filtered_mz = spectrum.peaks.mz[mz_mask]
        filtered_intensities = spectrum.peaks.intensities[mz_mask]

        intensities = normalize_array(filtered_intensities)
        for mz, intensity in zip(filtered_mz, intensities, strict=False):
            key = float(mz)
            value = int(intensity)
            ion_intens_dic[key] = value

        meta[name] = ion_intens_dic

    return meta


def load_data(
    msp_file_path: Path,
    mz_min: float,
    mz_max: float,
    retention_field: str = "retention_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data from an MSP file.

    Args:
        msp_file_path: Path to the MSP file.
        mz_min: Minimum m/z value.
        mz_max: Maximum m/z value.
        retention_field: Field name for retention data. Defaults to "retention_time".

    Returns:
        Tuple of (RT_data, matrix) where:
            - RT_data: DataFrame with retention times/indices.
            - matrix: DataFrame with ion intensities.
    """
    meta_1, RT_data = read_msp(msp_file_path, retention_field, mz_min, mz_max)
    matrix = create_ion_matrix(mz_min, mz_max, meta_1)

    RT_data = average_rts_for_duplicated_indices(RT_data)
    check_rt_data(RT_data)
    RT_data = RT_data.sort_values(by="RT")

    return RT_data, matrix


def write_msp(
    ion_df: pd.DataFrame, output_directory: Path, source_msp_file: Path
) -> None:
    """Write filtered spectra to MSP file.

    Args:
        ion_df: DataFrame with ion combinations.
        output_directory: Directory for output file.
        source_msp_file: Path to source MSP file.
    """
    spectra = list(load_from_msp(source_msp_file, metadata_harmonization=False))
    grouped_ions = ion_df.groupby(ion_df.index)
    filtered_spectra = []

    for compound, ions in grouped_ions:
        matching_spectra = [
            x for x in spectra if x.metadata.get("compound_name") == compound
        ]
        if not matching_spectra:
            logger.warning(f"No spectrum found for compound: {compound}")
            continue

        spectrum: Spectrum = matching_spectra[0]

        mzs_to_keep = ions["ion"].values
        mask = np.isin(spectrum.peaks.mz, mzs_to_keep)
        filtered_mz = spectrum.peaks.mz[mask]
        filtered_intensities = spectrum.peaks.intensities[mask]

        filtered_spectrum = Spectrum(
            mz=filtered_mz,
            intensities=filtered_intensities,
            metadata=spectrum.metadata,
        )
        filtered_spectra.append(filtered_spectrum)

    filtered_msp_path = str(output_directory / os.path.basename(source_msp_file))
    save_as_msp(filtered_spectra, filtered_msp_path)


def get_filtered_spectra(
    spectra: list[Spectrum], combinations: pd.DataFrame
) -> Generator[Spectrum, None, None]:
    """Generate filtered spectra based on ion combinations.

    Args:
        spectra: List of original Spectrum objects.
        combinations: DataFrame with ion combinations.

    Yields:
        Filtered Spectrum objects with only selected ions.
    """
    for spectrum in spectra:
        group = spectrum.get("compound_name")
        if group not in combinations.index:
            continue

        ions = combinations.loc[group, "Ion_Combination"]
        if ions != "NA" and ions is not None:
            # Handle both list and string formats
            if isinstance(ions, str):
                from wtv.similarity import get_ion_list

                ion_list = get_ion_list(ions)
            else:
                ion_list = [float(x) for x in ions]

            mask = np.isin(spectrum.peaks.mz, ion_list)
            yield Spectrum(
                mz=spectrum.peaks.mz[mask],
                intensities=spectrum.peaks.intensities[mask],
                metadata=spectrum.metadata,
            )


def create_ion_matrix(mz_min: float, mz_max: float, meta_1: dict) -> pd.DataFrame:
    """Create a matrix of ion intensities.

    Args:
        mz_min: Minimum m/z value.
        mz_max: Maximum m/z value.
        meta_1: Dictionary of compound ion intensities.

    Returns:
        DataFrame with compounds as rows, m/z values as columns.
    """
    matrix = pd.DataFrame.from_dict(meta_1, orient="index").fillna(0)

    valid_cols = [col for col in matrix.columns if mz_min <= float(col) <= mz_max]
    matrix = matrix[valid_cols]

    return matrix


def filter_and_sort_combinations(
    combination_df: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    """Filter and sort combinations based on score column.

    Args:
        combination_df: DataFrame with combinations and scores.
        score_column: Name of column to use for filtering/sorting.

    Returns:
        Filtered and sorted DataFrame.
    """
    combination_df = combination_df.sort_values(
        by=score_column, inplace=False, ascending=True
    )
    # Get the minimum score from the first row of the specified column
    min_score = combination_df[score_column].iloc[0]
    return combination_df[combination_df[score_column] >= min_score]


def check_rt_data(RT_data: pd.DataFrame) -> None:
    """Check RT data for issues.

    Args:
        RT_data: DataFrame with retention data.
    """
    duplicated_index = RT_data.index[RT_data.index.duplicated()]
    for index in duplicated_index:
        logger.error(f"Duplicated RT for index: {index}")

    for index_1, row in RT_data.iterrows():
        rt_value = row.iloc[0]
        if not isinstance(rt_value, (float, int, np.floating, np.integer)) or pd.isna(
            rt_value
        ):
            logger.error(f"RT format error for index: {index_1}")


def average_rts_for_duplicated_indices(RT_data: pd.DataFrame) -> pd.DataFrame:
    """Average retention times for duplicated compound names.

    Args:
        RT_data: DataFrame with retention data.

    Returns:
        DataFrame with averaged retention times.
    """
    return RT_data.groupby(RT_data.index).mean()


def parse_spectra(
    spectra: list[Spectrum],
    retention_field: str = "retention_time",
    mz_min: float = 0,
    mz_max: float = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse spectra into ion matrix and retention data.

    Args:
        spectra: List of Spectrum objects.
        retention_field: Field name for retention data. Defaults to "retention_time".
        mz_min: Minimum m/z value. Defaults to 0.
        mz_max: Maximum m/z value. Defaults to 1000.

    Returns:
        Tuple of (matrix, rt_data).
    """
    rt_data = average_rts_for_duplicated_indices(get_rt_data(retention_field, spectra))
    check_rt_data(rt_data)
    rt_data.sort_values(by="RT", inplace=True)

    meta = get_ion_dict(spectra, mz_min, mz_max)
    matrix = create_ion_matrix(mz_min, mz_max, meta)

    return matrix, rt_data
