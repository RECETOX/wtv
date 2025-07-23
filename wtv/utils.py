from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.exporting.metadata_export import get_metadata_as_array
from matchms.importing import load_from_msp



def read_msp(msp_file_path: str, retention: str = 'retention_time') -> Tuple[Dict[str, Dict[float, int]], pd.DataFrame]:
    """
    Read data from an MSP file and convert it into a dictionary format using matchms.
    Also, create a DataFrame with columns 'Name' and 'RT'.

    Args:
        msp_file (str): The path to the MSP file.

    Returns:
        Tuple[Dict[str, Dict[int, int]], pd.DataFrame]: A tuple containing:
            - A dictionary where keys are compound names and values are dictionaries of ion intensities.
            - A DataFrame with columns 'Name' and 'RT'.
    """
    spectra = list(load_from_msp(msp_file_path, metadata_harmonization=True))
    meta = {}
    # rt_data = []
    for spectrum in spectra:
        if spectrum is None:
            continue  # Skip empty spectra
        name = spectrum.metadata.get("compound_name")
        ion_intens_dic = {}
        for mz, intensity in zip(spectrum.mz, spectrum.intensities):
            key = float(mz)
            value = int(intensity)
            ion_intens_dic[key] = value
        meta[name] = ion_intens_dic

    spectra_md, _ = get_metadata_as_array(spectra)
    df = pd.DataFrame(spectra_md).rename(columns={'compound_name':'Name', retention: 'RT'}).get(["Name", "RT"])
    df.set_index("Name", inplace=True)
    return meta, df


def write_msp(
    ion_df: pd.DataFrame, output_directory: Path, source_msp_file: Path
) -> None:
    spectra = load_from_msp(source_msp_file)
    grouped_ions = ion_df.groupby(ion_df.index)
    filtered_spectra = []  # List to store filtered spectra
    for spectrum in spectra:
        if spectrum is None:
            continue  # Skip empty spectra
        name = spectrum.metadata.get("compound_name")
        ions = grouped_ions.get_group(name)
        mzs_to_keep = ions["ion"].values
        mask = np.isin(spectrum.peaks.mz, mzs_to_keep)
        # Apply the filter
        filtered_mz = spectrum.peaks.mz[mask]
        filtered_intensities = spectrum.peaks.intensities[mask]
        # Create a new filtered spectrum (or update the existing one)
        filtered_spectrum = Spectrum(
            mz=filtered_mz, intensities=filtered_intensities, metadata=spectrum.metadata
        )
        # Add the filtered spectrum to the list
        filtered_spectra.append(filtered_spectrum)
    filtered_msp_path = str(output_directory / "filtered_ions.msp")
    save_as_msp(filtered_spectra, filtered_msp_path)

def create_ion_matrix(mz_min, mz_max, meta_1):
    """Create a matrix of ions with compounds on the rows, mz values for ions on the columns, and ion intensities as values.

    Args:
        mz_min (float): Minimum m/z value.
        mz_max (float): Maximum m/z value.
        meta_1 (dict): Dictionary where keys are compound names and values are dictionaries of ion intensities.

    Returns:
        pd.DataFrame: DataFrame with compounds as rows, ions as columns, and ion intensities as values.
    """
    # Build DataFrame directly from meta_1
    matrix = pd.DataFrame.from_dict(meta_1, orient="index").fillna(0)

    # Filter columns to keep only m/z values within range, assuming all are float
    valid_cols = [col for col in matrix.columns if mz_min <= float(col) <= mz_max]
    matrix = matrix[valid_cols]

    return matrix


def filter_and_sort_combinations(
    combination_df: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    """
    Filter and sort combinations in a DataFrame based on a score column.

    Args:
        combination_df (pd.DataFrame): DataFrame containing combinations and their scores.
        score_column (str): The name of the column containing scores to filter and sort by.

    Returns:
        pd.DataFrame: A filtered and sorted DataFrame based on the score column.
    """
    # Sort the DataFrame by the specified score column in ascending order
    combination_df = combination_df.sort_values(
        by=score_column, inplace=False, ascending=True
    )

    # Filter the DataFrame to include only rows with scores greater than or equal to the minimum score
    return combination_df[combination_df[score_column] >= combination_df.iat[0, 1]]

def check_rt_data(RT_data):
    duplicated_index = RT_data.index[RT_data.index.duplicated()]
    for index in duplicated_index:
        logger.error(f"Duplicated RT for index: {index}")

    # RT_data.drop(duplicated_index, axis=0, inplace=True)

    for index_1, row in RT_data.iterrows():
        if (
            type(row.iloc[0]) not in [float, int, np.float64, np.int64]
            or pd.isna(row.iloc[0])
        ):
            logger.error(f"RT format error for index: {index_1}")
            # RT_data.drop(index=index_1, inplace=True)

def average_rts_for_duplicated_indices(RT_data):
    """
    Average the retention times for duplicated indices in the RT data.

    Args:
        RT_data (pd.DataFrame): DataFrame containing retention time data.

    Returns:
        pd.DataFrame: DataFrame with averaged retention times for duplicated indices.
    """
    RT_data = RT_data.groupby(RT_data.index).mean()
    return RT_data