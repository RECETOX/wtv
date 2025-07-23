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
    matrix = pd.DataFrame()
    for name, dic in meta_1.items():
        b = {name: dic}
        y = pd.DataFrame.from_dict(b).T
        matrix = pd.concat([matrix, y], axis=0, join="outer").fillna(0)

    for col_name in list(matrix):
        if (
            type(col_name) not in [float, int, np.float64, np.int64]
            and not col_name.isdigit()  # Fixed equality comparison
        ):
            matrix.drop(columns=col_name, inplace=True)

    for ion in list(matrix):
        if int(ion) < mz_min or int(ion) > mz_max:
            matrix.drop(columns=ion, inplace=(True))
    return matrix

def filter_matrix(
    matrix: pd.DataFrame, compound: str, min_ion_intensity: float
) -> pd.DataFrame:
    """
    Filter the matrix for a specific compound based on minimum ion intensity.

    Args:
        matrix (pd.DataFrame): The DataFrame containing compound data.
        compound (str): The name of the compound to filter.
        min_ion_intensity (float): The minimum ion intensity threshold.

    Returns:
        pd.DataFrame: A filtered DataFrame containing ions with intensity above the threshold.
    """
    # Extract the compound data as a DataFrame
    matrix_1 = matrix.loc[compound].to_frame()

    # Add a column for ion values converted to integers
    matrix_1["ion"] = matrix_1.index.astype(int)

    # Apply the minimum ion intensity threshold
    matrix_1[compound] = np.where(
        matrix_1[compound].astype(float) < min_ion_intensity,
        0,
        matrix_1[compound].astype(float),
    )

    # Return the filtered DataFrame with ions above the threshold
    return matrix_1.loc[matrix_1[compound] > 0, :]


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