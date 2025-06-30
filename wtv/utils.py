from pathlib import Path
from typing import Dict, Tuple

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
