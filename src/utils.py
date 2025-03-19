import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_msp


def read_msp(msp_file_path: str) -> Tuple[Dict[str, Dict[int, int]], pd.DataFrame]:
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
    spectra = load_from_msp(msp_file_path)
    meta = {}
    rt_data = []
    for spectrum in spectra:
        if spectrum is None:
            continue  # Skip empty spectra
        name = spectrum.metadata.get("compound_name")
        retention_time = spectrum.metadata.get("retention_time")
        if retention_time is None:
            raise ValueError(
                "Retention time is missing in spectra metadata. Specifically compund with name: ",
                name,
            )
        rt_data.append({"Name": name, "RT": retention_time})
        ion_intens_dic = {}
        for mz, intensity in zip(spectrum.mz, spectrum.intensities):
            key = float(mz)
            value = int(intensity)
            if key in ion_intens_dic:
                ion_intens_dic[key] = max(ion_intens_dic[key], value)
            else:
                ion_intens_dic[key] = value
        meta[name] = ion_intens_dic
    df = pd.DataFrame(rt_data).set_index("Name")
    return meta, df


def write_msp(ion_df: pd.DataFrame, output_directory: Path, source_msp_file: Path) -> None:
    spectra = load_from_msp(source_msp_file)
    grouped_ions = ion_df.groupby("Name")
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
    save_as_msp(filtered_spectra, output_directory/"filtered_msp.msp")




class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
