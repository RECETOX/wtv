from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_msp

from test_data import get_test_file
from wtv.utils import (
    average_rts_for_duplicated_indices,
    create_ion_matrix,
    get_filtered_spectra,
    parse_spectra,
    read_msp,
    write_msp,
)


@pytest.fixture
def test_msp_file(tmp_path):
    """Create a mock MSP file for testing."""
    test_msp_path = tmp_path / "test_spectra.msp"
    output_directory = tmp_path / "output"
    output_directory.mkdir(exist_ok=True)

    mock_spectra = [
        Spectrum(
            mz=np.array([100, 200, 300], dtype=float),
            intensities=np.array([10, 20, 30], dtype=float),
            metadata={"compound_name": "Compound1", "retention_time": 5.0},
        ),
        Spectrum(
            mz=np.array([150, 250, 350], dtype=float),
            intensities=np.array([15, 25, 35], dtype=float),
            metadata={"compound_name": "Compound2", "retention_time": 10.0},
        ),
    ]
    save_as_msp(mock_spectra, str(test_msp_path))

    yield test_msp_path, output_directory, mock_spectra


class TestUtils:
    """Test utility functions."""

    def test_read_msp(self, test_msp_file):
        """Test reading MSP file."""
        msp_path, _, _ = test_msp_file
        meta, rt_data = read_msp(msp_path)
        assert isinstance(meta, dict)
        assert isinstance(rt_data, pd.DataFrame)
        assert len(meta) == 2
        assert rt_data.shape == (2, 1)
        assert "Compound1" in meta
        assert "Compound2" in meta
        assert rt_data.loc["Compound1", "RT"] == 5.0
        assert rt_data.loc["Compound2", "RT"] == 10.0

    def test_write_msp(self, test_msp_file):
        """Test writing MSP file."""
        msp_path, output_directory, _ = test_msp_file

        # Create a mock ion DataFrame
        ion_df = pd.DataFrame(
            {
                "RT": [5.0, 10.0],
                "ion": [100, 150],
            },
            index=["Compound1", "Compound2"],
        )

        write_msp(ion_df, output_directory, msp_path)

        # Verify the output MSP file (uses same name as source)
        output_msp_path = output_directory / "test_spectra.msp"
        assert output_msp_path.exists()

        spectra = list(load_from_msp(output_msp_path))
        assert len(spectra) == 2
        assert len(spectra[0].peaks.mz) == 1
        assert len(spectra[1].peaks.mz) == 1
        assert spectra[0].peaks.mz[0] == 100
        assert spectra[1].peaks.mz[0] == 150

    def test_create_ion_matrix(self):
        """Test creating ion matrix."""
        meta = {
            "Compound1": {100.0: 10, 200.0: 20, 300.0: 30},
            "Compound2": {150.0: 15, 250.0: 25, 350.0: 35},
        }

        expected = pd.DataFrame(
            {"Compound1": [10, 20, 30, 0, 0, 0], "Compound2": [0, 0, 0, 15, 25, 35]},
            index=[100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            dtype=float,
        ).T

        actual = create_ion_matrix(50, 400, meta)
        assert actual.equals(expected)

    # @pytest.mark.skip(reason="Requires large test data file")
    def test_create_ion_matrix_2(self):
        """Test creating ion matrix with large dataset."""
        meta, _ = read_msp(Path(get_test_file("ei_spectra")))
        actual = create_ion_matrix(70, 800, meta)
        assert np.count_nonzero(actual) == 1609

    def test_average_rts_for_duplicated_indices(self):
        """Test averaging RTs for duplicated indices."""
        rt_data = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
            },
            index=["Compound1", "Compound1", "Compound2"],
        )
        expected = pd.DataFrame(
            {
                "RT": [5.5, 10.0],
            },
            index=["Compound1", "Compound2"],
        )
        actual = average_rts_for_duplicated_indices(rt_data)
        assert actual.equals(expected)

    def test_parse_spectra(self, test_msp_file):
        """Test parsing spectra."""
        _, _, mock_spectra = test_msp_file
        actual = parse_spectra(mock_spectra)
        assert actual is not None


class TestWriteFilteredSpectra:
    """Test filtered spectra functionality."""

    @pytest.fixture
    def setup_spectra(self):
        """Set up test spectra and combinations."""
        spectra = [
            Spectrum(
                mz=np.array([100, 200, 300], dtype=float),
                intensities=np.array([10, 20, 30], dtype=float),
                metadata={"compound_name": "Compound1", "retention_time": 5.0},
            ),
            Spectrum(
                mz=np.array([150, 250, 350], dtype=float),
                intensities=np.array([15, 25, 35], dtype=float),
                metadata={"compound_name": "Compound2", "retention_time": 10.0},
            ),
        ]

        combinations = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
                "Ion_Combination": [
                    [300.0, 204.09],
                    [300.0, 350.0],
                    [300.0, 250.0],
                ],
                "Note": [np.nan, np.nan, np.nan],
                "Similar_Compound_List": [["Compound2"], [], []],
                "SCL_Note": [np.nan, np.nan, "No adjacent compounds."],
            },
            index=["Compound1", "Compound2", "Compound3"],
            dtype=object,
        )

        return spectra, combinations

    def test_get_filtered_spectra(self, setup_spectra):
        """Test getting filtered spectra."""
        spectra, combinations = setup_spectra
        actual = list(get_filtered_spectra(spectra, combinations))
        assert len(actual) == 2
