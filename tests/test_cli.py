import os
import subprocess

import pytest

from test_data import get_test_file


@pytest.fixture
def setup_output_dir(tmp_path):
    """Create output directory for CLI tests."""
    outpath = tmp_path / "output_data"
    outpath.mkdir(parents=True)
    yield outpath


@pytest.fixture
def msp_path():
    """Return the test MSP file path."""
    return get_test_file("esi_spectra")


class TestCLI:
    """Test CLI functionality."""

    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skip in Github Actions.")
    def test_cli_call(self, setup_output_dir, msp_path):
        """Test CLI invocation and verify output matches ground truth."""
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "wtv.cli",
            "--msp_path",
            str(msp_path),
            "--outpath",
            str(setup_output_dir),
            "--mz_min",
            "35",
            "--mz_max",
            "400",
            "--rt_window",
            "2.00",
            "--min_ion_intensity_percent",
            "7",
            "--min_ion_num",
            "2",
            "--prefer_mz_threshold",
            "60",
            "--similarity_threshold",
            "0.85",
            "--fr_factor",
            "2",
            "--retention_time_max",
            "68.80",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI call failed with error: {result.stderr}"

        # The output file is named after the input file (esi_spectra.msp)
        output_files = ["esi_spectra.msp"]
        for file in output_files:
            output_path = setup_output_dir / file
            ground_truth_path = get_test_file(file.replace(".msp", "_filtered"))
            with (
                open(output_path, "r") as output_file,
                open(ground_truth_path, "r") as ground_truth_file,
            ):
                output_lines = [
                    line.rstrip() for line in output_file if line.strip() != ""
                ]
                ground_truth_lines = [
                    line.rstrip() for line in ground_truth_file if line.strip() != ""
                ]
                assert output_lines == ground_truth_lines
