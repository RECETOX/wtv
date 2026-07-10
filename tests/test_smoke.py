import os
from pathlib import Path

import pytest

from test_data import get_test_file
from wtv.ion_selection import run_ion_selection


class TestSmoke:
    """Smoke tests for ion selection functionality."""

    @pytest.mark.skipif(
        os.getenv("GITHUB_ACTIONS") == "true", reason="Skip in Github Actions."
    )
    @pytest.mark.parametrize(
        "msp_file",
        [
            "chunk_0",
        ],
    )
    def test_smoke(self, msp_file, tmp_path):
        """Test ion selection runs successfully and produces expected output."""
        setup_output_dir = tmp_path / "output_data"
        setup_output_dir.mkdir(parents=True)

        # Run the main function
        run_ion_selection(
            msp_file_path=Path(get_test_file(msp_file)),
            output_directory=setup_output_dir,
        )

        # Compare output files with ground truth
        output_path = setup_output_dir / f"{msp_file}.msp"
        ground_truth_path = get_test_file(f"{msp_file}_filtered")

        with (
            open(output_path, "r") as output_file,
            open(ground_truth_path, "r") as ground_truth_file,
        ):
            output_lines = [line.rstrip() for line in output_file if line.strip() != ""]
            ground_truth_lines = [
                line.rstrip() for line in ground_truth_file if line.strip() != ""
            ]
            assert output_lines == ground_truth_lines
