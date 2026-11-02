import os
import re
from pathlib import Path

import pytest

from test_data import get_test_file
from wtv.ion_selection import run_ion_selection
from wtv.utils import MissingRetentionDataError


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


class TestSmokeRetentionIndex:
    """End to end ion selection on GC data carrying only a retention index."""

    def test_run_with_retention_index(self, msp_without_retention_time, tmp_path):
        """Ion selection completes when only retention index is available."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_ion_selection(
            msp_file_path=msp_without_retention_time,
            output_directory=output_dir,
            mz_min=35.0,
            mz_max=400.0,
            min_ion_num=2,
            retention_field="retention_index",
        )

        output_path = output_dir / msp_without_retention_time.name
        assert output_path.exists()
        content = output_path.read_text()
        assert "2-Chlorobiphenyl" in content
        # Retention indices must be preserved in the written library.
        assert re.search(r"RETENTION_?INDEX:\s*15\d\d", content, re.I)

    def test_run_without_retention_time_fails_clearly(
        self, msp_without_retention_time, tmp_path
    ):
        """The default retention time field must not fail silently."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(MissingRetentionDataError) as excinfo:
            run_ion_selection(
                msp_file_path=msp_without_retention_time,
                output_directory=output_dir,
                mz_min=35.0,
                mz_max=400.0,
                min_ion_num=2,
            )
        assert "--retention_field retention_index" in str(excinfo.value)
        assert not (output_dir / msp_without_retention_time.name).exists()
