import os
import unittest
from pathlib import Path

from parameterized import parameterized

from test_data import get_test_file
from wtv.ion_selection import run_ion_selection as run_ion_selection


class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.msp_path = get_test_file("esi_spectra")
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.outpath = self.parent_dir / "output_data"
        self.output_files = ["filtered_ions.msp"]

        # Create output directory if it doesn't exist
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)

        # Remove output files if they exist
        for file in self.output_files:
            output_path = self.outpath / file
            if output_path.exists():
                output_path.unlink()  # Remove the file
    
    def tearDown(self):
        # Clean up output files after tests
        for file in os.listdir(self.outpath):
            filepath = self.outpath / file
            if filepath.is_file():
                filepath.unlink()


    @parameterized.expand([
        ("lcms", "esi_spectra"),
        ("gcms", "chunk_0"),
    ])
    def test_smoke(self, name, msp_file):
        # Run the Main method
        run_ion_selection(
            msp_file_path=Path(get_test_file(msp_file)),
            output_directory=self.outpath,
            mz_min=35,
            mz_max=400,
            rt_window=2.00,
            min_ion_intensity_percent=7,
            min_ion_num=2,
            prefer_mz_threshold=60,
            similarity_threshold=0.85,
            fr_factor=2,
            retention_time_max=68.80,
        )

        # Compare output files with ground truth
        output_path = self.outpath / f"{msp_file}.msp"
        ground_truth_path = get_test_file(f"{msp_file}_filtered")
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
            self.assertEqual(output_lines, ground_truth_lines)


if __name__ == "__main__":
    unittest.main()
