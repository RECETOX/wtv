import unittest
from pathlib import Path

from wtv.ion_selection import main as run_ion_selection
from test_data import get_test_file


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

    def test_smoke(self):
        # Run the Main method
        run_ion_selection(
            msp_file_path=self.msp_path,
            output_directory=str(self.outpath),
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
        for file in self.output_files:
            output_path = self.outpath / file
            ground_truth_path = get_test_file(file[:-4])
            with (
                open(output_path, "r") as output_file,
                open(ground_truth_path, "r") as ground_truth_file,
            ):
                output_lines = [line.rstrip() for line in output_file if line.strip() != ""]
                ground_truth_lines = [line.rstrip() for line in ground_truth_file if line.strip() != ""]
                self.assertEqual(output_lines, ground_truth_lines)


if __name__ == "__main__":
    unittest.main()
