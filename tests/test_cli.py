import subprocess
import unittest
from pathlib import Path

from test_data import get_test_file


class TestCLI(unittest.TestCase):

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

    def test_cli_call(self):
        # Simulate CLI call using poetry to ensure correct environment
        command = [
            "poetry", "run", "python",
            "-m",
            "src.cli",
            "--msp_path",
            str(self.msp_path),
            "--outpath",
            str(self.outpath),
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
        self.assertEqual(
            result.returncode, 0, msg=f"CLI call failed with error: {result.stderr}"
        )

        # Compare output files with ground truth
        for file in self.output_files:
            output_path = self.outpath / file
            ground_truth_path = get_test_file(file[:-4])
            with open(output_path, "r") as output_file, open(ground_truth_path, "r") as ground_truth_file:
                self.assertEqual(output_file.read(), ground_truth_file.read())


if __name__ == "__main__":
    unittest.main()
