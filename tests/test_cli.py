import unittest
from pathlib import Path
from test_data import get_test_file
import subprocess

class TestCLI(unittest.TestCase):

    def setUp(self):
        self.msp_path = get_test_file('esi_spectra')
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.outpath = self.parent_dir / "output_data"

        # Create output directory if it doesn't exist
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)

    def test_cli_call(self):
        # Simulate CLI call
        command = [
            "python", "-m", "src.ion_selection",
            "--msp_path", str(self.msp_path), "msp",
            "--outpath", str(self.outpath),
            "--mz_min", "35",
            "--mz_max", "400",
            "--rt_window", "2.00",
            "--min_ion_intensity_percent", "7",
            "--min_ion_num", "2",
            "--prefer_mz_threshold", "60",
            "--similarity_threshold", "0.85",
            "--fr_factor", "2",
            "--retention_time_max", "68.80",
            "--solvent_delay", "0.00",
            "--sim_sig_max", "99",
            "--min_dwell_time", "10",
            "--point_per_s", "2.0"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"CLI call failed with error: {result.stderr}")

if __name__ == "__main__":
    unittest.main()
