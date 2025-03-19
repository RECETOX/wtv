import unittest
from pathlib import Path

import pandas as pd

from src.ion_selection import main as method_generator
from src.utils import read_msp
from test_data import get_test_file


class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.msp_path = get_test_file("esi_spectra")
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.outpath = self.parent_dir / "output_data"

        # Create output directory if it doesn't exist
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)

    def test_smoke(self):
        processed_msp_data = read_msp(self.msp_path)
        self.assertIsInstance(processed_msp_data, tuple)

        # Run the Main method
        method_generator(
            processed_msp_data=processed_msp_data,
            mz_min=35,
            mz_max=400,
            output_directory=str(self.outpath),
            rt_window=2.00,
            min_ion_intensity_percent=7,
            min_ion_num=2,
            prefer_mz_threshold=60,
            similarity_threshold=0.85,
            fr_factor=2,
            retention_time_max=68.80,
            solvent_delay=0.00,
            sim_sig_max=99,
            min_dwell_time=10,
            point_per_s=2.0,
        )

        # Compare output files with ground truth
        output_files = [
            "input_data_error_info.csv",
            "ion_rt_data.csv",
            "SIM_seg_result.csv",
        ]
        for file in output_files:
            output_df = pd.read_csv(self.outpath / file)
            file_name = file[:-4]
            ground_truth_path = get_test_file(file_name)
            ground_truth_df = pd.read_csv(ground_truth_path)
            pd.testing.assert_frame_equal(output_df, ground_truth_df, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
