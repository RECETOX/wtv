import unittest
import pandas as pd
from pathlib import Path
from src.ion_selection import GetMethod
from test_data import get_test_file

class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.msp_path = get_test_file('msp_file')
        self.rt_data_path = get_test_file('rt_data')
        self.name_list_path = get_test_file('name_list')
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.outpath = self.parent_dir / "output_data"

        # Create output directory if it doesn't exist
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)

        # Initialize GetMethod class
        self.method_generator = GetMethod()

    def test_smoke(self):
        # Run the Main method
        self.method_generator.Main(
            msp_path=self.msp_path,
            rt_data_path=self.rt_data_path,
            set_name_list=False,
            name_list_path=self.name_list_path,
            mz_min=35,
            mz_max=400,
            outpath=str(self.outpath),
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
            convert_to_ag_method=False,
        )

        # Compare output files with ground truth
        output_files = ["combination_results.csv", "input_data_error_info.csv", "ion_rt_data.csv", "SIM_seg_result.csv"]
        for file in output_files:
            output_df = pd.read_csv(self.outpath / file)
            file_name = file[:-4]
            ground_truth_path = get_test_file(file_name)
            ground_truth_df = pd.read_csv(ground_truth_path)
            pd.testing.assert_frame_equal(output_df, ground_truth_df, check_dtype=False)

if __name__ == "__main__":
    unittest.main()