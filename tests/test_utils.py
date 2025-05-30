import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp
from wtv.utils import read_msp, write_msp


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.test_msp_path = Path("test_data/test_spectra.msp")
        self.output_directory = Path("test_data/output")
        self.output_directory.mkdir(exist_ok=True)

        # Create a mock MSP file for testing
        self.mock_spectra = [
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
        save_path = str(self.test_msp_path)
        save_as_msp(self.mock_spectra, save_path)

    def tearDown(self):
        # Clean up test files
        if self.test_msp_path.exists():
            self.test_msp_path.unlink()
        for file in self.output_directory.iterdir():
            file.unlink()
        self.output_directory.rmdir()

    def test_read_msp(self):
        meta, rt_data = read_msp(self.test_msp_path)
        self.assertIsInstance(meta, dict)
        self.assertIsInstance(rt_data, pd.DataFrame)
        self.assertEqual(len(meta), 2)
        self.assertEqual(rt_data.shape, (2, 1))
        self.assertIn("Compound1", meta)
        self.assertIn("Compound2", meta)
        self.assertEqual(rt_data.loc["Compound1", "RT"], 5.0)
        self.assertEqual(rt_data.loc["Compound2", "RT"], 10.0)

    def test_write_msp(self):
        # Create a mock ion DataFrame
        ion_df = pd.DataFrame(
            {
                "RT": [5.0, 10.0],
                "ion": [100, 150],
            },
            index=["Compound1", "Compound2"],
        )

        write_msp(ion_df, self.output_directory, self.test_msp_path)

        # Verify the output MSP file
        output_msp_path = self.output_directory / "filtered_ions.msp"
        self.assertTrue(output_msp_path.exists())

        spectra = list(load_from_msp(output_msp_path))
        self.assertEqual(len(spectra), 2)
        self.assertEqual(len(spectra[0].peaks.mz), 1)
        self.assertEqual(len(spectra[1].peaks.mz), 1)
        self.assertEqual(spectra[0].peaks.mz[0], 100)
        self.assertEqual(spectra[1].peaks.mz[0], 150)


if __name__ == "__main__":
    unittest.main()
