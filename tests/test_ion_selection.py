import unittest

import pandas as pd

from wtv.ion_selection import get_nearby_compounds


class TestIonSelection(unittest.TestCase):
    def setUp(self):
        self.matrix = pd.DataFrame(
            {
                "Compound1": [10, 20, 30, 0, 0, 0],
                "Compound2": [0, 0, 0, 15, 25, 35],
                "Compound3": [10, 0, 30, 0, 25, 0],
            },
            index=[100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            dtype=float
        ).T

        self.rt_data = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
            },
            index=["Compound1", "Compound2", "Compound3"],
        )
    
    def test_get_nearby_compounds(self):
        actual = get_nearby_compounds(2.0, self.rt_data)
        expected = {
            "Compound1": ["Compound1", "Compound2"],
            "Compound2": ["Compound1", "Compound2"],
            "Compound3": ["Compound3"]
        }

        assert actual == expected