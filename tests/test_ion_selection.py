import unittest

import numpy as np
import pandas as pd

from wtv.ion_selection import (
    filter_matrix,
    generate_ion_combinations,
    get_ion_rt,
    get_nearby_compound_ions,
    get_nearby_compounds,
)


class TestIonSelection(unittest.TestCase):
    def setUp(self):
        self.matrix = pd.DataFrame(
            {
                "Compound1": [10, 20, 30, 0, 0, 0],
                "Compound2": [0, 10, 20, 15, 25, 35],
                "Compound3": [10, 0, 30, 0, 25, 0],
            },
            index=[127.8, 204.09, 300.0, 150.0, 250.0, 350.0],
            dtype=float,
        ).T

        self.rt_data = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
            },
            index=["Compound1", "Compound2", "Compound3"],
        )

        self.combination_results = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
                "Ion_Combination": [
                    list([300.0, 204.09]),
                    "[300.0, 350.0]",
                    list([300.0, 250.0]),
                ],
                "Note": [np.nan, np.nan, np.nan],
                "Similar_Compound_List": [["Compound2"], [], []],
                "SCL_Note": [np.nan, np.nan, "No adjacent compounds."],
            },
            index=["Compound1", "Compound2", "Compound3"],
            dtype=object,
        )

    def test_get_nearby_compounds(self):
        actual = get_nearby_compounds(2.0, self.rt_data)
        expected = {
            "Compound1": ["Compound1", "Compound2"],
            "Compound2": ["Compound1", "Compound2"],
            "Compound3": ["Compound3"],
        }

        assert actual == expected

    def test_filter_matrix_single(self):
        actual = filter_matrix(self.matrix, "Compound1", 15)
        expected = pd.DataFrame(
            {"Compound1": [20.0, 30.0], "ion": [204.09, 300.0]}, index=[204.09, 300.0]
        )
        pd.testing.assert_frame_equal(actual, expected)
        assert isinstance(actual, pd.DataFrame)

    def test_filter_matrix_multiple(self):
        actual = filter_matrix(self.matrix, ["Compound1", "Compound2"], 20)
        expected = pd.DataFrame(
            {
                "Compound1": [20.0, 30.0, 0.0, 0.0],
                "Compound2": [0.0, 20.0, 25.0, 35.0],
                "ion": [204.09, 300.0, 250.0, 350.0],
            },
            index=[204.09, 300.0, 250.0, 350.0],
        )

        pd.testing.assert_frame_equal(actual, expected)

    def test_get_nearby_compound_ions(self):
        actual = get_nearby_compound_ions(
            self.matrix, 20, "Compound1", ["Compound1", "Compound2"]
        )
        expected = pd.DataFrame(
            {
                204.09: [20.0, 10.0],
                300.0: [30.0, 20.0],
            },
            index=["Compound1", "Compound2"],
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_generate_ion_combinations(self):
        actual = generate_ion_combinations(
            min_ion_intensity_percent=7,
            min_ion_num=2,
            prefer_mz_threshold=60,
            similarity_threshold=0.85,
            fr_factor=2,
            RT_data=self.rt_data,
            matrix=self.matrix,
            rt_window=2.0,
        )

        expected = pd.DataFrame(
            {
                "RT": [5.0, 6.0, 10.0],
                "Ion_Combination": [
                    list([300.0, 204.09]),
                    "[300.0, 350.0]",
                    list([300.0, 250.0]),
                ],
                "Note": [np.nan, np.nan, np.nan],
                "Similar_Compound_List": [["Compound2"], [], []],
                "SCL_Note": [np.nan, np.nan, "No adjacent compounds."],
            },
            index=["Compound1", "Compound2", "Compound3"],
            dtype=object,
        )

        pd.testing.assert_frame_equal(actual, self.combination_results)

    def test_get_ion_rt(self):
        actual = get_ion_rt(100, self.rt_data, self.combination_results)
        expected = pd.DataFrame(
            {
                "RT": [5.0, 5.0, 6.0, 6.0, 10.0, 10.0],
                "ion": [300.0, 204.0, 300.0, 350.0, 300.0, 250.0],
            },
            index=[
                "Compound1",
                "Compound1",
                "Compound2",
                "Compound2",
                "Compound3",
                "Compound3",
            ],
        )
        pd.testing.assert_frame_equal(actual, expected)
