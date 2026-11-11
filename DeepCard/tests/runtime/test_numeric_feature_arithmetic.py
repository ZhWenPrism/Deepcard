import unittest

import numpy as np
import pandas as pd

from DeepCard.feature_engineering import add_derived_features


class NumericFeatureArithmeticTests(unittest.TestCase):
    def test_derived_indices_preserve_values_for_integer_measurements(self):
        raw = pd.DataFrame({
            "LVEDV": np.array([120, 110], dtype=np.uint8),
            "LVESV": np.array([125, 40], dtype=np.uint8),
            "LVEDD": np.array([5, 6], dtype=np.int8),
            "IVSd": np.array([1, 2], dtype=np.int8),
            "LVPWd": np.array([1, 2], dtype=np.int8),
            "MV_E": np.array([100, 120], dtype=np.uint8),
            "e_septal": np.array([130, 129], dtype=np.uint8),
            "e_lateral": np.array([131, 130], dtype=np.uint8),
            "AV_Vmax": np.array([13, 15], dtype=np.int8),
            "TR_Vmax": np.array([13, 15], dtype=np.int8),
            "RAP": np.array([5, 6], dtype=np.int8),
        })
        actual = add_derived_features(raw)
        expected = add_derived_features(raw.astype(float))
        for column in expected.columns.difference(raw.columns):
            with self.subTest(column=column):
                np.testing.assert_allclose(actual[column], expected[column])
        pd.testing.assert_frame_equal(actual[raw.columns], raw)

    def test_nullable_measurements_keep_missing_values(self):
        raw = pd.DataFrame({"AV_Vmax": pd.array([2, pd.NA], dtype="Int8")})
        actual = add_derived_features(raw)
        self.assertEqual(actual["AV_peak_grad"].iloc[0], 16.0)
        self.assertTrue(pd.isna(actual["AV_peak_grad"].iloc[1]))
