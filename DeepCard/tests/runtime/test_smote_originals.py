import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from DeepCard.config import FEATURE_ORDER
from DeepCard.data import smote_oversample


class SmoteOriginalsTest(unittest.TestCase):
    def _features(self, n):
        x = np.zeros((n, 39))
        for name, value in (("LVEDD", 5), ("LVESD", 3), ("LVEDV", 100), ("LVESV", 40)):
            x[:, FEATURE_ORDER.index(name)] = value
        return x

    def test_real_resampling_preserves_every_original_row(self):
        x = self._features(14)
        x[0, FEATURE_ORDER.index("LVESD")] = 6
        y = np.array([0] * 8 + [1] * 6)
        with contextlib.redirect_stdout(io.StringIO()):
            actual_x, actual_y = smote_oversample(x, y)
        np.testing.assert_array_equal(actual_x[:len(x)], x)
        np.testing.assert_array_equal(actual_y[:len(y)], y)
        self.assertEqual(len(actual_x), 16)

    def test_only_invalid_synthetic_rows_are_filtered_and_counts_are_reported(self):
        x = self._features(2)
        x[0, FEATURE_ORDER.index("LVESD")] = 6
        bad = x[0].copy()
        augmented = np.vstack([x, x[1], bad])
        with patch("imblearn.over_sampling.SMOTE") as smote:
            smote.return_value.fit_resample.return_value = (augmented, np.array([0, 1, 1, 0]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                actual_x, actual_y = smote_oversample(x, [0, 1])
        np.testing.assert_array_equal(actual_x[:2], x)
        self.assertEqual(len(actual_y), 3)
        self.assertIn("original=2", output.getvalue())
        self.assertIn("retained_synthetic=1", output.getvalue())

