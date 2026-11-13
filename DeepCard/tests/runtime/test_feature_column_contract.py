import unittest

import numpy as np
import pandas as pd

from DeepCard.config import FEATURE_ORDER
from DeepCard.data import order_features


class FeatureColumnContractTest(unittest.TestCase):
    def test_duplicate_required_columns_do_not_expand_the_feature_matrix(self):
        columns = FEATURE_ORDER + [FEATURE_ORDER[0]]
        frame = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
        with self.assertRaisesRegex(ValueError, "Duplicate required features:.*LVEDD"):
            order_features(frame)

    def test_unrelated_duplicate_columns_are_ignored(self):
        columns = FEATURE_ORDER[::-1] + ["unused", "unused"]
        frame = pd.DataFrame([list(range(len(columns)))], columns=columns)
        np.testing.assert_array_equal(order_features(frame), [list(range(len(FEATURE_ORDER) - 1, -1, -1))])
