import unittest
from unittest.mock import patch

import numpy as np

from DeepCard import evaluate


class BootstrapWorkTest(unittest.TestCase):
    def test_auc_bootstrap_preserves_draws_and_skips_extra_confusion_matrices(self):
        truth = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([.1, .8, .4, .3, .9, .4])
        rng = np.random.default_rng(11)
        values = []
        for _ in range(64):
            sample = rng.integers(0, len(truth), len(truth))
            if len(np.unique(truth[sample])) == 2:
                values.append(evaluate.binary_metrics(truth[sample], scores[sample])["auc"])
        expected = np.percentile(values, [2.5, 97.5])

        with patch.object(evaluate, "confusion_matrix", wraps=evaluate.confusion_matrix) as matrix:
            actual = evaluate.bootstrap_ci(truth, scores, n=64, seed=11)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(matrix.call_count, 1)
