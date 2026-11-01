import unittest
import warnings
import numpy as np

from DeepCard.evaluate import binary_metrics, bootstrap_ci


class BootstrapMetricsTest(unittest.TestCase):
    def test_single_class_accuracy_keeps_its_bootstrap_replicates(self):
        self.assertEqual(bootstrap_ci([1, 1, 1], [0.9, 0.8, 0.7], metric="accuracy", n=20), (1.0, 1.0))

    def test_single_class_auc_remains_undefined(self):
        self.assertTrue(all(np.isnan(v) for v in bootstrap_ci([1, 1], [0.8, 0.9], n=20)))

    def test_single_class_point_metrics_do_not_emit_auc_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = binary_metrics([1, 1], [0.8, 0.9])
        self.assertTrue(np.isnan(result["auc"]))

    def test_resampling_is_reproducible(self):
        a = bootstrap_ci([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], metric="accuracy", n=40, seed=5)
        b = bootstrap_ci([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], metric="accuracy", n=40, seed=5)
        self.assertEqual(a, b)

