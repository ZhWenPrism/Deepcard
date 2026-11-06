import unittest

import numpy as np

from DeepCard.evaluate import binary_metrics, bootstrap_ci, multiclass_accuracy


class RuntimeEvaluationContractTests(unittest.TestCase):
    def test_invalid_binary_inputs_are_rejected(self):
        for truth, probabilities in [([0.9, 1], [0.1, 0.9]), ([0, 2], [0.1, 0.9]),
                                     ([0, 1], [float("nan"), 0.9]), ([0, 1], [-0.1, 1.1]),
                                     ([0, 1], [[0.1], [0.9]]), ([], [])]:
            with self.subTest(truth=truth, probabilities=probabilities), self.assertRaises(ValueError):
                binary_metrics(truth, probabilities)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                binary_metrics([0, 1], [0.1, 0.9], threshold)

    def test_multiclass_distributions_are_checked_before_argmax(self):
        for truth, probabilities in [([0.5], [[1, 0, 0, 0, 0]]), ([0], [[1, 0]]),
                                     ([0], [[0, 0, 0, 0, 0]]), ([0], [[2, -1, 0, 0, 0]]),
                                     ([0], [[float("nan"), 0, 0, 0, 0]])]:
            with self.subTest(truth=truth, probabilities=probabilities), self.assertRaises(ValueError):
                multiclass_accuracy(truth, probabilities)

    def test_bootstrap_settings_and_zero_draws(self):
        for n in (-1, 0.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                bootstrap_ci([0, 1], [0.1, 0.9], n=n)
        with self.assertRaises(ValueError):
            bootstrap_ci([0, 1], [0.1, 0.9], metric="missing", n=0)
        self.assertTrue(np.isnan(bootstrap_ci([0, 1], [0.1, 0.9], n=0)).all())

    def test_valid_metrics_preserve_their_values(self):
        metrics = binary_metrics([0, 1], [0.1, 0.9])
        for name in ("sensitivity", "specificity", "precision", "accuracy", "auc"):
            self.assertEqual(metrics[name], 1.)
        self.assertEqual(multiclass_accuracy([2], [[0., 0., 1., 0., 0.]]), 1.)
