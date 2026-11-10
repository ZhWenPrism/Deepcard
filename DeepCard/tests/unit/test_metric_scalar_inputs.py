import unittest

from DeepCard.evaluation.metrics import binary_metrics, multiclass_metrics


class MetricScalarInputsTest(unittest.TestCase):
    def test_integer_valued_labels_use_valid_confusion_matrix_indices(self):
        result = multiclass_metrics([0.0, 3.0], [1.0, 0.0])
        self.assertEqual(result.mean_severity_distance, 2.0)
        self.assertEqual(result.confusion_matrix[0][1], 1)
        self.assertEqual(result.confusion_matrix[3][0], 1)

    def test_invalid_class_counts_raise_value_error(self):
        for classes in (True, 1, 2.5):
            with self.subTest(classes=classes), self.assertRaisesRegex(ValueError, "classes"):
                multiclass_metrics([0], [0], classes=classes)

    def test_unsigned_scores_preserve_squared_probability_error(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        with np.errstate(over="raise"):
            result = binary_metrics(np.array([1, 0], dtype=np.uint8), np.array([0, 1], dtype=np.uint8))
        self.assertEqual(result.brier, 1.0)
