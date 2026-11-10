import unittest

from DeepCard.evaluation.metrics import expected_calibration_error


class CalibrationVectorTest(unittest.TestCase):
    def test_iterable_inputs_preserve_bin_edges(self):
        self.assertAlmostEqual(expected_calibration_error(iter([1, 0]), iter([1.0, 0.5]), bins=2), 0.25)

    def test_nonbinary_outcomes_and_invalid_bin_counts_are_rejected(self):
        for args in (([2], [0.8], 2), ([0], [0.8], True), ([0], [0.8], 2.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                expected_calibration_error(*args)
