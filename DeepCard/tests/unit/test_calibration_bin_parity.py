import math
import unittest

from DeepCard.evaluation.metrics import expected_calibration_error


class CalibrationBinParityTest(unittest.TestCase):
    def test_boundary_values_match_direct_bin_membership(self):
        for bins in (1, 3, 10, 19):
            edges = [i / bins for i in range(1, bins)]
            probabilities = [0.0, 1.0] + [
                value
                for edge in edges
                for value in (math.nextafter(edge, 0), edge, math.nextafter(edge, 1))
            ]
            truth = [int(i % 3 == 0) for i in range(len(probabilities))]
            expected = 0.0
            for index in range(bins):
                lo, hi = index / bins, (index + 1) / bins
                pairs = [(hit, value) for hit, value in zip(truth, probabilities)
                         if lo <= value < hi or (index == bins - 1 and value == 1)]
                if pairs:
                    hits, confidence = map(sum, zip(*pairs))
                    count = len(pairs)
                    expected += count / len(truth) * abs(hits / count - confidence / count)
            self.assertAlmostEqual(expected_calibration_error(truth, probabilities, bins), expected, places=15)
