import unittest

import numpy as np

from DeepCard.datasets import TaskType
from DeepCard.reporting import TaskResult, summarize_results


class ReportSupportWidthTests(unittest.TestCase):
    def test_numpy_counts_are_summed_without_fixed_width_overflow(self):
        count = np.int64(2**62)
        results = (
            TaskResult("a", TaskType.BINARY, count, {"f1": 0.0}),
            TaskResult("b", TaskType.BINARY, count, {"f1": 1.0}),
        )
        summary = summarize_results(results)
        self.assertEqual(summary["total_support"], 2**63)
        self.assertEqual(summary["weighted"]["f1"], 0.5)
