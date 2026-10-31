import unittest

from DeepCard.datasets import TaskType
from DeepCard.reporting import TaskResult, aggregate_metric, summarize_results


class SupportValidationTests(unittest.TestCase):
    def test_invalid_support_cannot_contaminate_weighted_results(self):
        for support in (1.5, float("nan"), float("inf"), True, 0, -1):
            with self.subTest(support=support), self.assertRaises(ValueError):
                TaskResult("task", TaskType.BINARY, support, {"f1": 0.5})

    def test_valid_support_preserves_weighted_aggregation(self):
        results = (
            TaskResult("a", TaskType.BINARY, 3, {"f1": 0.0}),
            TaskResult("b", TaskType.BINARY, 1, {"f1": 1.0}),
        )
        self.assertEqual(aggregate_metric(results, "f1", weighted=True), 0.25)
        self.assertEqual(summarize_results(results)["total_support"], 4)
