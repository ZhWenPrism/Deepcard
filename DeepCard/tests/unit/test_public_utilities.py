import unittest

from DeepCard.config import FEATURE_ORDER
from DeepCard.datasets import (
    ExamRecord,
    FeatureSchema,
    TaskType,
    build_default_registry,
    validate_task_labels,
)
from DeepCard.evaluation import (
    binary_metrics,
    expected_calibration_error,
    multiclass_metrics,
)
from DeepCard.reporting import TaskResult, aggregate_metric, summarize_results


class SchemaAndRegistryTests(unittest.TestCase):
    def test_default_registry_preserves_published_task_count(self):
        registry = build_default_registry()
        self.assertEqual(len(registry), 17)
        self.assertEqual(sum(task.task_type is TaskType.MULTICLASS for task in registry), 8)
        self.assertEqual(sum(task.task_type is TaskType.BINARY for task in registry), 9)

    def test_exam_record_tracks_missingness(self):
        schema = FeatureSchema(tuple(FEATURE_ORDER), {"LVEDD": "mm"})
        features = tuple([1.0] * 38 + [None])
        record = ExamRecord("patient-1", "exam-1", "internal", features, {})
        self.assertEqual(len(schema.names), 39)
        self.assertEqual(record.missing_feature_count, 1)

    def test_task_label_range_is_validated(self):
        registry = build_default_registry()
        binary_task = next(task for task in registry if task.task_type is TaskType.BINARY)
        with self.assertRaises(ValueError):
            validate_task_labels({binary_task.name: 2}, registry)


class MetricTests(unittest.TestCase):
    def test_binary_metrics_include_calibration_error(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
        self.assertAlmostEqual(metrics.sensitivity, 1.0)
        self.assertAlmostEqual(metrics.specificity, 0.5)
        self.assertGreater(metrics.brier, 0.0)

    def test_multiclass_metrics_preserve_ordinal_distance(self):
        metrics = multiclass_metrics([0, 1, 2, 2], [0, 2, 2, 1], classes=3)
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.mean_severity_distance, 0.5)
        self.assertEqual(sum(sum(row) for row in metrics.confusion_matrix), 4)

    def test_expected_calibration_error(self):
        value = expected_calibration_error([1, 0], [0.9, 0.8], bins=2)
        self.assertAlmostEqual(value, 0.35)


class ReportingTests(unittest.TestCase):
    def test_macro_and_weighted_aggregation_remain_distinct(self):
        results = (
            TaskResult("task-a", TaskType.BINARY, 10, {"f1": 1.0}),
            TaskResult("task-b", TaskType.BINARY, 30, {"f1": 0.0}),
        )
        self.assertAlmostEqual(aggregate_metric(results, "f1"), 0.5)
        self.assertAlmostEqual(aggregate_metric(results, "f1", weighted=True), 0.25)
        summary = summarize_results(results)
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["total_support"], 40)


if __name__ == "__main__":
    unittest.main()
