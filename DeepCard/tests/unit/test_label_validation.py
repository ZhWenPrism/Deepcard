import unittest

from DeepCard.datasets import ExamRecord, TaskDefinition, TaskType, validate_task_labels


class LabelValidationTests(unittest.TestCase):
    def test_exam_labels_reject_fractional_and_nonfinite_values(self):
        for label in (0.5, 1.0, float("nan"), float("inf")):
            with self.subTest(label=label), self.assertRaises(ValueError):
                ExamRecord("p1", "e1", "internal", (1.0,) * 39, {"task": label})

    def test_registry_rejects_float_class_indices(self):
        registry = (TaskDefinition("task", TaskType.BINARY, ("no", "yes")),)
        with self.assertRaises(ValueError):
            validate_task_labels({"task": 1.0}, registry)
        validate_task_labels({"task": 1}, registry)
        validate_task_labels({"task": None}, registry)
