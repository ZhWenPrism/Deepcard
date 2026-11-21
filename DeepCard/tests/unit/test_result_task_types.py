import unittest

from DeepCard.datasets.schema import TaskType
from DeepCard.reporting.task_report import TaskResult


class ResultTaskTypesTest(unittest.TestCase):
    def test_invalid_task_types_are_rejected(self):
        for value in (None, "", "survival", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "task_type"):
                    TaskResult("task", value, 1, {"accuracy": 0.5})

    def test_serialized_types_are_normalized(self):
        for value in ("binary", "multiclass"):
            self.assertIs(TaskResult("task", value, 1, {"accuracy": 0.5}).task_type, TaskType(value))
