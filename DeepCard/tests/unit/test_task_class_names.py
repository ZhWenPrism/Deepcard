import unittest
from DeepCard.datasets.schema import TaskDefinition, TaskType


class TaskClassNamesTest(unittest.TestCase):
    def test_invalid_class_names_are_rejected(self):
        for names in (("", "positive"), (" ", "positive"), ("normal", " normal"),
                      (1, 2), (None, "positive")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                TaskDefinition("task", TaskType.BINARY, names)

    def test_valid_names_preserve_their_order(self):
        task = TaskDefinition("task", TaskType.BINARY, ["absent", "present"])
        self.assertEqual(task.classes, ("absent", "present"))
