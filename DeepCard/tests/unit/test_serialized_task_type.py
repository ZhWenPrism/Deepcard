import unittest

from DeepCard.datasets.schema import TaskDefinition, TaskType


class SerializedTaskTypeTest(unittest.TestCase):
    def test_valid_strings_are_normalized(self):
        binary = TaskDefinition("Example", "binary", ("no", "yes"))
        self.assertIs(binary.task_type, TaskType.BINARY)

    def test_unrecognized_type_and_invalid_binary_width_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "task_type"):
            TaskDefinition("Example", "ordinal", ("low", "high"))
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            TaskDefinition("Example", "binary", ("no", "maybe", "yes"))
