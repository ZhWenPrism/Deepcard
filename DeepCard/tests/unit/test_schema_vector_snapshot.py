import unittest

from DeepCard.datasets.schema import ExamRecord, FeatureSchema, TaskDefinition, TaskType


class SchemaVectorSnapshotTest(unittest.TestCase):
    def test_mutating_caller_lists_does_not_change_validated_schema(self):
        classes = ["negative", "positive"]
        names = [f"feature_{i}" for i in range(39)]
        features = [0.0] * 39
        task = TaskDefinition("task", TaskType.BINARY, classes)
        schema = FeatureSchema(names)
        record = ExamRecord("patient", "exam", "development", features)
        classes.pop()
        names[0] = names[1]
        features[0] = float("nan")
        self.assertEqual(task.classes, ("negative", "positive"))
        self.assertEqual(schema.names[0], "feature_0")
        self.assertEqual(record.features[0], 0.0)
