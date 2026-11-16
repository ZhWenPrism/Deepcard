import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from DeepCard import run_evaluate, run_train
from DeepCard.config import BINARY_TASKS, FEATURE_ORDER, MULTICLASS_TASKS


class LabeledCSVEntryPointsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv = self.root / "cohort.csv"
        self.frame = pd.DataFrame({column: [1.0] for column in FEATURE_ORDER})
        for name in MULTICLASS_TASKS + BINARY_TASKS:
            self.frame[name] = 0

    def test_missing_labels_are_reported_by_training_loader(self):
        missing = BINARY_TASKS[0]
        self.frame.drop(columns=missing).to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, f"Missing task labels:.*{missing}"):
            run_train.load_xy(self.csv)

    def test_evaluation_checks_label_values_before_loading_artifacts(self):
        name = BINARY_TASKS[0]
        self.frame[name] = 0.5
        self.frame.to_csv(self.csv, index=False)
        with patch.object(sys, "argv", ["run_evaluate", "--test_csv", str(self.csv),
                                        "--artifacts", str(self.root / "absent")]):
            with self.assertRaisesRegex(ValueError, f"{name} must be a non-empty integer label vector"):
                run_evaluate.main()

    def test_valid_inputs_keep_feature_order_and_task_alignment(self):
        self.frame.to_csv(self.csv, index=False)
        features, labels = run_train.load_xy(self.csv)
        self.assertEqual(features.shape, (1, len(FEATURE_ORDER)))
        self.assertEqual(tuple(labels), tuple(MULTICLASS_TASKS + BINARY_TASKS))
        for column in labels.values():
            np.testing.assert_array_equal(column, [0])
