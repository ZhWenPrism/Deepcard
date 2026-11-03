import unittest
from unittest.mock import patch

import numpy as np

from DeepCard.config import MULTICLASS_TASKS, BINARY_TASKS
from DeepCard.data import EchoDataset
from DeepCard.losses import compute_class_weights
from DeepCard.train import cross_validate


def labels(n=3):
    return {task: np.zeros(n) for task in MULTICLASS_TASKS + BINARY_TASKS}


class TrainingContractTest(unittest.TestCase):
    def test_rejects_invalid_labels_and_missing_tasks(self):
        for value in (0.5, -1, 5, np.nan):
            bad = labels()
            bad[MULTICLASS_TASKS[0]][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                EchoDataset(np.zeros((3, 39)), bad)
        for change in ("binary", "length", "missing"):
            bad = labels()
            if change == "binary":
                bad[BINARY_TASKS[0]][0] = 2
            elif change == "length":
                bad[BINARY_TASKS[0]] = np.zeros(2)
            else:
                bad.pop(BINARY_TASKS[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                EchoDataset(np.zeros((3, 39)), bad)

    def test_rejects_invalid_feature_matrices(self):
        for X in (np.zeros((3, 38)), np.zeros(39), np.full((3, 39), np.nan)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                EchoDataset(X, labels())

    def test_class_weights_do_not_truncate_labels(self):
        for y in ([0, .5], [-1, 0], [0, 5], [0, np.nan], []):
            with self.subTest(y=y), self.assertRaises(ValueError):
                compute_class_weights(y, 5)

    def test_cv_validates_before_constructing_splits(self):
        bad = labels()
        bad[MULTICLASS_TASKS[0]][0] = .5
        with patch("DeepCard.train.StratifiedKFold") as split:
            with self.assertRaises(ValueError):
                cross_validate(np.zeros((3, 39)), bad)
            split.assert_not_called()

    def test_valid_data_and_weights_retain_values(self):
        ds = EchoDataset(np.arange(117).reshape(3, 39), labels())
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[1][0].tolist(), list(range(39, 78)))
        self.assertEqual(len(ds[0][1]), 17)
        np.testing.assert_allclose(compute_class_weights([0, 0, 1], 2).numpy(), [.75, 1.5])
