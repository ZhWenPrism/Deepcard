import unittest
from unittest.mock import patch
import numpy as np
from DeepCard import train
from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS


class CVTrainingBudgetTest(unittest.TestCase):
    def test_invalid_budgets_fail_before_splits_or_model_construction(self):
        labels = {t: np.zeros(10, dtype=int) for t in MULTICLASS_TASKS + BINARY_TASKS}
        for name, value in (("epochs", 0), ("epochs", True), ("batch_size", 0),
                            ("cosine_T_max", 0), ("n_folds", 1), ("lr", float("nan")),
                            ("weight_decay", -1.), ("gaussian_noise_std", -1.)):
            with self.subTest(name=name), patch.dict(train.TRAIN, {name: value}):
                with patch.object(train, "StratifiedKFold") as split, patch.object(train, "DeepCard") as model:
                    with self.assertRaisesRegex(ValueError, name):
                        train.cross_validate(np.zeros((10, 39)), labels)
                    split.assert_not_called()
                    model.assert_not_called()

    def test_a_small_valid_budget_trains_every_fold(self):
        labels = {t: np.zeros(10, dtype=int) for t in MULTICLASS_TASKS + BINARY_TASKS}
        model = {**train.MODEL, "hidden_channels": 8, "attn_heads": 2, "num_res_blocks": 1}
        with patch.dict(train.MODEL, model), patch.dict(train.TRAIN, {"n_folds": 2, "epochs": 1, "batch_size": 5}):
            folds = train.cross_validate(np.arange(390).reshape(10, 39) / 390, labels)
        self.assertEqual(len(folds), 2)
        self.assertTrue(all(m.stem[1].num_batches_tracked.item() == 1 for m in folds))
