import unittest

import torch

from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS, NUM_SEVERITY
from DeepCard.losses import MultiTaskLoss


class LossOutputDtypeTests(unittest.TestCase):
    def test_weighted_loss_supports_double_precision_logits(self):
        outputs = {
            **{t: torch.zeros(2, NUM_SEVERITY, dtype=torch.float64, requires_grad=True) for t in MULTICLASS_TASKS},
            **{t: torch.zeros(2, dtype=torch.float64, requires_grad=True) for t in BINARY_TASKS},
        }
        targets = {t: torch.tensor([0, 1]) for t in outputs}
        class_weights = {t: torch.arange(1, NUM_SEVERITY + 1, dtype=torch.float32) for t in MULTICLASS_TASKS}
        pos_weights = {t: torch.tensor(2.) for t in BINARY_TASKS}
        objective = MultiTaskLoss(class_weights=class_weights, pos_weights=pos_weights)

        loss, logs = objective(outputs, targets)
        loss.backward()

        self.assertEqual(loss.dtype, torch.float64)
        self.assertAlmostEqual(float(loss.detach()), sum(logs.values()))
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in outputs.values()))
