import unittest
import torch

from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS, NUM_SEVERITY
from DeepCard.losses import MultiTaskLoss


class ClassWeightWidthTest(unittest.TestCase):
    def test_wrong_width_is_rejected_at_construction(self):
        for size in (1, NUM_SEVERITY - 1, NUM_SEVERITY + 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "severity class"):
                    MultiTaskLoss(class_weights={MULTICLASS_TASKS[0]: torch.ones(size)})

    def test_valid_class_weights_keep_loss_and_gradients_finite(self):
        criterion = MultiTaskLoss(class_weights={MULTICLASS_TASKS[0]: torch.ones(NUM_SEVERITY)})
        outputs = {task: torch.zeros((2, NUM_SEVERITY), requires_grad=True) for task in MULTICLASS_TASKS}
        outputs.update({task: torch.zeros(2, requires_grad=True) for task in BINARY_TASKS})
        targets = {task: torch.zeros(2, dtype=torch.long) for task in outputs}
        loss, _ = criterion(outputs, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(output.grad).all() for output in outputs.values()))
