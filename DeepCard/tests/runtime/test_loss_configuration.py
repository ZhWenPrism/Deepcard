import unittest
import torch
from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS
from DeepCard.losses import MultiTaskLoss


class LossConfigurationTest(unittest.TestCase):
    def test_unusable_weights_cannot_produce_nan_or_reverse_the_loss(self):
        for weight in (torch.zeros(5), torch.ones(2, 2), torch.tensor([1., -1.]),
                       torch.tensor([float("nan")]), torch.empty(0)):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "class weights"):
                MultiTaskLoss(class_weights={MULTICLASS_TASKS[0]: weight})
        for weight in (torch.tensor([-1.]), torch.tensor([float("inf")]), torch.ones(2)):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "positive-class weights"):
                MultiTaskLoss(pos_weights={BINARY_TASKS[0]: weight})
        for weights in ({"typo": 1.}, {BINARY_TASKS[0]: -1.},
                        {BINARY_TASKS[0]: float("nan")},
                        {t: 0. for t in MULTICLASS_TASKS + BINARY_TASKS}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MultiTaskLoss(task_weights=weights)

    def test_configured_task_weights_are_snapshotted(self):
        weights = {BINARY_TASKS[0]: 2.}
        loss = MultiTaskLoss(task_weights=weights)
        weights[BINARY_TASKS[0]] = -2.
        self.assertEqual(loss.task_weights[BINARY_TASKS[0]], 2.)
