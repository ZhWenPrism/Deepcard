import unittest

import numpy as np
import torch

from DeepCard.config import BINARY_TASKS, MODEL, MULTICLASS_TASKS
from DeepCard.model import DeepCard
from DeepCard.train import ensemble_predict


class EnsembleDeviceTest(unittest.TestCase):
    def _model(self):
        torch.manual_seed(5)
        return DeepCard({**MODEL, "hidden_channels": 8, "num_res_blocks": 1, "attn_heads": 2, "dropout": 0.0})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_model_can_score_on_the_requested_cuda_device(self):
        result = ensemble_predict([self._model()], np.zeros((2, 39)), device="cuda")
        self.assertEqual(result[MULTICLASS_TASKS[0]].shape, (2, 5))
        self.assertEqual(result[BINARY_TASKS[0]].shape, (2,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_model_can_score_on_the_requested_cpu_device(self):
        result = ensemble_predict([self._model().to("cuda")], np.zeros((2, 39)), device="cpu")
        self.assertTrue(np.isfinite(result[MULTICLASS_TASKS[0]]).all())

    def test_empty_ensemble_is_rejected(self):
        with self.assertRaises(ValueError):
            ensemble_predict([], np.zeros((2, 39)))

    def test_cpu_probabilities_preserve_all_task_outputs(self):
        result = ensemble_predict([self._model()], np.zeros((2, 39)))
        self.assertEqual(len(result), 17)
        for task in MULTICLASS_TASKS:
            np.testing.assert_allclose(result[task].sum(axis=1), 1.0, atol=1e-6)
        for task in BINARY_TASKS:
            self.assertTrue(((result[task] >= 0) & (result[task] <= 1)).all())

