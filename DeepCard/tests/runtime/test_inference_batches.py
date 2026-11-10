import unittest

import numpy as np
import torch

from DeepCard.config import BINARY_TASKS, MODEL, MULTICLASS_TASKS
from DeepCard.model import DeepCard
from DeepCard.train import ensemble_predict


class InferenceBatchTests(unittest.TestCase):
    def test_no_forward_receives_more_rows_than_the_batch_size(self):
        class Limited(torch.nn.Module):
            def forward(self, x):
                self.assert_batch(x)
                result = {task: torch.zeros(len(x), 5, device=x.device) for task in MULTICLASS_TASKS}
                result.update({task: torch.zeros(len(x), device=x.device) for task in BINARY_TASKS})
                return result, None

            def assert_batch(self, x):
                if len(x) > 3:
                    raise AssertionError("the whole cohort reached a forward pass")

        predictions = ensemble_predict([Limited()], np.zeros((8, 39)), batch_size=3)
        self.assertEqual(predictions[BINARY_TASKS[0]].shape, (8,))

    def _compare(self, device):
        torch.manual_seed(11)
        models = [DeepCard({**MODEL, "hidden_channels": 8, "num_res_blocks": 1, "attn_heads": 2,
                            "dropout": 0.}).to(device).eval() for _ in range(2)]
        x = np.random.default_rng(2).normal(size=(7, 39)).astype(np.float32)
        with torch.no_grad():
            outputs = [model(torch.as_tensor(x, device=device))[0] for model in models]
        expected = {}
        for task in MULTICLASS_TASKS:
            expected[task] = np.mean([torch.softmax(out[task], 1).cpu().numpy() for out in outputs], axis=0)
        for task in BINARY_TASKS:
            expected[task] = np.mean([torch.sigmoid(out[task]).cpu().numpy() for out in outputs], axis=0)
        actual = ensemble_predict(models, x, device=device, batch_size=3)
        for task in expected:
            np.testing.assert_allclose(actual[task], expected[task], atol=1e-6, rtol=1e-6)

    def test_cpu_matches_full_cohort_probabilities(self):
        self._compare("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_full_cohort_probabilities(self):
        self._compare("cuda")

    def test_invalid_batch_sizes_and_features_are_rejected(self):
        model = DeepCard({**MODEL, "hidden_channels": 8, "num_res_blocks": 1, "attn_heads": 2})
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ensemble_predict([model], np.zeros((2, 39)), batch_size=size)
        with self.assertRaises(ValueError):
            ensemble_predict([model], np.full((2, 39), np.nan))
