import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from DeepCard.config import FEATURE_ORDER
from DeepCard.interpret import shap_global_importance


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return {"task": x.sum(dim=1)}, None


class ShapAxesTest(unittest.TestCase):
    def _importance(self, values):
        class Explainer:
            def __init__(self, *args):
                pass
            def shap_values(self, *args):
                return values
        with patch.dict("sys.modules", {"shap": types.SimpleNamespace(GradientExplainer=Explainer)}):
            return shap_global_importance(DummyModel(), np.zeros((2, 39)), np.ones((2, 39)), "task")

    def test_array_outputs_keep_the_feature_axis(self):
        base = np.arange(1.0, 40.0)
        values = np.stack([np.tile(base, (2, 1)), np.tile(3 * base, (2, 1))], axis=-1)
        result = self._importance(values)
        self.assertEqual(list(result), FEATURE_ORDER)
        np.testing.assert_allclose(list(result.values()), 2 * base)

    def test_legacy_list_outputs_remain_supported(self):
        base = np.arange(1.0, 40.0)
        result = self._importance([np.tile(base, (2, 1)), np.tile(3 * base, (2, 1))])
        np.testing.assert_allclose(list(result.values()), 2 * base)

    def test_binary_matrix_outputs_remain_supported(self):
        result = self._importance(np.tile(np.arange(1.0, 40.0), (2, 1)))
        np.testing.assert_allclose(list(result.values()), np.arange(1.0, 40.0))

    def test_unaligned_outputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self._importance(np.ones((2, 2, 39)))

    def test_installed_explainer_keeps_all_feature_contributions(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(1.0, 40.0))
            def forward(self, x):
                return {"task": (x * self.weight).sum(dim=1)}, None
        result = shap_global_importance(Linear(), np.zeros((2, 39)), np.ones((2, 39)), "task")
        self.assertEqual(list(result), FEATURE_ORDER)
        np.testing.assert_allclose(list(result.values()), np.arange(1.0, 40.0), atol=1e-6)

