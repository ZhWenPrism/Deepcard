import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from DeepCard.interpret import shap_global_importance


class Model(torch.nn.Module):
    def forward(self, values):
        return {"task": values.sum(dim=1)}, None


class ShapFiniteContributionsTest(unittest.TestCase):
    def importance(self, values):
        class Explainer:
            def __init__(self, *args):
                pass
            def shap_values(self, *args):
                return values
        with patch.dict("sys.modules", {"shap": types.SimpleNamespace(GradientExplainer=Explainer)}):
            return shap_global_importance(Model(), np.zeros((2, 39)), np.ones((2, 39)), "task")

    def test_invalid_contributions_do_not_become_nan_importance(self):
        for values in (np.full((2, 39), np.nan), np.full((2, 39, 1), np.inf),
                       np.zeros((2, 39, 0))):
            with self.subTest(shape=values.shape), self.assertRaisesRegex(ValueError, "finite contributions"):
                self.importance(values)

    def test_finite_contributions_keep_the_existing_aggregation(self):
        values = np.stack([np.ones((2, 39)), -3 * np.ones((2, 39))], axis=-1)
        np.testing.assert_allclose(list(self.importance(values).values()), 2)
