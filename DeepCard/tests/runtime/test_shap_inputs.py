import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from DeepCard.interpret import shap_global_importance


class ShapInputTest(unittest.TestCase):
    def test_wrong_width_is_rejected_before_constructing_an_explainer(self):
        for bg, ex in ((np.ones((2, 38)), np.ones((2, 39))),
                       (np.ones((2, 39)), np.ones((2, 38)))):
            with self.subTest(background=bg.shape, explain=ex.shape):
                def explainer(*_):
                    self.fail("invalid inputs reached the explainer")

                shap = types.SimpleNamespace(GradientExplainer=explainer)
                with patch.dict("sys.modules", {"shap": shap}):
                    with self.assertRaisesRegex(ValueError, "39 columns"):
                        shap_global_importance(torch.nn.Identity(), bg, ex, "task")
