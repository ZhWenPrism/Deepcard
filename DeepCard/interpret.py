"""SHAP-based interpretability (global and local feature importance).

Implements the manuscript "Interpretability analysis": SHAP values decompose a
prediction into per-feature contributions; global importance is the mean
absolute SHAP value across the dataset.
"""
import numpy as np
import torch

if __package__:
    from .config import FEATURE_ORDER
    from .validation import feature_matrix
else:
    from config import FEATURE_ORDER
    from validation import feature_matrix


class _TaskWrapper(torch.nn.Module):
    """Expose a single task's logits so SHAP can explain one head at a time."""

    def __init__(self, base, task):
        super().__init__()
        self.base = base
        self.task = task

    def forward(self, x):
        out, _ = self.base(x)
        logit = out[self.task]
        return logit.unsqueeze(1) if logit.dim() == 1 else logit


def shap_global_importance(model, X_background, X_explain, task, device="cpu"):
    """Return {feature: mean_abs_SHAP} for one task using GradientExplainer."""
    background = feature_matrix(X_background)
    explain = feature_matrix(X_explain)
    import shap

    wrapper = _TaskWrapper(model, task).to(device).eval()
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    ex = torch.tensor(explain, dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapper, bg)
    values = explainer.shap_values(ex)
    sv = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if sv.shape[:2] != tuple(ex.shape) or sv.ndim not in (2, 3):
        raise ValueError("SHAP values must align with the sample and feature axes")
    axes = (0, 2) if sv.ndim == 3 else (0,)
    mean_abs = np.mean(np.abs(sv), axis=axes)
    return dict(zip(FEATURE_ORDER, mean_abs))
