"""Dataset, fixed feature ordering, augmentation, and SMOTE oversampling."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if __package__:
    from .config import FEATURE_ORDER, MULTICLASS_TASKS, BINARY_TASKS, TRAIN
    from .validation import feature_matrix, training_labels
else:
    from config import FEATURE_ORDER, MULTICLASS_TASKS, BINARY_TASKS, TRAIN
    from validation import feature_matrix, training_labels


class EchoDataset(Dataset):
    """Standardized feature matrix + per-task labels.

    Multi-class task labels are encoded 0..4 (the five severity grades);
    binary task labels are encoded 0/1. See DATA_FORMAT.md for the schema.
    """

    def __init__(self, X, labels):
        X = feature_matrix(X)
        labels = training_labels(labels, len(X))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = {t: torch.tensor(v) for t, v in labels.items()}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        y = {t: v[i] for t, v in self.labels.items()}
        return self.X[i], y


def order_features(df: pd.DataFrame) -> np.ndarray:
    """Return the feature matrix arranged in the fixed clinical 1-D order."""
    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    duplicates = df.columns[df.columns.duplicated()]
    duplicates = [f for f in FEATURE_ORDER if f in duplicates]
    if duplicates:
        raise ValueError(f"Duplicate required features: {duplicates}")
    return df[FEATURE_ORDER].to_numpy(dtype=float)


def add_gaussian_noise(x: torch.Tensor, std: float = TRAIN["gaussian_noise_std"]):
    """Augmentation: inject Gaussian noise to simulate measurement uncertainty."""
    return x + torch.randn_like(x) * std


def _physiologically_valid(row, feat_index) -> bool:
    """Rule-based filter for synthetic SMOTE samples.

    Discards anatomically/hemodynamically impossible instances, e.g. the
    end-systolic dimension/volume must be smaller than the end-diastolic one.
    """
    try:
        if row[feat_index["LVESD"]] >= row[feat_index["LVEDD"]]:
            return False
        if row[feat_index["LVESV"]] >= row[feat_index["LVEDV"]]:
            return False
    except KeyError:
        pass
    return True


def smote_oversample(X, y, random_state=42):
    """SMOTE oversampling for one label vector, followed by a rule-based
    physiological-plausibility filter. TRAINING DATA ONLY.

    The manuscript applies SMOTE exclusively to the training split and never to
    the internal test set or the external validation cohort.
    """
    from imblearn.over_sampling import SMOTE
    feat_index = {f: i for i, f in enumerate(FEATURE_ORDER)}
    X, y = np.asarray(X), np.asarray(y)
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
    n_original = len(X)
    synthetic_keep = np.array([_physiologically_valid(r, feat_index) for r in X_res[n_original:]], dtype=bool)
    keep = np.concatenate([np.ones(n_original, dtype=bool), synthetic_keep])
    print(f"SMOTE: original={n_original}, synthetic={len(synthetic_keep)}, retained_synthetic={int(synthetic_keep.sum())}")
    return X_res[keep], y_res[keep]
