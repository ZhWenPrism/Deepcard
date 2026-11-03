"""Array contracts used by the training entry points."""
from numbers import Integral

import numpy as np

if __package__:
    from .config import FEATURE_ORDER, MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY
else:
    from config import FEATURE_ORDER, MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY


def feature_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or not len(X) or X.shape[1] != len(FEATURE_ORDER) or not np.isfinite(X).all():
        raise ValueError("features must be a finite non-empty matrix with 39 columns")
    return X


def label_vector(values, num_classes, name="labels"):
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(num_classes, Integral) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, np.arange(num_classes)).all():
        raise ValueError(f"{name} must be a non-empty integer label vector in [0, {num_classes - 1}]")
    return values.astype(np.int64)


def training_labels(labels, n_rows):
    tasks = MULTICLASS_TASKS + BINARY_TASKS
    if set(labels) != set(tasks):
        raise ValueError("labels must contain the 17 configured tasks")
    result = {}
    for task in tasks:
        value = label_vector(labels[task], NUM_SEVERITY if task in MULTICLASS_TASKS else 2, task)
        if len(value) != n_rows:
            raise ValueError(f"{task} must have one label per feature row")
        result[task] = value
    return result
