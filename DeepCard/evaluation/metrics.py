"""Framework-independent metrics for binary and multiclass cardiac tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


def _labels(values: Iterable[int], classes: int) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(value not in range(classes) for value in result):
        raise ValueError(f"labels must be non-empty integers in [0, {classes - 1}]")
    return result


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    brier: float


def binary_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> BinaryMetrics:
    truth = _labels(y_true, 2)
    scores = tuple(probabilities)
    if len(truth) != len(scores) or any(not isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError("probabilities must match labels and lie in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    predicted = [int(value >= threshold) for value in scores]
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(truth, predicted))
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(truth, predicted))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(truth, predicted))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(truth, predicted))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    brier = sum((score - actual) ** 2 for score, actual in zip(scores, truth)) / len(truth)
    return BinaryMetrics(sensitivity, specificity, precision, f1, (tp + tn) / len(truth), brier)


@dataclass(frozen=True, slots=True)
class MulticlassMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_severity_distance: float
    confusion_matrix: tuple[tuple[int, ...], ...]


def multiclass_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], classes: int = 5
) -> MulticlassMetrics:
    truth = _labels(y_true, classes)
    predicted = _labels(y_pred, classes)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length")
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for actual, guess in zip(truth, predicted):
        matrix[actual][guess] += 1
    precisions, recalls, f1_values = [], [], []
    for label in range(classes):
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in range(classes) if row != label)
        fn = sum(matrix[label][column] for column in range(classes) if column != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    return MulticlassMetrics(
        accuracy=sum(actual == guess for actual, guess in zip(truth, predicted)) / len(truth),
        macro_precision=sum(precisions) / classes,
        macro_recall=sum(recalls) / classes,
        macro_f1=sum(f1_values) / classes,
        mean_severity_distance=sum(abs(actual - guess) for actual, guess in zip(truth, predicted)) / len(truth),
        confusion_matrix=tuple(tuple(row) for row in matrix),
    )


def expected_calibration_error(
    correctness: Sequence[bool | int], confidence: Sequence[float], bins: int = 10
) -> float:
    if not correctness or len(correctness) != len(confidence):
        raise ValueError("correctness and confidence must be non-empty and aligned")
    if bins <= 0 or any(not isfinite(value) or not 0 <= value <= 1 for value in confidence):
        raise ValueError("bins must be positive and confidence must lie in [0, 1]")
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(confidence) if low <= value <= high if index == bins - 1 or value < high]
        if not members:
            continue
        accuracy = sum(bool(correctness[i]) for i in members) / len(members)
        mean_confidence = sum(confidence[i] for i in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return error
