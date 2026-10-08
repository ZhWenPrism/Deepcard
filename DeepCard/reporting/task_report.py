"""Safe aggregation that preserves task identity and support."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

from ..datasets.schema import TaskType


@dataclass(frozen=True, slots=True)
class TaskResult:
    task: str
    task_type: TaskType
    support: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip() or self.support <= 0:
            raise ValueError("task must be named and support must be positive")
        if not self.metrics or any(
            not name.strip() or not isfinite(value) for name, value in self.metrics.items()
        ):
            raise ValueError("metrics must contain finite named values")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def aggregate_metric(
    results: Iterable[TaskResult], metric: str, *, weighted: bool = False
) -> float:
    items = tuple(result for result in results if metric in result.metrics)
    if not items:
        raise ValueError(f"metric {metric!r} is unavailable")
    weights = [result.support if weighted else 1 for result in items]
    return sum(result.metrics[metric] * weight for result, weight in zip(items, weights)) / sum(weights)


def summarize_results(results: Iterable[TaskResult]) -> Mapping[str, object]:
    items = tuple(results)
    if not items or len({item.task for item in items}) != len(items):
        raise ValueError("results must contain unique tasks")
    metric_names = sorted(set.intersection(*(set(item.metrics) for item in items)))
    return MappingProxyType(
        {
            "task_count": len(items),
            "total_support": sum(item.support for item in items),
            "macro": {name: aggregate_metric(items, name) for name in metric_names},
            "weighted": {name: aggregate_metric(items, name, weighted=True) for name in metric_names},
            "tasks": tuple(items),
        }
    )
