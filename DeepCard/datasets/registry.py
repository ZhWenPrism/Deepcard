"""Canonical task registry derived from the central DeepCard configuration."""

from __future__ import annotations

from typing import Mapping

from ..config import BINARY_TASKS, MULTICLASS_TASKS, SEVERITY_CLASSES
from .schema import TaskDefinition, TaskType


def build_default_registry() -> tuple[TaskDefinition, ...]:
    multiclass = tuple(
        TaskDefinition(name, TaskType.MULTICLASS, tuple(SEVERITY_CLASSES))
        for name in MULTICLASS_TASKS
    )
    binary = tuple(
        TaskDefinition(name, TaskType.BINARY, ("negative", "positive"))
        for name in BINARY_TASKS
    )
    registry = multiclass + binary
    if len(registry) != 17 or len({task.name for task in registry}) != 17:
        raise RuntimeError("central configuration must define 17 unique tasks")
    return registry


def validate_task_labels(
    labels: Mapping[str, int | None],
    registry: tuple[TaskDefinition, ...] | None = None,
    *,
    require_all: bool = False,
) -> None:
    definitions = registry or build_default_registry()
    by_name = {task.name: task for task in definitions}
    unknown = set(labels) - set(by_name)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    if require_all and set(labels) != set(by_name):
        missing = sorted(set(by_name) - set(labels))
        raise ValueError(f"missing task labels: {missing}")
    for name, label in labels.items():
        if label is not None and label not in range(len(by_name[name].classes)):
            raise ValueError(f"label for {name!r} is outside its class range")
