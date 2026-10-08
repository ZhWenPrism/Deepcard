"""Validated contracts for the 39-feature, 17-task DeepCard interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping


class TaskType(str, Enum):
    MULTICLASS = "multiclass"
    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    name: str
    task_type: TaskType
    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name must not be empty")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("task classes must contain at least two unique values")
        if self.task_type is TaskType.BINARY and len(self.classes) != 2:
            raise ValueError("binary tasks require exactly two classes")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    names: tuple[str, ...]
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.names) != 39 or len(set(self.names)) != 39:
            raise ValueError("DeepCard requires exactly 39 unique feature names")
        unknown = set(self.units) - set(self.names)
        if unknown:
            raise ValueError(f"units contain unknown features: {sorted(unknown)}")
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))


@dataclass(frozen=True, slots=True)
class ExamRecord:
    patient_id: str
    exam_id: str
    cohort: str
    features: tuple[float | None, ...]
    labels: Mapping[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("patient_id", self.patient_id), ("exam_id", self.exam_id)):
            if not value.strip() or any(ch.isspace() for ch in value):
                raise ValueError(f"{name} must be a non-empty token")
        if not self.cohort.strip():
            raise ValueError("cohort must not be empty")
        if len(self.features) != 39:
            raise ValueError("features must contain exactly 39 values")
        if any(value is not None and not isfinite(value) for value in self.features):
            raise ValueError("features must be finite or None")
        clean: dict[str, int | None] = {}
        for task, label in self.labels.items():
            if not task.strip() or (label is not None and label < 0):
                raise ValueError("labels require non-empty tasks and non-negative values")
            clean[task] = label
        object.__setattr__(self, "labels", MappingProxyType(clean))

    @property
    def missing_feature_count(self) -> int:
        return sum(value is None for value in self.features)
