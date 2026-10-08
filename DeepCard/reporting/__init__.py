"""Structured task-level result aggregation."""

from .task_report import TaskResult, aggregate_metric, summarize_results

__all__ = ["TaskResult", "aggregate_metric", "summarize_results"]
