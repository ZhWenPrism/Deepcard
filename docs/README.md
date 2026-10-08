# Documentation

Technical notes for the public DeepCard research repository.

| Document | Purpose |
|:---|:---|
| [Data interface](DATA_INTERFACE.md) | Defines the 39-feature examination schema and 17-task registry |
| [Experiment record](EXPERIMENT_RECORD.md) | Captures preprocessing, task heads, losses, thresholds, and checkpoints |
| [Evaluation metrics](METRICS.md) | Separates multiclass, binary, calibration, and external-validation results |
| [Artifact manifest](ARTIFACT_MANIFEST.md) | Links schemas, task definitions, models, SHAP tables, and figures |
| [Failure analysis](FAILURE_ANALYSIS.md) | Reviews rare tasks, ordinal errors, missingness, and cohort drift |
| [Reproducibility scope](REPRODUCIBILITY.md) | States patient-level split and clinical-data boundaries |

## Recommended order

Freeze the feature schema, task registry, and patient partitions first; record every run; report all tasks before pooled summaries; bind interpretation to the evaluated checkpoint; then examine external and low-prevalence failures.
