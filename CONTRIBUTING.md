# Contributing

Contributions should preserve the published 39-measurement, 17-task formulation and keep clinical data outside the repository.

## Suitable contributions

- Feature-schema validation, preprocessing, training, evaluation, and reporting improvements.
- Task-wise calibration, confidence-interval, and interpretability utilities.
- Tests and documentation for the existing lightweight implementation.
- Corrections that keep repository claims aligned with the published article.

## Clinical safeguards

- Do not commit patient records, identifiers, institution-specific mappings, or trained checkpoints.
- Keep patient-level splits fixed and the external cohort isolated from model selection.
- Preserve the distinction between eight multiclass and nine binary endpoints.
- Report task-level results before pooled summaries; state the averaging rule and uncertainty.

## Pull requests

Use a focused branch and describe the affected tasks, feature-schema changes, validation cohort, and tests performed. Any change to label definitions or reported metrics must be explicit.
