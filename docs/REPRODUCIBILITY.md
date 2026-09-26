# Reproducibility scope

DeepCard operates on pre-measured echocardiographic parameters. It does not perform image acquisition, view selection, or automated caliper placement.

## Cohort protocol

- Preserve the patient-level 400/100 development and internal-test separation.
- Keep the independent 102-patient external cohort isolated from model selection.
- Fit preprocessing statistics using training data only.

## Multi-task evaluation

- Report all 17 endpoints rather than only pooled averages.
- Separate multiclass severity tasks from binary diagnostic tasks.
- Include sensitivity, specificity, precision, F1, accuracy, and calibration where applicable.
- Generate SHAP summaries per task using the same feature schema as inference.

## Determinism controls

Record seeds, package versions, split manifests, task definitions, loss weights, and the selected checkpoint for every reported run.

## Data boundary

Patient-level measurements, identifiers, private split files, and trained checkpoints must not be committed to the public repository.
