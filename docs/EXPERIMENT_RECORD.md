# Experiment Record

Use one immutable record for every DeepCard training or evaluation run.

## Identity

| Field | Value |
|:---|:---|
| Experiment ID |  |
| Git revision |  |
| 39-feature schema version |  |
| Task-registry version |  |
| Patient-level split hash |  |
| Random seeds |  |
| Hardware and software environment |  |

## Training configuration

- Missing-value and normalization policy:
- Residual encoder and attention configuration:
- Eight multiclass and nine binary head definitions:
- Task-loss weights and optimizer schedule:
- Checkpoint-selection metric:
- Threshold and calibration procedure:

## Evaluation record

For each task, record class support, confusion matrix, discrimination, calibration, and confidence intervals. Report internal and external cohorts independently before any pooled summary. Bind SHAP artifacts to the evaluated checkpoint and feature schema.

## Release gate

- [ ] Repeated examinations remain in one patient partition.
- [ ] The external cohort did not influence preprocessing or model selection.
- [ ] All 17 task results are present.
- [ ] Macro and weighted averages are clearly distinguished.
- [ ] No patient record, identifier, or private site mapping is exported.
