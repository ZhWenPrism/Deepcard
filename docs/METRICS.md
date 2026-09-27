# Evaluation Metrics

DeepCard contains eight multiclass severity tasks and nine binary diagnostic tasks. Task-level results are primary; pooled scores are summaries, not substitutes.

| Task family | Primary metrics | Diagnostic detail |
|:---|:---|:---|
| Multiclass severity | Macro F1, weighted F1, balanced accuracy | Per-class recall and confusion matrix |
| Binary diagnosis | Sensitivity, specificity, precision, F1 | ROC-AUC and precision-recall AUC |
| Probability quality | Brier score, calibration error | Reliability plot by task |
| External validation | Absolute metric and internal-to-external delta | Confidence interval and support |
| Interpretation | Task-wise SHAP ranking stability | Direction and feature definition |

## Aggregation

- Compute each endpoint independently before macro-averaging across tasks.
- Include support for every class and task; weighted averages can conceal rare-endpoint failure.
- Use patient-level confidence intervals and preserve the external cohort as an untouched evaluation set.
- For multiclass tasks, specify whether AUC is one-vs-rest macro, weighted, or class-specific.

## Error analysis

Review clinically adjacent severity confusions, low-prevalence endpoints, calibration drift, and missing-measurement sensitivity. Interpretability results should be associated with the exact checkpoint and cohort used for performance evaluation.

## Minimum result record

Record the 39-feature schema version, label mapping, patient split, missing-data policy, checkpoint, task thresholds, averaging rule, confidence-interval method, and evaluation cohort.
