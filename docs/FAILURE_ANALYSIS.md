# Failure Analysis

DeepCard failure analysis is task-specific because pooled performance can conceal rare diseases and adjacent severity errors.

## Error taxonomy

| Failure class | Diagnostic view | Minimum context |
|:---|:---|:---|
| Adjacent severity error | Ordinal confusion matrix | True and predicted grade |
| Distant severity error | Severity-distance distribution | Probability vector |
| Binary false negative | Task-specific review | Prevalence and threshold |
| Missing-measurement sensitivity | Metric versus missingness pattern | Available feature mask |
| External-cohort drift | Internal/external metric delta | Feature and label shift |
| Calibration failure | Reliability curve and Brier score | Task support |

## Required stratification

Report every task separately, with class support, cohort, missingness burden, and clinically relevant subgroups when permitted. Macro summaries never replace low-prevalence endpoint review.

## Case review

For severe or high-confidence errors, record the task, label definition, calibrated probabilities, missing measurements, top SHAP features, feature-range flags, and internal or external cohort. Interpretability is diagnostic evidence, not proof of causality.

## Corrective-action rule

Changes to preprocessing, task weighting, calibration, or thresholds must be evaluated on frozen patient splits. External data remain evaluation-only and cannot be used to retroactively choose the correction.
