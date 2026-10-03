# Artifact Manifest

Every DeepCard result should be traceable from the 39-feature schema and task registry to its reported clinical metric.

| Artifact | Visibility | Required identity |
|:---|:---:|:---|
| Cohort inventory | Private | Cohort version, criteria, patient count, hash |
| Patient split manifest | Private | Internal/external role and split hash |
| Feature schema | Public-safe | Ordered variables, units, missingness rules |
| Task registry | Public-safe | Eight multiclass and nine binary definitions |
| Model checkpoint | Controlled | Config, task weights, seed, weight hash |
| Prediction table | Private | Patient key, task, score, label, checkpoint ID |
| Metric and SHAP tables | Public-safe | Cohort, task, checkpoint, method version |
| Figure | Public | Source-table hash and rendering revision |

## Required metadata

Each record stores `artifact_id`, parent IDs, Git revision, schema and registry versions, configuration hash, content hash, creation time, and access class.

## Lineage rule

Performance and interpretability artifacts must reference the same checkpoint, feature order, task definitions, and cohort version. External-validation outputs never become parents of training or threshold-selection artifacts.

## Integrity checks

- All 17 tasks map to the frozen registry.
- Patient keys remain non-identifying and cohort-local.
- Figure summaries reproduce their task-level source tables.
- Replaced artifacts receive new IDs rather than overwriting history.
