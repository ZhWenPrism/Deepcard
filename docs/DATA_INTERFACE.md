# Data Interface

DeepCard consumes pre-measured echocardiographic parameters. This contract does not cover image acquisition, view selection, or automated measurement extraction.

## Examination record

| Field group | Required content |
|:---|:---|
| Identity | Non-identifying `patient_id` and `exam_id` |
| Measurements | Ordered 39-feature vector with names, units, and missingness mask |
| Multiclass targets | Eight severity labels with versioned class definitions |
| Binary targets | Nine diagnostic labels with explicit positive and negative encodings |
| Cohort metadata | Internal or external cohort and patient-level split assignment |
| Provenance | Measurement protocol, schema version, and label-review version |

## Task registry

The registry maps every task to its type, allowed labels, loss, output activation, evaluation metrics, and decision threshold. Changes to label order or severity definitions require a new registry version.

## Validation checks

- Every measurement has a declared unit and valid numeric range.
- Missing values remain distinguishable from physiological zero.
- Repeated examinations from one patient share a partition.
- The external cohort is excluded from preprocessing and model selection.
- Task tensors follow the published eight-multiclass and nine-binary split.

## Public boundary

Only the schema, validation logic, and synthetic examples may be public. Patient measurements, identifiers, site mappings, private manifests, and checkpoints remain excluded.
