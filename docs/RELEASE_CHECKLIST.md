# Release Checklist

Complete this checklist before publishing a DeepCard result, model artifact, or implementation update.

## Evidence

- [ ] The ordered 39-feature schema and all measurement units are versioned.
- [ ] Eight multiclass and nine binary task definitions are frozen.
- [ ] Patient-level internal and external cohort roles remain disjoint.
- [ ] All 17 task results precede pooled summaries.
- [ ] SHAP tables reference the same checkpoint, schema, and cohort as performance results.

## Clinical and privacy review

- [ ] No patient measurement, identifier, site mapping, or private manifest is exposed.
- [ ] The repository does not imply automated image acquisition or caliper placement.
- [ ] Rare tasks, distant severity errors, missingness, and calibration have been reviewed.
- [ ] External data did not influence preprocessing, thresholds, or checkpoint selection.

## Repository quality

- [ ] Seeds, loss weights, thresholds, environment, and metric averaging are recorded.
- [ ] Public examples are synthetic or separately approved.
- [ ] Documentation and citation metadata match the released scope.
- [ ] Checkpoints, prediction tables, logs, and credentials are excluded or access-controlled.
- [ ] The tagged revision reproduces every public-safe result artifact.
