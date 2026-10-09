<div align="center">

# DeepCard

**Standardized multi-task interpretation of multiparameter cardiac ultrasound**

[![Paper](https://img.shields.io/badge/Paper-iScience_2026-5B5BD6?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Open Access](https://img.shields.io/badge/Open_Access-CC_BY--NC--ND_4.0-2A9D8F?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Tasks](https://img.shields.io/badge/Tasks-17-8B5CF6?style=flat-square)](#task-space)
[![Code](https://img.shields.io/badge/Code-available-2563EB?style=flat-square)](DeepCard)

<sub>39 measurements · 17 diagnostic tasks · multi-task learning · external validation</sub>

</div>

<p align="center">
  <img src="assets/graphical-abstract.png" alt="DeepCard graphical abstract">
</p>

DeepCard maps pre-measured echocardiographic parameters to standardized diagnostic interpretations, targeting the interpretation layer rather than image acquisition or automated caliper placement.

The model addresses a distinct source of variability in cardiac ultrasound: clinicians may reach different diagnostic interpretations even when working from the same standardized measurements. DeepCard therefore does not replace acquisition or measurement protocols. It learns a reproducible mapping from 39 quantitative parameters to 17 report-level endpoints spanning valvular disease, ventricular function, pressure estimates, chamber remodeling, and structural abnormalities.

## At a glance

<table align="center">
  <tr align="center">
    <th>Input</th><th>Shared representation</th><th>Outputs</th><th>Interpretation</th>
  </tr>
  <tr align="center">
    <td>39 quantitative<br>echo measurements</td>
    <td>Residual 1D CNN<br>+ attention</td>
    <td>8 multiclass<br>+ 9 binary tasks</td>
    <td>Task-wise<br>SHAP attribution</td>
  </tr>
</table>

<table align="center">
  <tr align="center">
    <th>Training cohort</th><th>Internal test</th><th>External cohort</th><th>External degradation</th>
  </tr>
  <tr align="center">
    <td><b>400 patients</b></td><td><b>100 patients</b></td><td><b>102 patients</b></td><td><b>2.6% mean</b></td>
  </tr>
</table>

## Task space

<table align="center">
  <tr align="center">
    <th>Valvular and pressure</th><th>Ventricular function</th><th>Structural findings</th>
  </tr>
  <tr align="center">
    <td>MR · TR · AR · AS · PR · PAP</td><td>LVDD · LVSD · LAE · RAE · LVE</td><td>RVE · LVH · IVST · PE · WMA · VSD</td>
  </tr>
</table>

## Model design

1. **Input standardization** — 2D, M-mode, Doppler, and tissue-Doppler measurements are normalized into a unified feature vector.
2. **Residual feature encoder** — one-dimensional convolutional blocks learn nonlinear interactions among quantitative measurements.
3. **Attention layer** — query–key–value weighting emphasizes task-relevant cardiac features in the shared representation.
4. **Multi-task heads** — softmax heads model disease severity; sigmoid heads model binary diagnostic endpoints.
5. **Joint optimization** — task-specific losses are aggregated while preserving a common cardiovascular representation.
6. **Explainable reporting** — calibrated probabilities and task-wise SHAP profiles support standardized interpretation.

## Architecture

DeepCard treats quantitative echocardiographic measurements as a clinically ordered sequence. A residual 1D encoder and attention layer build a shared representation, while task-specific heads produce eight severity-graded and nine binary diagnostic outputs.

The feature sequence preserves clinically meaningful neighborhoods among chamber dimensions, Doppler measurements, valve-related variables, and functional indices. Residual convolutional blocks capture local interactions, attention redistributes emphasis across the full measurement profile, and separate softmax or sigmoid heads match the label structure of each endpoint. Joint optimization then allows correlated tasks to share statistical strength without forcing them to use identical decision boundaries.

<p align="center">
  <img src="assets/architecture.png" alt="DeepCard acquisition, architecture, and optimization framework"><br>
  <sub>Figure 4. Measurement standardization, shared representation, and multi-task optimization.</sub>
</p>

## Results

<table align="center">
  <tr align="center">
    <th>Validation</th><th>Sensitivity</th><th>Precision</th><th>F1</th><th>Accuracy</th>
  </tr>
  <tr align="center"><td>Internal</td><td><b>0.77</b></td><td><b>0.74</b></td><td><b>0.75</b></td><td><b>0.77</b></td></tr>
  <tr align="center"><td>External · n=102</td><td><b>0.75</b></td><td><b>0.72</b></td><td><b>0.73</b></td><td><b>0.75</b></td></tr>
</table>

### Internal and external validation

The internal test set comprised 100 patients, and generalization was assessed in an independent 102-patient cohort from a separate medical center. Mean sensitivity, precision, F1, and accuracy changed from 0.77, 0.74, 0.75, and 0.77 internally to 0.75, 0.72, 0.73, and 0.75 externally. The mean decline in sensitivity across the 17 endpoints was 2.6%.

Performance did not transfer uniformly across tasks. Common valvular and ventricular endpoints retained relatively stable estimates, whereas lower-prevalence structural findings showed wider confidence intervals and should be interpreted more cautiously. The repository therefore reports endpoint-level results and confidence intervals rather than relying only on a single macro average.

<table align="center">
  <tr align="center"><th>Clinical endpoint</th><th>Published result</th></tr>
  <tr align="center"><td>Valvular assessment</td><td><b>91% specificity</b></td></tr>
  <tr align="center"><td>Ventricular evaluation</td><td><b>82% accuracy</b></td></tr>
  <tr align="center"><td>Inter-observer variability</td><td>Reduced to <b>13.4%</b></td></tr>
  <tr align="center"><td>External mitral regurgitation</td><td>Sensitivity <b>0.85</b> · Specificity <b>0.89</b> · F1 <b>0.84</b></td></tr>
</table>

### Multiclass severity discrimination

Severity-specific ROC curves show how discrimination changes from mild to severe disease across eight graded endpoints. Performance is strongest for clinically advanced valvular disease while remaining stable across ventricular-function grades.

For mitral, tricuspid, and aortic regurgitation, the severe categories reached AUCs of 0.92, 0.91, and 0.91, respectively. Left-ventricular diastolic dysfunction remained between 0.85 and 0.88 across severity levels, while systolic dysfunction ranged from 0.83 to 0.87. Presenting the full set of class-specific curves makes it possible to distinguish strong severe-disease discrimination from the more difficult separation of adjacent mild and moderate grades.

<p align="center">
  <img src="assets/results-multiclass-roc.png" alt="DeepCard multiclass ROC results"><br>
  <sub>Figure 5. Multiclass ROC analysis for severity-graded endpoints.</sub>
</p>

### Binary disease discrimination

Nine binary tasks cover chamber enlargement, hypertrophy, effusion, wall-motion abnormality, and septal defect. Class-wise confidence bands expose both discrimination and uncertainty for the presence and absence of each finding.

For example, left-atrial enlargement achieved an AUC of 0.82 for disease detection and 0.85 for exclusion, while pericardial-effusion detection reached an AUC of 0.76. These curves should be read together with disease prevalence and interval width: endpoints with fewer positive observations can show apparently acceptable point estimates while retaining greater statistical uncertainty.

<p align="center">
  <img src="assets/results-binary-roc.png" alt="DeepCard binary ROC results with confidence intervals"><br>
  <sub>Figure 6. Binary ROC curves with 95% confidence intervals.</sub>
</p>

### Error structure

Confusion matrices complement aggregate metrics by showing where errors occur within each endpoint. Binary matrices expose class imbalance and false-negative patterns; multiclass matrices show whether errors remain close to the neighboring severity grade.

This distinction matters for clinical interpretation. A one-grade error between adjacent severity categories is not equivalent to confusing a normal examination with severe disease, and a false negative in a low-prevalence structural endpoint has a different implication from a false positive. The full matrices preserve these task-specific error structures rather than compressing them into a single accuracy value.

<p align="center">
  <img src="assets/confusion-matrices-binary.png" alt="Binary-task confusion matrices"><br>
  <sub>Figure 7. Confusion matrices for the nine binary endpoints.</sub>
</p>

<p align="center">
  <img src="assets/confusion-matrices-multiclass.png" alt="Multiclass confusion matrices"><br>
  <sub>Figure 8. Confusion matrices for representative severity-graded endpoints.</sub>
</p>

### Task-wise interpretation

Global SHAP importance identifies the measurements consistently used across diagnostic tasks. The task-by-feature heatmap preserves differences between valvular, functional, and structural endpoints instead of collapsing interpretation into a single ranking.

The resulting attribution structure is clinically heterogeneous. Pressure-related predictions emphasize right-sided and pulmonary vascular measurements; diastolic dysfunction places greater weight on left-atrial size and filling parameters; systolic dysfunction is driven by global functional markers such as LVEF and LVESD. This pattern shows that the shared encoder does not reduce all tasks to the same generic feature profile.

<p align="center">
  <img src="assets/results-shap.png" alt="DeepCard task-wise SHAP feature importance"><br>
  <sub>Figure 10. Global task-wise feature attribution.</sub>
</p>

Local SHAP profiles provide a more detailed view for representative tasks, linking the direction and magnitude of individual measurements to each model output.

Each point represents one patient-level attribution, so the plots show both the global ordering of important variables and the direction in which high or low measurements shift a prediction. The local views are intended as model-behavior evidence rather than causal explanations: they clarify which standardized measurements influenced the output, but they do not establish that changing a measurement would change the underlying disease state.

<p align="center">
  <img src="assets/local-shap.png" alt="Local SHAP profiles for four representative DeepCard tasks"><br>
  <sub>Figure 11. Local attribution profiles for four representative tasks.</sub>
</p>

Published tables: [task metrics with confidence intervals](results/disease_performance_with_ci.csv) · [external validation](results/internal_external_validation.csv) · [baseline benchmark](results/baseline_benchmark.csv) · [ordering ablation](results/ordering_ablation.csv)

## Codebase blueprint

```text
Deepcard/
├── assets/                         # graphical abstract, architecture, result figures
├── results/                        # machine-readable published tables
├── DeepCard/
│   ├── config.py                   # experiment configuration
│   ├── data.py                     # tabular data ingestion
│   ├── feature_engineering.py      # measurement preprocessing
│   ├── model.py                    # shared encoder and task heads
│   ├── losses.py                   # multi-task objectives
│   ├── train.py                    # training workflow
│   ├── evaluate.py                 # performance evaluation
│   ├── interpret.py                # SHAP analysis
│   ├── run_train.py                # training entry point
│   ├── run_evaluate.py             # evaluation entry point
│   ├── configs/                    # modular configuration scaffold
│   ├── datasets/
│   │   ├── schemas/                # measurement and label schemas
│   │   ├── transforms/             # preprocessing components
│   │   └── splits/                 # internal and external manifests
│   ├── models/
│   │   ├── backbones/              # residual 1D encoders
│   │   ├── attention/              # shared attention blocks
│   │   └── heads/                  # multiclass and binary heads
│   ├── training/
│   │   ├── losses/                 # task weighting and objectives
│   │   ├── optimizers/             # schedules and optimization
│   │   └── callbacks/              # checkpoints and early stopping
│   ├── evaluation/
│   │   ├── internal/               # holdout evaluation
│   │   ├── external/               # cross-center validation
│   │   └── calibration/            # confidence analysis
│   ├── explainability/             # task-wise feature attribution
│   ├── reporting/                  # standardized output generation
│   └── tests/
│       ├── unit/
│       └── integration/
└── README.md
```

The existing lightweight implementation is preserved and now complemented by validated 39-feature/17-task contracts, dependency-light evaluation metrics, a canonical task registry, structured reporting utilities, unit tests, and CI. Training still requires the dependencies and private data described in `DeepCard/requirements.txt` and `DeepCard/DATA_FORMAT.md`.

<details>
<summary><b>Citation</b></summary>

```bibtex
@article{wen2026deepcard,
  title   = {A Deep Learning Framework for Standardized Interpretation of Multiparameter Cardiac Ultrasound and Disease Classification},
  author  = {Wen, Zhihong and Liu, Xiangpeng and Liu, Yi and Tang, Wenbo and An, Kang and Guo, Shengli},
  journal = {iScience},
  volume  = {29},
  number  = {8},
  pages   = {116904},
  year    = {2026},
  doi     = {10.1016/j.isci.2026.116904}
}
```

</details>
