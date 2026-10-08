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
  <img src="assets/graphical-abstract.png" width="920" alt="DeepCard graphical abstract">
</p>

DeepCard maps pre-measured echocardiographic parameters to standardized diagnostic interpretations, targeting the interpretation layer rather than image acquisition or automated caliper placement.

## At a glance

| Input | Shared representation | Outputs | Interpretation |
|:---|:---|:---|:---|
| 39 quantitative echo measurements | Residual 1D CNN + attention | 8 multiclass + 9 binary tasks | Task-wise SHAP attribution |

| Training cohort | Internal test | External cohort | External degradation |
|:---:|:---:|:---:|:---:|
| **400 patients** | **100 patients** | **102 patients** | **2.6% mean** |

## Task space

| Valvular and pressure | Ventricular function | Structural findings |
|:---|:---|:---|
| MR · TR · AR · AS · PR · PAP | LVDD · LVSD · LAE · RAE · LVE | RVE · LVH · IVST · PE · WMA · VSD |

## Model design

1. **Input standardization** — 2D, M-mode, Doppler, and tissue-Doppler measurements are normalized into a unified feature vector.
2. **Residual feature encoder** — one-dimensional convolutional blocks learn nonlinear interactions among quantitative measurements.
3. **Attention layer** — query–key–value weighting emphasizes task-relevant cardiac features in the shared representation.
4. **Multi-task heads** — softmax heads model disease severity; sigmoid heads model binary diagnostic endpoints.
5. **Joint optimization** — task-specific losses are aggregated while preserving a common cardiovascular representation.
6. **Explainable reporting** — calibrated probabilities and task-wise SHAP profiles support standardized interpretation.

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="940" alt="DeepCard acquisition, architecture, and optimization framework">
</p>

## Results

| Validation | Sensitivity | Precision | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| Internal | **0.77** | **0.74** | **0.75** | **0.77** |
| External · n=102 | **0.75** | **0.72** | **0.73** | **0.75** |

| Clinical endpoint | Published result |
|:---|:---|
| Valvular assessment | **91% specificity** |
| Ventricular evaluation | **82% accuracy** |
| Inter-observer variability | Reduced to **13.4%** |
| External mitral regurgitation | Sensitivity **0.85** · Specificity **0.89** · F1 **0.84** |

### Multiclass severity discrimination

<p align="center">
  <img src="assets/results-multiclass-roc.png" width="940" alt="DeepCard multiclass ROC results">
</p>

### Binary disease discrimination

<p align="center">
  <img src="assets/results-binary-roc.png" width="900" alt="DeepCard binary ROC results with confidence intervals">
</p>

### Task-wise interpretability

<p align="center">
  <img src="assets/results-shap.png" width="940" alt="DeepCard task-wise SHAP feature importance">
</p>

## Codebase blueprint

```text
Deepcard/
├── assets/                         # graphical abstract, architecture, results
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
