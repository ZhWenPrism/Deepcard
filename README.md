<div align="center">

# DeepCard

**Standardized multi-task interpretation of multiparameter cardiac ultrasound**

[![Paper](https://img.shields.io/badge/Paper-iScience_2026-5B5BD6?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Open Access](https://img.shields.io/badge/Open_Access-CC_BY--NC--ND_4.0-2A9D8F?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Tasks](https://img.shields.io/badge/Tasks-17-8B5CF6?style=flat-square)](#results)
[![Code](https://img.shields.io/badge/Code-available-2563EB?style=flat-square)](DeepCard)

<sub>39 measurements · 17 diagnostic tasks · external validation</sub>

</div>

<p align="center">
  <img src="assets/graphical-abstract.png" width="920" alt="DeepCard graphical abstract">
</p>

DeepCard transforms pre-measured echocardiographic parameters into standardized, multi-task clinical interpretations.

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="940" alt="DeepCard acquisition, architecture, and optimization framework">
</p>

## Results

| Validation | Sensitivity | Precision | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| Internal | **0.77** | **0.74** | **0.75** | **0.77** |
| External · n=102 | **0.75** | **0.72** | **0.73** | **0.75** |

| Representative external endpoint | Result |
|:---|:---|
| Mitral regurgitation | Sensitivity **0.85** · Specificity **0.89** · F1 **0.84** |
| LV systolic dysfunction | Accuracy **0.79** |
| LV diastolic dysfunction | Accuracy **0.77** |

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

## Repository layout

```text
Deepcard/
├── assets/                 # graphical abstract, architecture, and results
├── DeepCard/
│   ├── model.py            # multi-task network
│   ├── train.py            # stratified training pipeline
│   ├── evaluate.py         # internal and external evaluation
│   ├── interpret.py        # SHAP interpretation
│   ├── DATA_FORMAT.md      # privacy-safe input schema
│   └── requirements.txt
└── README.md
```

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
