<div align="center">

# DeepCard

**Standardized multi-task interpretation of multiparameter cardiac ultrasound**

[![Paper](https://img.shields.io/badge/Paper-iScience_2026-5B5BD6?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Open Access](https://img.shields.io/badge/Open_Access-CC_BY--NC--ND_4.0-2A9D8F?style=flat-square)](https://doi.org/10.1016/j.isci.2026.116904)
[![Code](https://img.shields.io/badge/Code-available-2563EB?style=flat-square)](DeepCard)

</div>

DeepCard jointly interprets 39 quantitative echocardiographic measurements across 17 diagnostic tasks with external validation.

## Architecture

```mermaid
flowchart LR
    A[39 echo measurements] --> B[Residual 1D CNN]
    B --> C[Multi-head attention]
    C --> D[Shared representation]
    D --> E[8 severity tasks]
    D --> F[9 binary tasks]
```

## Results

| Valvular specificity | Ventricular accuracy | External performance drop |
|:---:|:---:|:---:|
| **91%** | **82%** | **2.6%** |

<p align="center">
  <img src="assets/results.png" width="900" alt="DeepCard ROC results">
</p>

## Repository layout

```text
Deepcard/
├── assets/                 # result figures
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
