# DP-FL-AE-Clinical-TimeSeries
## Differential Private Federated Autoencoders for FHIR-Compliant Clinical Time-Series Reconstruction

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)  
![DOI Placeholder](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)

This repository contains the full source code and analysis pipeline used for the paper:

**"Architectural Robustness of Differential Private Federated Autoencoders for FHIR-Compliant Clinical Time-Series Reconstruction"**  
Submitted to the International Journal of Medical Informatics (IJMI).

This project evaluates three Autoencoder architectures (Conv1D, BiLSTM, Transformer) under Differentially Private Federated Learning (DP-FL) using real clinical time-series data (MIMIC-IV) and converted into the FHIR standard.

Conv1D-AE is shown to be the most stable and utility-efficient under DP-FL.

---

## 1. Reproducibility and Data Access

### MIMIC-IV Access Requirement
The dataset used is MIMIC-IV v2.2, which cannot be shared.  
To reproduce experiments:

1. Obtain credentialed access via PhysioNet.  
2. Download MIMIC-IV locally.  
3. Set the BASE_DIR variable in data/process_mimic.py.

### Random Seed
All computations are fully reproducible using:

```
RANDOM_SEED = 42
```

---

## 2. Repository Structure

```
DP-FL-AE-Clinical-TimeSeries/
├── config/
│   └── optimized_config.json
│
├── data/
│   ├── process_mimic.py
│   └── fhir_to_sequence.py
│
├── src/
│   ├── models.py
│   ├── fl_core.py
│   └── utils.py
│
├── main_run.py
├── requirements.txt
└── README.md
```

---

## 3. Setup and Installation

Recommended: Conda environment.

```
conda create -n dp-fl-ae python=3.10
conda activate dp-fl-ae
pip install -r requirements.txt
```

Key libraries: tensorflow, tensorflow-privacy, numpy, pandas, scikit-learn, matplotlib, seaborn, scipy.

---

## 4. Running the Pipeline

### Stage 1: Data Preparation

Set MIMIC-IV root directory:

Open data/process_mimic.py and edit:

```
BASE_DIR = "path/to/mimic-iv"
```

Then run:

```
python data/process_mimic.py
python data/fhir_to_sequence.py
```

This converts raw MIMIC-IV to FHIR and then standardized sequences.

---

### Stage 2: DP-FL Experiments

Run:

```
python main_run.py
```

This executes the full experiment suite (Centralized, FL, DP-FL).

Privacy budget from RDP accountant:

```
epsilon = 58.93
delta = 1e-5
```

---

## 5. Output Files

Generated in the results/ folder:

| File | Description |
|------|-------------|
| metrics_summary.csv | RMSE, MAE, performance ratios |
| multi_model_rmse_comparison.png | RMSE comparison bar chart |
| performance_ratio_plot.png | Utility degradation ratio |
| fl_convergence_plot.png | Global loss over 50 rounds |
| conv1d_sequence_reconstruction.png | Best model qualitative reconstruction |

---

## 6. Citation

```
@article{YOUR_PAPER,
  title={Architectural Robustness of Differential Private Federated Autoencoders for FHIR-Compliant Clinical Time-Series Reconstruction},
  author={Your Name},
  journal={International Journal of Medical Informatics},
  year={2025}
}
```

```
@software{DP_FL_AE_Repo,
  author       = {Your Name},
  title        = {DP-FL-AE-Clinical-TimeSeries},
  doi          = {10.5281/zenodo.XXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXX}
}
```
