# DP-FL-AE-Clinical-TimeSeries
## Differentially Private Federated Autoencoders for FHIR-Compliant Clinical Time-Series Reconstruction


This repository contains the full codebase and experiment pipeline for the manuscript:

> **"Federated Differentially Private Autoencoder for Multivariate Clinical Time-Series Reconstruction Using Real-World FHIR Data"**  
> *Submitted to the International Journal of Medical Informatics (IJMI).*

The project benchmarks three deep sequence Autoencoder architectures—**Conv1D-AE**, **BiLSTM-AE**, and **Transformer-AE**—within a **Differentially Private Federated Learning (DP-FL)** framework using real clinical time-series data derived from **MIMIC-IV** and transformed into the **FHIR** interoperability standard.

A key finding is that the **Conv1D-AE** demonstrates the highest stability and utility efficiency under strict DP-FL constraints, while BiLSTM-AE and Transformer-AE illustrate varying robustness to differential privacy noise.

---

# 1. Reproducibility and Data Access

## ⚠️ MIMIC-IV Requirement

This project uses **MIMIC-IV v2.2**, which cannot be redistributed due to data-use agreements.  
To reproduce experiments:

1. Request credentialed access from **PhysioNet**.  
2. Download the MIMIC-IV dataset locally.  
3. Set the dataset root path in `data/process_mimic.py`:

```python
BASE_DIR = "path/to/mimic-iv"
```

### Random Seed

A fixed seed is applied throughout to ensure deterministic reproducibility:

```python
RANDOM_SEED = 42
```

---

# 2. Repository Structure

This repository is organized to clearly separate data processing, model definitions, federated learning logic, and the unified execution pipeline.

```
DP-FL-AE-Clinical-TimeSeries/
├── config/
│   └── optimized_config.json          # Hyperparameters & experiment configuration
│
├── data/
│   ├── process_mimic.py               # Raw MIMIC → FHIR transformation pipeline
│   └── fhir_to_sequence.py            # FHIR → multivariate time-series conversion
│
├── src/
│   ├── models.py                      # Conv1D-AE, BiLSTM-AE, Transformer-AE
│   ├── fl_core.py                     # Federated averaging & DP-FL training logic
│   └── utils.py                       # RDP accountant, plotting utilities
│
├── main_pipeline_combined.py          # ★ SINGLE EXECUTABLE ENTRYPOINT
│                                      # (Runs Centralized → FL → DP-FL)
│
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation (this file)
```

Files in `data/` and `src/` act as **structural documentation** and core program components.  
The entire end-to-end workflow is wrapped into **main_pipeline_combined.py**, which reviewers can execute directly.

---

# 3. Installation

Recommended environment: **conda**

```bash
conda create -n dp-fl-ae python=3.10
conda activate dp-fl-ae
pip install -r requirements.txt
```

Key libraries include:

- TensorFlow  
- TensorFlow-Privacy  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- SciPy  

---

# 4. Running the Full Pipeline

### Step 1: Data Processing (MIMIC-IV → FHIR)

```bash
python data/process_mimic.py
```

These scripts convert raw MIMIC-IV tables into:
- FHIR-compliant Observations  

---

### Step 2: Execute All Experiments (FHIR → Sequences & Centralized, FL, DP-FL)

Run the single entrypoint script:

```bash
python main_pipeline_combined.py
```

The script automatically performs:

- Centralized AE training  
- Federated Learning (FL)  
- Differentially Private FL (DP-FL) via DP-SGD  
- RDP privacy accounting  
- Reconstruction evaluation & plotting  

---

### Privacy Budget (DP-SGD)

Computed using the Rényi Differential Privacy (RDP) accountant:

```
epsilon = 58.93
delta = 1e-5
```

---

# 5. Output Files

All experiment outputs are saved in the `results/` directory.

| File | Description |
|------|-------------|
| metrics_summary.csv | RMSE, MAE, and performance ratios for all models |
| multi_model_rmse_comparison.png | Centralized vs FL vs DP-FL RMSE comparison |
| performance_ratio_plot.png | Utility degradation ratios across architectures |
| fl_convergence_plot.png | Global federated loss vs rounds |
| conv1d_sequence_reconstruction.png | Example reconstruction from the best model |
| Additional per-model reconstructions | Generated automatically |

---

# 6. Citation

If you use this work, please cite both the paper and the code:

### Paper

```bibtex
@article{YOUR_PAPER,
  title={Differentially Private Federated Autoencoders for FHIR-Compliant Clinical Time-Series Reconstruction},
  author={Your Name},
  journal={International Journal of Medical Informatics},
  year={2025}
}
```

### Code

```bibtex
@software{DP_FL_AE_Repo,
  author       = {Your Name},
  title        = {DP-FL-AE-Clinical-TimeSeries},
  doi          = {10.5281/zenodo.XXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXX},
  year         = 2025
}
```

---
