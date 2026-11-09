"""
ECO331 Project: Explainable Macroeconomic Inflation Forecasting.

Compares econometric benchmarks, regularized linear models, tree ensembles,
and deep neural/sequence architectures (MLP, LSTM, Temporal Transformer)
on US macroeconomic monthly data (FRED, 1978–2026), with SHAP and temporal
attention explainability.
"""

import os
from pathlib import Path

# Prevent macOS ARM64 OpenMP runtime collision when combining PyTorch + XGBoost
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
