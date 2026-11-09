"""
Causal macroeconomic feature engineering, regime labeling, and horizon-purged splits.

Constructs economically interpretable stationary predictors for US CPI inflation
forecasting across horizons h in {1, 3, 6, 12} months ahead, ensuring zero look-ahead
bias and strict h-month target embargo between train/validation/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src import DATA_PROCESSED_DIR

FEATURE_LABELS: Dict[str, str] = {
    "cpi_yoy": "Headline CPI YoY (%)",
    "core_headline_gap": "Core - Headline Gap (pp)",
    "cpi_mom_ann": "CPI 1m Momentum (Ann. %)",
    "cpi_3m_ann": "CPI 3m Momentum (Ann. %)",
    "base_3m_ann": "Base-Effect CPI (t-12..t-9, %)",
    "cpi_yoy_lag3": "Headline CPI YoY (t-3)",
    "cpi_yoy_lag6": "Headline CPI YoY (t-6)",
    "mich_exp_1y": "1Y Inflation Expect. (Mich. %)",
    "mich_exp_chg3m": "Inflation Expect. 3m Chg (pp)",
    "unrate": "Unemployment Rate (%)",
    "unrate_chg_12m": "Unemployment 12m Chg (pp)",
    "indpro_yoy": "Industrial Prod. YoY (%)",
    "ppi_yoy": "PPI Producer Inflation YoY (%)",
    "oil_yoy": "WTI Crude Oil YoY (%)",
    "real_fedfunds": "Real Fed Funds Rate (pp)",
    "fedfunds_chg_12m": "Fed Funds 12m Chg (pp)",
    "yield_spread_10y2y": "Yield Spread 10Y-2Y (pp)",
    "m2_yoy": "M2 Money Growth YoY (%)",
    "umcsent_chg_12m": "Consumer Sentiment 12m Chg",
    "elec_yoy": "Electricity CPI YoY (%)",
    "stock_yoy": "US Stock Market YoY (%)",
    "gdp_yoy": "Real GDP Growth YoY (%, Lag 1Q)",
    "month_sin": "Seasonality (sin)",
    "month_cos": "Seasonality (cos)",
}

TABULAR_FEATURES: List[str] = list(FEATURE_LABELS.keys())
CORE_12_TABULAR_FEATURES: List[str] = [
    c for c in TABULAR_FEATURES if c not in ("elec_yoy", "stock_yoy", "gdp_yoy")
]

# Core contemporaneous macro channels fed as a 12-month sequence [t-11 ... t] into LSTM & Transformer
SEQUENCE_FEATURES: List[str] = [
    "cpi_yoy",
    "core_headline_gap",
    "cpi_3m_ann",
    "base_3m_ann",
    "mich_exp_1y",
    "unrate",
    "indpro_yoy",
    "ppi_yoy",
    "oil_yoy",
    "elec_yoy",
    "real_fedfunds",
    "yield_spread_10y2y",
    "m2_yoy",
    "stock_yoy",
]

PHILLIPS_CURVE_FEATURES: List[str] = [
    "cpi_yoy",
    "cpi_yoy_lag3",
    "mich_exp_1y",
    "unrate",
    "ppi_yoy",
    "oil_yoy",
]

AR_BENCHMARK_FEATURES: List[str] = [
    "cpi_yoy",
    "cpi_mom_ann",
    "cpi_3m_ann",
    "cpi_yoy_lag3",
    "cpi_yoy_lag6",
]

HORIZONS: List[int] = [1, 3, 6, 12]
SEQ_LEN: int = 12


def assign_macro_regime(dates: pd.DatetimeIndex) -> pd.Series:
    """Assign historical and out-of-sample macroeconomic regimes for storytelling & evaluation."""
    regimes = []
    for d in dates:
        if d < pd.Timestamp("1984-01-01"):
            regimes.append("Great Inflation & Volcker (1978-1983)")
        elif d < pd.Timestamp("2008-01-01"):
            regimes.append("Great Moderation (1984-2007)")
        elif d < pd.Timestamp("2019-01-01"):
            regimes.append("Post-GFC Low Inflation (2008-2018)")
        elif d < pd.Timestamp("2020-03-01"):
            regimes.append("Pre-COVID Stability (2019-Early 2020)")
        elif d < pd.Timestamp("2023-01-01"):
            regimes.append("Pandemic & Supply Shock (2020-2022)")
        else:
            regimes.append("Monetary Tightening & Disinflation (2023-2026)")
    return pd.Series(regimes, index=dates, name="regime")


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw monthly FRED series into stationary, causal macroeconomic features
    and multi-horizon targets (h = 1, 3, 6, 12 months ahead).
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(index=panel.index)

    # Price inflation & momentum dynamics (all strictly backward-looking at date t)
    df["cpi_yoy"] = panel["CPIAUCSL"].pct_change(12) * 100.0
    df["core_cpi_yoy"] = panel["CPILFESL"].pct_change(12) * 100.0
    df["core_headline_gap"] = df["core_cpi_yoy"] - df["cpi_yoy"]
    df["cpi_mom_ann"] = ((panel["CPIAUCSL"] / panel["CPIAUCSL"].shift(1)) ** 12 - 1.0) * 100.0
    df["cpi_3m_ann"] = ((panel["CPIAUCSL"] / panel["CPIAUCSL"].shift(3)) ** 4 - 1.0) * 100.0
    df["base_3m_ann"] = ((panel["CPIAUCSL"].shift(9) / panel["CPIAUCSL"].shift(12)) ** 4 - 1.0) * 100.0
    df["cpi_yoy_lag3"] = df["cpi_yoy"].shift(3)
    df["cpi_yoy_lag6"] = df["cpi_yoy"].shift(6)

    # Inflation expectations (NKPC forward channel)
    df["mich_exp_1y"] = panel["MICH"]
    df["mich_exp_chg3m"] = panel["MICH"] - panel["MICH"].shift(3)

    # Phillips Curve / Real Activity Slack
    df["unrate"] = panel["UNRATE"]
    df["unrate_chg_12m"] = panel["UNRATE"] - panel["UNRATE"].shift(12)
    df["indpro_yoy"] = panel["INDPRO"].pct_change(12) * 100.0

    # Upstream Supply & Commodity Shocks (winsorize extreme oil base-effect spikes softly at +-120%)
    df["ppi_yoy"] = panel["PPIACO"].pct_change(12) * 100.0
    df["oil_yoy"] = (panel["WTISPLC"].pct_change(12) * 100.0).clip(-80.0, 140.0)

    # Monetary Policy, Yield Curve, and Liquidity
    df["fedfunds"] = panel["FEDFUNDS"]
    df["real_fedfunds"] = panel["FEDFUNDS"] - df["cpi_yoy"]
    df["fedfunds_chg_12m"] = panel["FEDFUNDS"] - panel["FEDFUNDS"].shift(12)
    df["yield_spread_10y2y"] = panel["GS10"] - panel["GS2"]
    df["m2_yoy"] = panel["M2SL"].pct_change(12) * 100.0

    # Consumer Demand Confidence
    df["umcsent"] = panel["UMCSENT"]
    df["umcsent_chg_12m"] = panel["UMCSENT"] - panel["UMCSENT"].shift(12)

    # Cross-Paper Synthesis Channels: Sticky Electricity (Kuskaya 2025), Equities (Vamossy 2024), Real GDP (Yoon 2021)
    df["elec_yoy"] = panel["CUSR0000SEHF01"].pct_change(12) * 100.0
    df["stock_yoy"] = panel["SPASTT01USM661N"].pct_change(12) * 100.0
    df["gdp_yoy"] = panel["GDPC1"].pct_change(12) * 100.0

    # Cyclical calendar seasonality
    months = df.index.month.values
    df["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

    # Regimes & metadata
    df["regime"] = assign_macro_regime(df.index)
    df["is_imputed_gap"] = panel["is_imputed_gap"]

    # Direct h-step ahead Headline CPI YoY (%) targets and anchored changes (pi_{t+h} - pi_t)
    for h in HORIZONS:
        df[f"target_cpi_h{h}"] = df["cpi_yoy"].shift(-h)
        df[f"target_delta_h{h}"] = df[f"target_cpi_h{h}"] - df["cpi_yoy"]
        df[f"target_date_h{h}"] = df.index.to_series().shift(-h)

    # Drop initial warmup period (1978-01 .. 1979-06) needed for 12m changes + 6m lags
    df = df.loc[df.index >= pd.Timestamp("1979-07-01")].copy()
    df.to_csv(DATA_PROCESSED_DIR / "macro_features_1979_2026.csv")
    return df


@dataclass
class HorizonDataset:
    horizon: int
    feature_names: List[str]
    seq_feature_names: List[str]
    # Full aligned pre-test (train + val) with strict h-month embargo before test_start
    dates_train_full: pd.DatetimeIndex
    X_train_full: pd.DataFrame
    X_seq_train_full: np.ndarray
    y_train_full: np.ndarray          # actual pi_{t+h}
    anchor_train_full: np.ndarray     # current pi_t
    delta_train_full: np.ndarray      # pi_{t+h} - pi_t
    # Sub-split for neural net early stopping & hyperparameter validation (strictly purged)
    X_subtrain: pd.DataFrame
    X_seq_subtrain: np.ndarray
    delta_subtrain: np.ndarray
    X_val: pd.DataFrame
    X_seq_val: np.ndarray
    y_val: np.ndarray
    anchor_val: np.ndarray
    delta_val: np.ndarray
    # Out-of-sample test set (2019-01 onward)
    dates_test: pd.DatetimeIndex
    target_dates_test: pd.DatetimeIndex
    regimes_test: pd.Series
    X_test: pd.DataFrame
    X_seq_test: np.ndarray
    y_test: np.ndarray
    anchor_test: np.ndarray
    delta_test: np.ndarray


def prepare_horizon_dataset(
    df: pd.DataFrame,
    horizon: int = 3,
    val_start: str = "2014-01-01",
    test_start: str = "2019-01-01",
    seq_len: int = SEQ_LEN,
) -> HorizonDataset:
    """
    Build strictly aligned tabular and 12-month sequence datasets for a given forecast
    horizon h, enforcing an h-month embargo before both validation and test boundaries
    so no future inflation realization ever leaks across splits.
    """
    target_col = f"target_cpi_h{horizon}"
    delta_col = f"target_delta_h{horizon}"

    # Pre-construct 3D sequence array [N_total, seq_len, D_seq] over df
    seq_mat = df[SEQUENCE_FEATURES].values
    n_rows = len(df)
    valid_indices = []
    seq_list = []

    for i in range(seq_len - 1, n_rows):
        if pd.isna(df[target_col].iloc[i]):
            continue
        window = seq_mat[i - seq_len + 1 : i + 1]
        if np.isnan(window).any() or df[TABULAR_FEATURES].iloc[i].isna().any():
            continue
        valid_indices.append(i)
        seq_list.append(window)

    sub_df = df.iloc[valid_indices].copy()
    X_seq_all = np.stack(seq_list, axis=0).astype(np.float32)

    val_ts = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)

    # Target realization date t + h months
    realized_dates = sub_df.index + pd.DateOffset(months=horizon)

    # Strict causal masks:
    # 1. Subtrain: target realized strictly before val_start
    mask_subtrain = np.asarray(realized_dates < val_ts, dtype=bool)
    # 2. Validation: origin >= val_start AND target realized strictly before test_start
    mask_val = np.asarray((sub_df.index >= val_ts) & (realized_dates < test_ts), dtype=bool)
    # 3. Full pre-test train: target realized strictly before test_start
    mask_train_full = np.asarray(realized_dates < test_ts, dtype=bool)
    # 4. Out-of-sample test: forecast origin >= test_start
    mask_test = np.asarray(sub_df.index >= test_ts, dtype=bool)

    return HorizonDataset(
        horizon=horizon,
        feature_names=TABULAR_FEATURES,
        seq_feature_names=SEQUENCE_FEATURES,
        dates_train_full=sub_df.index[mask_train_full],
        X_train_full=sub_df.loc[mask_train_full, TABULAR_FEATURES].copy(),
        X_seq_train_full=X_seq_all[mask_train_full],
        y_train_full=sub_df.loc[mask_train_full, target_col].values.astype(np.float64),
        anchor_train_full=sub_df.loc[mask_train_full, "cpi_yoy"].values.astype(np.float64),
        delta_train_full=sub_df.loc[mask_train_full, delta_col].values.astype(np.float64),
        X_subtrain=sub_df.loc[mask_subtrain, TABULAR_FEATURES].copy(),
        X_seq_subtrain=X_seq_all[mask_subtrain],
        delta_subtrain=sub_df.loc[mask_subtrain, delta_col].values.astype(np.float64),
        X_val=sub_df.loc[mask_val, TABULAR_FEATURES].copy(),
        X_seq_val=X_seq_all[mask_val],
        y_val=sub_df.loc[mask_val, target_col].values.astype(np.float64),
        anchor_val=sub_df.loc[mask_val, "cpi_yoy"].values.astype(np.float64),
        delta_val=sub_df.loc[mask_val, delta_col].values.astype(np.float64),
        dates_test=sub_df.index[mask_test],
        target_dates_test=pd.DatetimeIndex(realized_dates[mask_test]),
        regimes_test=sub_df.loc[mask_test, "regime"].copy(),
        X_test=sub_df.loc[mask_test, TABULAR_FEATURES].copy(),
        X_seq_test=X_seq_all[mask_test],
        y_test=sub_df.loc[mask_test, target_col].values.astype(np.float64),
        anchor_test=sub_df.loc[mask_test, "cpi_yoy"].values.astype(np.float64),
        delta_test=sub_df.loc[mask_test, delta_col].values.astype(np.float64),
    )
