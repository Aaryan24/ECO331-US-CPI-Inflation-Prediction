"""
Live Multi-Model Inflation Predictor Engine for Interactive Demo Website.

Trains (once) and caches all 12 econometric, regularized linear, tree ensemble,
deep sequence, and hybrid `XGB + Deep Net` models across horizons h in {1, 3, 6, 12}
months, enabling <15ms real-time inference and live exact XGBoost TreeSHAP explanations.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from xgboost import XGBRegressor

from src import OUTPUTS_DIR, TABLES_DIR
from src.data_loader import load_macro_dataset
from src.features import (
    AR_BENCHMARK_FEATURES,
    FEATURE_LABELS,
    HORIZONS,
    PHILLIPS_CURVE_FEATURES,
    SEQ_LEN,
    SEQUENCE_FEATURES,
    TABULAR_FEATURES,
    build_feature_matrix,
    prepare_horizon_dataset,
)
from src.models import (
    InterpretableTemporalTransformer,
    MODEL_FAMILIES,
    MacroLSTM,
    MacroMLP,
    XGBDeepTrendNet,
    _extract_causal_xgb_hybrid_features,
    set_global_seed,
)

torch.set_num_threads(1)

BUNDLE_PATH: Path = OUTPUTS_DIR / "demo_predictor_bundle.pkl"

# Clean, intuitive slider definitions grouped for a clutter-free classroom demo
SLIDER_GROUPS: List[Dict[str, Any]] = [
    {
        "group_id": "price_base",
        "title": "1. Current Inflation & Base Effect",
        "sliders": [
            {
                "key": "cpi_yoy",
                "label": "Today's Headline CPI (YoY %)",
                "min": -1.0,
                "max": 12.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Starting anchor inflation rate right now",
            },
            {
                "key": "base_3m_ann",
                "label": "Base-Effect CPI (1 Yr Ago, %)",
                "min": -5.0,
                "max": 14.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Old spike from 10–12m ago rolling out of YoY math",
            },
        ],
    },
    {
        "group_id": "supply_oil",
        "title": "2. Supply, Electricity & Oil",
        "sliders": [
            {
                "key": "elec_yoy",
                "label": "Electricity CPI Inflation (YoY %)",
                "min": -4.0,
                "max": 18.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Sticky utility grid cost pass-through (Kuskaya 2025)",
            },
            {
                "key": "ppi_yoy",
                "label": "PPI Producer Inflation (YoY %)",
                "min": -10.0,
                "max": 24.0,
                "step": 0.2,
                "unit": "%",
                "hint": "Factory/wholesale cost pressure (>5% warns of pass-through)",
            },
            {
                "key": "oil_yoy",
                "label": "WTI Crude Oil Change (YoY %)",
                "min": -55.0,
                "max": 100.0,
                "step": 1.0,
                "unit": "%",
                "hint": "Immediate pump & transport energy shock",
            },
        ],
    },
    {
        "group_id": "policy_money",
        "title": "3. Fed Policy, M2 & Stock Market",
        "sliders": [
            {
                "key": "real_fedfunds",
                "label": "Real Fed Funds Rate (Fed − CPI, pp)",
                "min": -8.0,
                "max": 7.0,
                "step": 0.1,
                "unit": " pp",
                "hint": "Below -2 pp fuels inflation; above 0 pp brakes it",
            },
            {
                "key": "m2_yoy",
                "label": "M2 Money Supply Growth (YoY %)",
                "min": -4.0,
                "max": 25.0,
                "step": 0.2,
                "unit": "%",
                "hint": "Broad liquidity fuel (>10% sustains demand pressure)",
            },
            {
                "key": "stock_yoy",
                "label": "US Stock Market Return (YoY %)",
                "min": -35.0,
                "max": 45.0,
                "step": 0.5,
                "unit": "%",
                "hint": "Financial conditions & household wealth effect (Vamossy 2024)",
            },
        ],
    },
    {
        "group_id": "expect_slack",
        "title": "4. Expectations, GDP & Labor",
        "sliders": [
            {
                "key": "mich_exp_1y",
                "label": "1Y Inflation Expectations (Mich. %)",
                "min": 1.5,
                "max": 8.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Consumer wage-price anchor (critical at 12m horizon)",
            },
            {
                "key": "gdp_yoy",
                "label": "Real GDP Growth (YoY %, Lag 1Q)",
                "min": -8.0,
                "max": 12.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Broad service + goods aggregate demand (Yoon 2021)",
            },
            {
                "key": "unrate",
                "label": "Unemployment Rate (%)",
                "min": 3.0,
                "max": 11.0,
                "step": 0.1,
                "unit": "%",
                "hint": "Phillips Curve labor slack gauge",
            },
        ],
    },
]

ADVANCED_SLIDERS: List[Dict[str, Any]] = [
    {
        "key": "cpi_mom_ann",
        "label": "CPI 1-Month Momentum (Ann. %)",
        "min": -8.0,
        "max": 15.0,
        "step": 0.2,
        "unit": "%",
        "hint": "Latest single-month annualized CPI speed",
    },
    {
        "key": "core_headline_gap",
        "label": "Core − Headline CPI Gap (pp)",
        "min": -3.0,
        "max": 3.0,
        "step": 0.1,
        "unit": " pp",
        "hint": "Negative when oil/food temporarily push Headline above Core",
    },
    {
        "key": "indpro_yoy",
        "label": "Industrial Production (YoY %)",
        "min": -12.0,
        "max": 12.0,
        "step": 0.2,
        "unit": "%",
        "hint": "Real factory output growth (<0% reinforces disinflation)",
    },
]

PRESET_DEFINITIONS: List[Dict[str, str]] = [
    {
        "id": "latest_2026_09",
        "date": "2026-09-01",
        "badge": "Live Snapshot",
        "title": "Sep 2026 (Latest FRED)",
        "description": "Most recent US macro reading in dataset — pure forward what-if forecast.",
    },
    {
        "id": "surge_2021_03",
        "date": "2021-03-01",
        "badge": "Surge Breakout",
        "title": "Mar 2021 (Reopening Surge)",
        "description": "Stimulus + M2 boom + PPI jumping to +11.3% just before inflation broke out.",
    },
    {
        "id": "peak_2022_06",
        "date": "2022-06-01",
        "badge": "9.0% Peak",
        "title": "Jun 2022 (Peak 9% CPI)",
        "description": "40-year CPI peak (8.98%) after Ukraine energy shock & +22.4% PPI inflation.",
    },
    {
        "id": "hikes_2022_12",
        "date": "2022-12-01",
        "badge": "Fed 75bp Turn",
        "title": "Dec 2022 (Fed Hikes Disinflation)",
        "description": "After four 75bp Fed rate hikes + huge 10.1% base effect poised to roll off.",
    },
    {
        "id": "pivot_2024_09",
        "date": "2024-09-01",
        "badge": "Rate-Cut Pivot",
        "title": "Sep 2024 (Soft Landing Pivot)",
        "description": "Inflation cooled to 2.43% with +2.7 pp restrictive real rates as Fed began cuts.",
    },
]


def _train_torch_single(
    builder,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    lr: float = 2.0e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std = X_tr.std(axis=0, keepdims=True) + 1e-6
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) + 1e-6

    X_s = ((X_tr - x_mean) / x_std).astype(np.float32)
    y_s = ((y_tr - y_mean) / y_std).astype(np.float32)

    set_global_seed(seed)
    net = builder()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2.0e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.75)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_s)),
        batch_size=64,
        shuffle=True,
    )
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    net.eval()
    return {
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def build_or_load_predictor_bundle(force_rebuild: bool = False) -> Dict[str, Any]:
    """Load cached multi-horizon model bundle or train & save it if not yet cached."""
    if BUNDLE_PATH.exists() and not force_rebuild:
        with open(BUNDLE_PATH, "rb") as f:
            return pickle.load(f)

    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    set_global_seed(42)
    raw_panel = load_macro_dataset(force_refresh=False)
    feat_df = build_feature_matrix(raw_panel)

    # Out-of-sample metrics table for accuracy badges
    metrics_path = TABLES_DIR / "metrics_all_horizons.csv"
    all_metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    # Build preset snapshots (full 21 tabular features + 12x12 sequence + realized future CPIs)
    valid_tab = feat_df.dropna(subset=TABULAR_FEATURES).copy()
    presets_payload: Dict[str, Dict[str, Any]] = {}

    for pdef in PRESET_DEFINITIONS:
        dt = pd.Timestamp(pdef["date"])
        if dt not in valid_tab.index:
            dt = valid_tab.index[-1]
        pos = valid_tab.index.get_loc(dt)
        seq_slice = valid_tab.iloc[max(0, pos - SEQ_LEN + 1) : pos + 1][SEQUENCE_FEATURES].values.astype(np.float32)
        row_s = valid_tab.loc[dt]

        realized_by_h: Dict[int, Optional[float]] = {}
        for h in HORIZONS:
            val = row_s.get(f"target_cpi_h{h}", np.nan)
            realized_by_h[h] = round(float(val), 2) if pd.notna(val) else None

        # Recent 3 months of tabular rows for hybrid rolling/trend context
        hist3_df = valid_tab.iloc[max(0, pos - 3) : pos][TABULAR_FEATURES].copy()

        presets_payload[pdef["id"]] = {
            **pdef,
            "actual_date": str(dt.date()),
            "tabular_row": {c: float(row_s[c]) for c in TABULAR_FEATURES},
            "sequence_12x12": seq_slice,
            "prev3_tabular": hist3_df.to_dict(orient="records"),
            "realized_cpi_by_horizon": realized_by_h,
        }

    # Train full-history models for each horizon h in {1, 3, 6, 12}
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_cfg = {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 190,
        "min_child_weight": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 4.0,
    }
    rf_cfg = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5, "max_features": 0.65}

    models_by_horizon: Dict[int, Dict[str, Any]] = {}
    for h in HORIZONS:
        ds = prepare_horizon_dataset(feat_df, horizon=h)
        X_all = pd.concat([ds.X_train_full, ds.X_test], axis=0)
        X_seq_all = np.concatenate([ds.X_seq_train_full, ds.X_seq_test], axis=0)
        d_all = np.concatenate([ds.delta_train_full, ds.delta_test], axis=0)

        tr_mean = X_all.mean()
        tr_sd = X_all.std() + 1e-6

        ar_m = LinearRegression().fit(X_all[AR_BENCHMARK_FEATURES], d_all)
        pc_m = LinearRegression().fit(X_all[PHILLIPS_CURVE_FEATURES], d_all)

        sc_tab = StandardScaler().fit(X_all)
        X_all_sc = sc_tab.transform(X_all)
        ridge_m = RidgeCV(alphas=np.logspace(-0.5, 3.5, 35), cv=tscv).fit(X_all_sc, d_all)
        enet_m = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95],
            alphas=np.logspace(-2.2, 0.8, 30),
            cv=tscv,
            max_iter=6000,
            random_state=42,
        ).fit(X_all_sc, d_all)

        rf_m = RandomForestRegressor(**rf_cfg, random_state=42, n_jobs=1).fit(X_all, d_all)
        xgb_m = XGBRegressor(
            **xgb_cfg,
            subsample=0.80,
            colsample_bytree=0.75,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ).fit(X_all, d_all)

        n_tab = X_all.shape[1]
        n_seq_ch = X_seq_all.shape[2]

        mlp_pack = _train_torch_single(
            lambda: MacroMLP(in_features=n_tab, hidden_dim=64, dropout=0.22),
            X_all.values,
            d_all,
            epochs=55,
        )
        lstm_pack = _train_torch_single(
            lambda: MacroLSTM(in_channels=n_seq_ch, hidden_dim=32, dropout=0.20),
            X_seq_all,
            d_all,
            epochs=50,
        )
        tf_pack = _train_torch_single(
            lambda: InterpretableTemporalTransformer(
                in_channels=n_seq_ch, seq_len=SEQ_LEN, d_model=32, n_heads=4, dropout=0.20
            ),
            X_seq_all,
            d_all,
            epochs=45,
            lr=1.8e-3,
        )

        # Hybrid `XGB + Deep Net`
        Z_tr, oof_pred, valid_oof, _, _, _ = _extract_causal_xgb_hybrid_features(
            X_all, d_all, X_all.iloc[-2:], xgb_cfg, horizon=h
        )
        top_idx = np.argsort(xgb_m.feature_importances_)[-10:]
        sc_hyb = StandardScaler().fit(Z_tr[valid_oof])
        Z_tr_s = sc_hyb.transform(Z_tr[valid_oof]).astype(np.float32)

        set_global_seed(42)
        hyb_net = XGBDeepTrendNet(in_dim=Z_tr_s.shape[1], hidden_dim=32, dropout=0.25)
        opt_h = torch.optim.AdamW(hyb_net.parameters(), lr=2.0e-3, weight_decay=3.0e-2)
        loss_h = nn.SmoothL1Loss(beta=0.60)
        zt = torch.from_numpy(Z_tr_s)
        bt = torch.from_numpy(oof_pred[valid_oof].astype(np.float32))
        yt = torch.from_numpy(d_all[valid_oof].astype(np.float32))
        hyb_net.train()
        for _ in range(75):
            opt_h.zero_grad()
            l_val = loss_h(hyb_net(zt, bt), yt)
            l_val.backward()
            nn.utils.clip_grad_norm_(hyb_net.parameters(), 1.0)
            opt_h.step()
        hyb_net.eval()

        # Out-of-sample RMSE lookup for this horizon
        oos_rmse_map: Dict[str, float] = {}
        top_oos_model = "XGBoost"
        if not all_metrics_df.empty:
            sub_m = all_metrics_df[all_metrics_df["Horizon_Months"] == h].sort_values("RMSE")
            if not sub_m.empty:
                top_oos_model = str(sub_m.iloc[0]["Model"])
                for _, mr in sub_m.iterrows():
                    oos_rmse_map[str(mr["Model"])] = round(float(mr["RMSE"]), 3)

        models_by_horizon[h] = {
            "tr_mean": tr_mean.to_dict(),
            "tr_sd": tr_sd.to_dict(),
            "ar_m": ar_m,
            "pc_m": pc_m,
            "sc_tab": sc_tab,
            "ridge_m": ridge_m,
            "enet_m": enet_m,
            "rf_m": rf_m,
            "xgb_m": xgb_m,
            "mlp_pack": mlp_pack,
            "lstm_pack": lstm_pack,
            "tf_pack": tf_pack,
            "hyb_pack": {
                "state_dict": {k: v.cpu() for k, v in hyb_net.state_dict().items()},
                "in_dim": int(Z_tr_s.shape[1]),
                "top_idx": top_idx,
                "sc_hyb": sc_hyb,
            },
            "oos_rmse_map": oos_rmse_map,
            "top_oos_model": top_oos_model,
        }

    bundle = {
        "presets": presets_payload,
        "models_by_horizon": models_by_horizon,
        "feature_names": TABULAR_FEATURES,
        "sequence_features": SEQUENCE_FEATURES,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


class LiveInflationPredictor:
    """Fast in-memory inference wrapper serving live predictions & TreeSHAP attributions."""

    def __init__(self, bundle: Optional[Dict[str, Any]] = None):
        self.bundle = bundle if bundle is not None else build_or_load_predictor_bundle()
        self._rehydrated_nets: Dict[int, Dict[str, nn.Module]] = {}
        for h, hp in self.bundle["models_by_horizon"].items():
            n_tab = len(self.bundle["feature_names"])
            n_seq = len(self.bundle["sequence_features"])

            mlp = MacroMLP(in_features=n_tab, hidden_dim=64, dropout=0.22)
            mlp.load_state_dict(hp["mlp_pack"]["state_dict"])
            mlp.eval()

            lstm = MacroLSTM(in_channels=n_seq, hidden_dim=32, dropout=0.20)
            lstm.load_state_dict(hp["lstm_pack"]["state_dict"])
            lstm.eval()

            tf = InterpretableTemporalTransformer(
                in_channels=n_seq, seq_len=SEQ_LEN, d_model=32, n_heads=4, dropout=0.20
            )
            tf.load_state_dict(hp["tf_pack"]["state_dict"])
            tf.eval()

            hyb = XGBDeepTrendNet(in_dim=hp["hyb_pack"]["in_dim"], hidden_dim=32, dropout=0.25)
            hyb.load_state_dict(hp["hyb_pack"]["state_dict"])
            hyb.eval()

            self._rehydrated_nets[h] = {"mlp": mlp, "lstm": lstm, "tf": tf, "hyb": hyb}

    def get_metadata(self) -> Dict[str, Any]:
        presets_clean = []
        all_slider_keys = [s["key"] for g in SLIDER_GROUPS for s in g["sliders"]] + [
            s["key"] for s in ADVANCED_SLIDERS
        ]
        for pdef in PRESET_DEFINITIONS:
            p = self.bundle["presets"][pdef["id"]]
            presets_clean.append(
                {
                    "id": p["id"],
                    "date": p["actual_date"],
                    "badge": p["badge"],
                    "title": p["title"],
                    "description": p["description"],
                    "values": {k: round(float(p["tabular_row"][k]), 2) for k in all_slider_keys},
                    "realized_cpi_by_horizon": p["realized_cpi_by_horizon"],
                }
            )
        return {
            "horizons": [
                {"value": 1, "label": "1 Month", "tag": "Nowcast"},
                {"value": 3, "label": "3 Months", "tag": "Flagship"},
                {"value": 6, "label": "6 Months", "tag": "2 Quarters"},
                {"value": 12, "label": "12 Months", "tag": "1 Year"},
            ],
            "slider_groups": SLIDER_GROUPS,
            "advanced_sliders": ADVANCED_SLIDERS,
            "presets": presets_clean,
        }

    def predict(
        self,
        horizon: int = 3,
        preset_id: str = "latest_2026_09",
        overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        h = horizon if horizon in HORIZONS else 3
        presets = self.bundle["presets"]
        preset = presets.get(preset_id, presets["latest_2026_09"])
        hp = self.bundle["models_by_horizon"][h]
        nets = self._rehydrated_nets[h]

        row_dict = dict(preset["tabular_row"])
        seq_arr = preset["sequence_12x12"].copy()
        overrides = overrides or {}

        # Detect whether user modified any slider away from preset defaults
        is_modified = False
        orig_cpi = row_dict["cpi_yoy"]
        orig_mom_explicit = "cpi_mom_ann" in overrides and abs(
            float(overrides["cpi_mom_ann"]) - row_dict["cpi_mom_ann"]
        ) > 0.05

        for k, v in overrides.items():
            if k in row_dict and v is not None:
                fv = float(v)
                if abs(fv - row_dict[k]) > 0.05:
                    is_modified = True
                row_dict[k] = fv

        # Smoothly propagate headline CPI adjustments into coupled momentum if user only moved main sliders
        cpi_diff = row_dict["cpi_yoy"] - orig_cpi
        if abs(cpi_diff) > 1e-4:
            if not orig_mom_explicit:
                row_dict["cpi_mom_ann"] += cpi_diff
            row_dict["cpi_3m_ann"] += cpi_diff * 0.85

        # Update current month t (last time step) of the 12-month sequence for LSTM & Transformer
        for ch_i, col_name in enumerate(self.bundle["sequence_features"]):
            seq_arr[-1, ch_i] = float(row_dict[col_name])

        x_df = pd.DataFrame([row_dict], columns=self.bundle["feature_names"])
        anchor = float(row_dict["cpi_yoy"])

        # Winsorized copy for unregularized linear stability
        tr_mean = pd.Series(hp["tr_mean"])
        tr_sd = pd.Series(hp["tr_sd"])
        x_clip = x_df.clip(lower=tr_mean - 4.5 * tr_sd, upper=tr_mean + 4.5 * tr_sd, axis=1)

        preds_cpi: Dict[str, float] = {}
        preds_cpi["Random Walk"] = anchor
        preds_cpi["AR(p) Benchmark"] = anchor + float(hp["ar_m"].predict(x_clip[AR_BENCHMARK_FEATURES])[0])
        preds_cpi["OLS Phillips Curve"] = anchor + float(hp["pc_m"].predict(x_clip[PHILLIPS_CURVE_FEATURES])[0])

        x_sc = np.clip(hp["sc_tab"].transform(x_df), -4.5, 4.5)
        preds_cpi["Ridge (L2)"] = anchor + float(hp["ridge_m"].predict(x_sc)[0])
        preds_cpi["Elastic Net (L1+L2)"] = anchor + float(hp["enet_m"].predict(x_sc)[0])

        preds_cpi["Random Forest"] = anchor + float(hp["rf_m"].predict(x_df)[0])

        xgb_m = hp["xgb_m"]
        xgb_delta = float(xgb_m.predict(x_df)[0])
        preds_cpi["XGBoost"] = anchor + xgb_delta

        # Exact live C++ TreeSHAP contributions on this exact scenario!
        dmat = xgb.DMatrix(x_df, feature_names=self.bundle["feature_names"])
        contribs = xgb_m.get_booster().predict(dmat, pred_contribs=True)[0]
        shap_vals = contribs[:-1]
        shap_base = float(contribs[-1])

        # Deep Neural: MLP, LSTM, Temporal Transformer
        with torch.no_grad():
            mp = hp["mlp_pack"]
            xm = np.clip((x_df.values - mp["x_mean"]) / mp["x_std"], -4.5, 4.5).astype(np.float32)
            mlp_delta = float(nets["mlp"](torch.from_numpy(xm)).item() * mp["y_std"] + mp["y_mean"])
            preds_cpi["MLP (Deep Net)"] = anchor + mlp_delta

            lp = hp["lstm_pack"]
            xl = np.clip((seq_arr[None, :, :] - lp["x_mean"]) / lp["x_std"], -4.5, 4.5).astype(np.float32)
            lstm_delta = float(nets["lstm"](torch.from_numpy(xl)).item() * lp["y_std"] + lp["y_mean"])
            preds_cpi["LSTM (Sequence)"] = anchor + lstm_delta

            tp = hp["tf_pack"]
            xt = np.clip((seq_arr[None, :, :] - tp["x_mean"]) / tp["x_std"], -4.5, 4.5).astype(np.float32)
            tf_delta = float(nets["tf"](torch.from_numpy(xt)).item() * tp["y_std"] + tp["y_mean"])
            preds_cpi["Temporal Transformer"] = anchor + tf_delta

            # Hybrid `XGB + Deep Net`
            hyb_p = hp["hyb_pack"]
            top_idx = hyb_p["top_idx"]
            prev_rows = preset["prev3_tabular"]
            if len(prev_rows) > 0:
                prev_df = pd.DataFrame(prev_rows, columns=self.bundle["feature_names"])
                prev_xgb = xgb_m.predict(prev_df).astype(np.float32)
                s_recent = np.concatenate([prev_xgb, np.array([xgb_delta], dtype=np.float32)])
            else:
                s_recent = np.array([xgb_delta], dtype=np.float32)

            roll3 = float(np.mean(s_recent[-3:]))
            diff1 = float(s_recent[-1] - s_recent[-2]) if len(s_recent) >= 2 else 0.0
            diff3 = float(s_recent[-1] - s_recent[-4]) if len(s_recent) >= 4 else diff1

            z_raw = np.hstack(
                [
                    x_df.values[0, top_idx],
                    shap_vals[top_idx],
                    [xgb_delta, roll3, diff1, diff3],
                ]
            )[None, :].astype(np.float32)
            z_sc = np.clip(hyb_p["sc_hyb"].transform(z_raw), -4.5, 4.5).astype(np.float32)
            hyb_delta = float(
                nets["hyb"](
                    torch.from_numpy(z_sc),
                    torch.tensor([xgb_delta], dtype=torch.float32),
                ).item()
            )
            preds_cpi["XGB + Deep Net"] = anchor + hyb_delta

        preds_cpi["Forecast Combination"] = (
            preds_cpi["Elastic Net (L1+L2)"] + preds_cpi["XGBoost"] + preds_cpi["LSTM (Sequence)"]
        ) / 3.0

        oos_map = hp["oos_rmse_map"]
        top_model_name = hp["top_oos_model"]

        model_rows = []
        for m_name, pred_val in preds_cpi.items():
            rmse_v = oos_map.get(m_name, 9.99)
            model_rows.append(
                {
                    "model": m_name,
                    "family": MODEL_FAMILIES[m_name],
                    "predicted_cpi": round(pred_val, 2),
                    "predicted_change_pp": round(pred_val - anchor, 2),
                    "oos_rmse_pp": rmse_v,
                    "is_recommended": m_name == top_model_name,
                }
            )
        model_rows.sort(key=lambda r: r["oos_rmse_pp"])

        non_rw_vals = [r["predicted_cpi"] for r in model_rows if r["model"] != "Random Walk"]
        consensus_mean = round(float(np.mean(non_rw_vals)), 2)

        # Top 7 live SHAP drivers sorted by |SHAP|
        order = np.argsort(np.abs(shap_vals))[::-1][:7]
        shap_drivers = []
        for idx in order:
            fname = self.bundle["feature_names"][int(idx)]
            shap_drivers.append(
                {
                    "key": fname,
                    "label": FEATURE_LABELS.get(fname, fname),
                    "feature_value": round(float(row_dict[fname]), 2),
                    "shap_pp": round(float(shap_vals[int(idx)]), 3),
                }
            )

        actual_realized = preset["realized_cpi_by_horizon"].get(h) if not is_modified else None

        return {
            "horizon": h,
            "preset_id": preset["id"],
            "preset_title": preset["title"],
            "is_modified": is_modified,
            "current_cpi": round(anchor, 2),
            "recommended_model": top_model_name,
            "recommended_cpi": round(preds_cpi[top_model_name], 2),
            "recommended_change_pp": round(preds_cpi[top_model_name] - anchor, 2),
            "consensus_cpi": consensus_mean,
            "consensus_change_pp": round(consensus_mean - anchor, 2),
            "consensus_min_cpi": round(float(np.min(non_rw_vals)), 2),
            "consensus_max_cpi": round(float(np.max(non_rw_vals)), 2),
            "actual_realized_cpi": actual_realized,
            "xgb_predicted_cpi": round(preds_cpi["XGBoost"], 2),
            "xgb_base_shift_pp": round(shap_base, 3),
            "models": model_rows,
            "shap_drivers": shap_drivers,
        }
