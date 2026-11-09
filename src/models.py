"""
Econometric, Regularized Linear, Tree Ensemble, and Deep Neural/Sequence Models.

Implements 10 forecasting specifications across 4 methodological families (plus an
equal-weight cross-family Forecast Combination inspired by Singh & Bhoi, RBI 2022),
evaluated via causal annual expanding-window out-of-sample forecasting (with strict
h-month target embargo before every re-estimation boundary) and Diebold-Mariano
(Harvey-Leybourne-Newbold HAC) statistical significance tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from xgboost import XGBRegressor

from src.features import (
    AR_BENCHMARK_FEATURES,
    HorizonDataset,
    PHILLIPS_CURVE_FEATURES,
)

torch.set_num_threads(1)


def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Statistical Evaluation & Diebold-Mariano (HLN-adjusted HAC) Test
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_bench: np.ndarray,
    horizon: int = 3,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) finite-sample
    correction and Bartlett (Newey-West) HAC covariance up to lag (h - 1).
    """
    e_m = y_true - y_pred_model
    e_b = y_true - y_pred_bench
    d = (e_m ** 2) - (e_b ** 2)
    n = len(d)
    if np.allclose(d, 0.0) or n < 10:
        return 0.0, 1.0

    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    gamma_0 = float(np.sum(d_centered ** 2) / n)
    var_d = gamma_0
    max_lag = max(0, horizon - 1)
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)
        gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / n)
        var_d += 2.0 * weight * gamma_k

    var_d = max(var_d, 1e-12)
    dm_raw = d_mean / np.sqrt(var_d / n)
    hln_factor = np.sqrt(max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-6))
    dm_stat = float(dm_raw * hln_factor)
    p_val = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    return dm_stat, p_val


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchor: np.ndarray,
    y_bench_rw: np.ndarray,
    horizon: int,
) -> Dict[str, float]:
    """Compute comprehensive accuracy, directional, and statistical significance metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_rw = float(np.sqrt(mean_squared_error(y_true, y_bench_rw)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    actual_dir = np.sign(y_true - anchor)
    pred_dir = np.sign(y_pred - anchor)
    non_zero = actual_dir != 0
    dir_acc = float(np.mean(actual_dir[non_zero] == pred_dir[non_zero]) * 100.0) if np.any(non_zero) else 50.0

    dm_stat, dm_p = diebold_mariano_test(y_true, y_pred, y_bench_rw, horizon=horizon)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Rel_RMSE_vs_RW": rmse / rmse_rw if rmse_rw > 0 else 1.0,
        "R2": r2,
        "Directional_Acc_pct": dir_acc,
        "DM_Stat_vs_RW": dm_stat,
        "DM_p_value": dm_p,
    }


# ---------------------------------------------------------------------------
# PyTorch Neural & Sequence Architectures (MLP, LSTM, Temporal Transformer)
# ---------------------------------------------------------------------------


class MacroMLP(nn.Module):
    """Feedforward Deep Neural Network with LayerNorm and Dropout."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MacroLSTM(nn.Module):
    """Sequence LSTM over 12-month macroeconomic lookback windows."""

    def __init__(self, in_channels: int, hidden_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        last_state = self.drop(self.norm(out[:, -1, :]))
        return self.head(last_state).squeeze(-1)


class InterpretableTemporalTransformer(nn.Module):
    """
    Lightweight 1-layer Multi-Head Self-Attention + Temporal Attention Pooling
    over 12-month macroeconomic sequences, exposing explicit per-lag attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int = 12,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.temporal_query = nn.Linear(d_model, d_model)
        self.temporal_key = nn.Linear(d_model, d_model)
        self.scale = float(np.sqrt(d_model))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward_with_attention(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x_seq) + self.pos_embed
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))

        q = self.temporal_query(h[:, -1:, :])
        k = self.temporal_key(h)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / self.scale
        lag_weights = torch.softmax(scores, dim=-1)

        pooled = torch.bmm(lag_weights.unsqueeze(1), h).squeeze(1)
        pred = self.head(pooled).squeeze(-1)
        return pred, lag_weights

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        pred, _ = self.forward_with_attention(x_seq)
        return pred


class XGBDeepTrendNet(nn.Module):
    """
    Gated Hybrid Tree-Neural Architecture (`XGB + Deep Net`).

    Takes standardized [Top-K Raw Macro || Top-K XGBoost Nonlinear SHAP Signals ||
    XGBoost 1m/3m Trend & Rolling Signals] and learns a gated smooth trend refinement
    around XGBoost's nonlinear tree anchor:
        delta_hat = w(z) * delta_xgb + (1 - w(z)) * trend_head(z).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.25):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.trend_head = nn.Linear(hidden_dim // 2, 1)
        self.gate = nn.Linear(hidden_dim // 2, 1)
        nn.init.zeros_(self.trend_head.bias)
        nn.init.normal_(self.trend_head.weight, std=0.02)
        nn.init.constant_(self.gate.bias, 1.1)

    def forward_with_gate(self, z: torch.Tensor, xgb_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_rep = self.backbone(z)
        w = torch.sigmoid(self.gate(h_rep)).squeeze(-1)
        nn_trend = self.trend_head(h_rep).squeeze(-1)
        out = w * xgb_base + (1.0 - w) * nn_trend
        return out, w

    def forward(self, z: torch.Tensor, xgb_base: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_gate(z, xgb_base)
        return out


def _extract_causal_xgb_hybrid_features(
    X_tr_df: pd.DataFrame,
    d_tr: np.ndarray,
    X_eval_df: pd.DataFrame,
    xgb_cfg: Dict[str, Any],
    horizon: int,
    top_k_count: int = 10,
    return_shap_meta: bool = False,
):
    """
    Build leak-free hybrid feature matrices for `XGB + Deep Net`:
      - Out-of-fold causal XGBoost predictions & exact TreeSHAP values on `X_tr_df`
      - Full-fit causal XGBoost predictions & exact TreeSHAP values on `X_eval_df`
      - 1-month & 3-month temporal trends + 3-month rolling average of XGBoost's signal
    """
    n_tr, n_f = X_tr_df.shape
    oof_shap = np.zeros((n_tr, n_f), dtype=np.float32)
    oof_pred = np.zeros(n_tr, dtype=np.float32)
    valid_oof = np.zeros(n_tr, dtype=bool)

    for ti, vi in TimeSeriesSplit(n_splits=5).split(X_tr_df):
        tp = ti[:-horizon] if len(ti) > horizon + 20 else ti
        mf = XGBRegressor(
            **xgb_cfg,
            subsample=0.80,
            colsample_bytree=0.75,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ).fit(X_tr_df.iloc[tp], d_tr[tp])
        oof_pred[vi] = mf.predict(X_tr_df.iloc[vi])
        dm_v = xgb.DMatrix(X_tr_df.iloc[vi], feature_names=list(X_tr_df.columns))
        oof_shap[vi] = mf.get_booster().predict(dm_v, pred_contribs=True)[:, :-1]
        valid_oof[vi] = True

    m_full = XGBRegressor(
        **xgb_cfg,
        subsample=0.80,
        colsample_bytree=0.75,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    ).fit(X_tr_df, d_tr)

    eval_pred = m_full.predict(X_eval_df).astype(np.float32)
    dm_eval = xgb.DMatrix(X_eval_df, feature_names=list(X_eval_df.columns))
    eval_shap = m_full.get_booster().predict(dm_eval, pred_contribs=True)[:, :-1].astype(np.float32)

    top_idx = np.argsort(m_full.feature_importances_)[-top_k_count:]
    raw_cols = list(X_tr_df.columns)
    hybrid_col_names = (
        [f"Raw: {raw_cols[i]}" for i in top_idx]
        + [f"XGB-SHAP: {raw_cols[i]}" for i in top_idx]
        + ["XGB Signal (t)", "XGB 3m Rolling Mean", "XGB 1m Trend", "XGB 3m Trend"]
    )

    # Compute causal temporal trends across the concatenated timeline
    # Fill initial burn-in fold smoothly for rolling/diff calculation only (excluded from training loss via `valid_oof`)
    burn_pred = m_full.predict(X_tr_df.iloc[~valid_oof]).astype(np.float32) * 0.85
    timeline_pred = oof_pred.copy()
    timeline_pred[~valid_oof] = burn_pred
    s_all = pd.Series(np.concatenate([timeline_pred, eval_pred]))

    roll3 = s_all.rolling(3, min_periods=1).mean().values[:, None]
    diff1 = (s_all - s_all.shift(1)).fillna(0.0).values[:, None]
    diff3 = (s_all - s_all.shift(3)).fillna(0.0).values[:, None]

    Z_tr = np.hstack(
        [
            X_tr_df.values[:, top_idx],
            oof_shap[:, top_idx],
            timeline_pred[:, None],
            roll3[:n_tr],
            diff1[:n_tr],
            diff3[:n_tr],
        ]
    ).astype(np.float32)

    Z_eval = np.hstack(
        [
            X_eval_df.values[:, top_idx],
            eval_shap[:, top_idx],
            eval_pred[:, None],
            roll3[n_tr:],
            diff1[n_tr:],
            diff3[n_tr:],
        ]
    ).astype(np.float32)

    if return_shap_meta:
        return Z_tr, oof_pred, valid_oof, Z_eval, eval_pred, hybrid_col_names, eval_shap, top_idx
    return Z_tr, oof_pred, valid_oof, Z_eval, eval_pred, hybrid_col_names


def _fit_and_predict_xgb_deep_net(
    X_tr_df: pd.DataFrame,
    d_tr: np.ndarray,
    X_eval_df: pd.DataFrame,
    xgb_cfg: Dict[str, Any],
    horizon: int,
    epochs: int = 75,
    seeds: Tuple[int, ...] = (42, 2026),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Train `XGBDeepTrendNet` on causal out-of-fold XGBoost + trend features and predict on `X_eval_df`."""
    Z_tr, oof_pred, valid_oof, Z_eval, eval_xgb_pred, col_names, eval_shap, top_idx = _extract_causal_xgb_hybrid_features(
        X_tr_df, d_tr, X_eval_df, xgb_cfg, horizon=horizon, return_shap_meta=True
    )
    n_eval, n_raw = X_eval_df.shape
    k_top = len(top_idx)

    sc = StandardScaler().fit(Z_tr[valid_oof])
    Z_tr_s = sc.transform(Z_tr[valid_oof]).astype(np.float32)
    Z_eval_s = np.clip(sc.transform(Z_eval), -4.5, 4.5).astype(np.float32)

    zt = torch.from_numpy(Z_tr_s)
    bt = torch.from_numpy(oof_pred[valid_oof].astype(np.float32))
    yt = torch.from_numpy(d_tr[valid_oof].astype(np.float32))

    z_eval_t = torch.from_numpy(Z_eval_s)
    b_eval_t = torch.from_numpy(eval_xgb_pred.astype(np.float32))
    loss_fn = nn.SmoothL1Loss(beta=0.60)

    abs_shap = np.abs(eval_shap)
    shap_row_shares = abs_shap / (abs_shap.sum(axis=1, keepdims=True) + 1e-8)

    preds = []
    gates = []
    grad_imps = []
    raw24_attrs = []

    for sd in seeds:
        set_global_seed(sd)
        net = XGBDeepTrendNet(in_dim=Z_tr_s.shape[1], hidden_dim=32, dropout=0.25)
        opt = torch.optim.AdamW(net.parameters(), lr=2.0e-3, weight_decay=3.0e-2)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(zt, bt), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        # Chain-rule attribution wrt hybrid inputs Z_eval_s AND gated XGBoost anchor b_eval_t
        z_req = z_eval_t.clone().detach().requires_grad_(True)
        b_req = b_eval_t.clone().detach().requires_grad_(True)
        out_sum = net(z_req, b_req).sum()
        out_sum.backward()
        grad_z = z_req.grad.detach().numpy()
        grad_b = b_req.grad.detach().numpy()
        z_contrib = np.abs(grad_z * Z_eval_s)
        grad_imps.append(np.mean(z_contrib, axis=0))

        # Map hybrid attribution cleanly onto all 24 original economic features (in pp)
        feat_attr = np.abs(grad_b[:, None]) * abs_shap
        for j_pos, raw_j in enumerate(top_idx):
            feat_attr[:, raw_j] += z_contrib[:, j_pos] + z_contrib[:, k_top + j_pos]
        agg_trend_contrib = z_contrib[:, 2 * k_top :].sum(axis=1, keepdims=True)
        feat_attr += agg_trend_contrib * shap_row_shares
        raw24_attrs.append(feat_attr)

        with torch.no_grad():
            p_sd, g_sd = net.forward_with_gate(z_eval_t, b_eval_t)
            preds.append(p_sd.numpy())
            gates.append(g_sd.numpy())

    return (
        np.mean(preds, axis=0),
        np.mean(gates, axis=0),
        np.mean(grad_imps, axis=0),
        col_names,
        np.mean(raw24_attrs, axis=0),
    )


def _select_torch_epochs_on_val(
    model_builder,
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 2e-3,
    weight_decay: float = 2.0e-2,
    max_epochs: int = 120,
    batch_size: int = 64,
) -> Tuple[int, np.ndarray]:
    """Select early-stopping epoch strictly on pre-2019 validation data."""
    x_mean = X_sub.mean(axis=0, keepdims=True)
    x_std = X_sub.std(axis=0, keepdims=True) + 1e-6
    y_mean = float(y_sub.mean())
    y_std = float(y_sub.std()) + 1e-6

    X_sub_s = ((X_sub - x_mean) / x_std).astype(np.float32)
    X_val_s = ((X_val - x_mean) / x_std).astype(np.float32)
    y_sub_s = ((y_sub - y_mean) / y_std).astype(np.float32)

    set_global_seed(42)
    probe = model_builder()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.75)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_sub_s), torch.from_numpy(y_sub_s)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_s)

    val_losses = []
    val_preds = []
    for _ in range(max_epochs):
        probe.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
        sched.step()

        probe.eval()
        with torch.no_grad():
            vp = probe(X_val_t).numpy() * y_std + y_mean
            val_preds.append(vp)
            val_losses.append(float(np.mean((vp - y_val) ** 2)))

    smooth_val = np.convolve(val_losses, np.ones(3) / 3.0, mode="valid")
    best_epoch = int(np.clip(int(np.argmin(smooth_val)) + 2, 18, max_epochs))
    return best_epoch, val_preds[best_epoch - 1]


def _fit_and_predict_torch(
    model_builder,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_eval: np.ndarray,
    epochs: int,
    lr: float = 2e-3,
    weight_decay: float = 2.0e-2,
    batch_size: int = 64,
    seeds: Tuple[int, ...] = (42, 2026),
    return_attention: bool = False,
) -> Tuple[np.ndarray, List[nn.Module], Dict[str, np.ndarray]]:
    """Fit multi-seed neural ensemble on causal training slice and predict on `X_eval`."""
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std = X_tr.std(axis=0, keepdims=True) + 1e-6
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) + 1e-6

    X_tr_s = ((X_tr - x_mean) / x_std).astype(np.float32)
    # Soft-clip extreme out-of-distribution inputs (>4.5 sigma, e.g., April 2020 lockdown shock)
    X_eval_s = np.clip((X_eval - x_mean) / x_std, -4.5, 4.5).astype(np.float32)
    y_tr_s = ((y_tr - y_mean) / y_std).astype(np.float32)

    X_eval_t = torch.from_numpy(X_eval_s)
    loss_fn = nn.SmoothL1Loss(beta=0.75)

    models: List[nn.Module] = []
    preds = []
    attns = []
    ig_attrs = []

    for sd in seeds:
        set_global_seed(sd)
        m = model_builder()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)),
            batch_size=batch_size,
            shuffle=True,
        )
        m.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(m(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sched.step()

        m.eval()
        with torch.no_grad():
            if return_attention and hasattr(m, "forward_with_attention"):
                tp, aw = m.forward_with_attention(X_eval_t)
                attns.append(aw.numpy())
            else:
                tp = m(X_eval_t)
            preds.append(tp.numpy() * y_std + y_mean)

        # Exact Integrated Gradients [N_eval, 12, 14] (in pp) for Temporal Transformer
        if return_attention and hasattr(m, "forward_with_attention"):
            n_steps = 12
            grad_accum = np.zeros_like(X_eval_s, dtype=np.float32)
            for k_step in range(1, n_steps + 1):
                alpha = float(k_step) / n_steps
                x_interp = (X_eval_t * alpha).clone().detach().requires_grad_(True)
                out_k = m(x_interp).sum()
                out_k.backward()
                grad_accum += x_interp.grad.detach().numpy()
            ig_pp = (X_eval_s * (grad_accum / n_steps)) * y_std
            ig_attrs.append(ig_pp)

        models.append(m)

    info = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.array([y_mean]),
        "y_std": np.array([y_std]),
    }
    if attns:
        info["attention"] = np.mean(attns, axis=0)
    if ig_attrs:
        info["ig_attributions"] = np.mean(ig_attrs, axis=0)
    return np.mean(preds, axis=0), models, info


# ---------------------------------------------------------------------------
# Master Horizon Training & Causal Annual Expanding-Window Evaluation
# ---------------------------------------------------------------------------


@dataclass
class HorizonRunArtifacts:
    horizon: int
    metrics_df: pd.DataFrame
    predictions_df: pd.DataFrame
    regime_metrics_df: pd.DataFrame
    fitted_objects: Dict[str, Any]


MODEL_FAMILIES: Dict[str, str] = {
    "Random Walk": "1. Econometric Benchmark",
    "AR(p) Benchmark": "1. Econometric Benchmark",
    "OLS Phillips Curve": "1. Econometric Benchmark",
    "Ridge (L2)": "2. Regularized Linear",
    "Elastic Net (L1+L2)": "2. Regularized Linear",
    "Random Forest": "3. Tree Ensemble",
    "XGBoost": "3. Tree Ensemble",
    "MLP (Deep Net)": "4. Deep Learning",
    "LSTM (Sequence)": "4. Deep Learning",
    "Temporal Transformer": "4. Deep Learning",
    "XGB + Deep Net": "5. Hybrid Tree + Deep Net",
    "Forecast Combination": "6. Cross-Family Ensemble",
}


def train_and_evaluate_horizon(ds: HorizonDataset) -> HorizonRunArtifacts:
    """
    1. Tune hyperparameters & stopping epochs strictly on pre-2019 validation data (`X_subtrain -> X_val`).
    2. Evaluate out-of-sample (2019–2026) using Central-Bank Causal Annual Expanding Windows:
       for each test year Y in {2019..2026}, models re-estimate ONLY on historical observations
       whose h-month target realization occurred strictly before Jan 1 of year Y.
    """
    h = ds.horizon
    tscv = TimeSeriesSplit(n_splits=5)

    # ------------------------------------------------------------------
    # STAGE A: Pre-2019 Hyperparameter & Architecture Selection (Zero Leakage)
    # ------------------------------------------------------------------
    val_preds_cpi: Dict[str, np.ndarray] = {}

    rf_candidates = [
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 7, "max_features": 0.65},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5, "max_features": 0.65},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 4, "max_features": 0.55},
    ]
    best_rf_cfg = min(
        rf_candidates,
        key=lambda c: mean_squared_error(
            ds.delta_val,
            RandomForestRegressor(**c, random_state=42, n_jobs=1).fit(ds.X_subtrain, ds.delta_subtrain).predict(ds.X_val),
        ),
    )

    xgb_candidates = [
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 180, "min_child_weight": 6, "reg_alpha": 0.5, "reg_lambda": 3.0},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "min_child_weight": 6, "reg_alpha": 0.5, "reg_lambda": 4.0},
        {"max_depth": 3, "learning_rate": 0.025, "n_estimators": 160, "min_child_weight": 8, "reg_alpha": 1.0, "reg_lambda": 5.0},
        {"max_depth": 4, "learning_rate": 0.025, "n_estimators": 180, "min_child_weight": 7, "reg_alpha": 0.8, "reg_lambda": 4.0},
    ]
    best_xgb_cfg = min(
        xgb_candidates,
        key=lambda c: mean_squared_error(
            ds.delta_val,
            XGBRegressor(**c, subsample=0.80, colsample_bytree=0.75, objective="reg:squarederror", random_state=42, n_jobs=1)
            .fit(ds.X_subtrain, ds.delta_subtrain)
            .predict(ds.X_val),
        ),
    )

    n_tab = ds.X_train_full.shape[1]
    n_seq_ch = ds.X_seq_train_full.shape[2]
    seq_len = ds.X_seq_train_full.shape[1]

    mlp_epochs, mlp_val_d = _select_torch_epochs_on_val(
        lambda: MacroMLP(in_features=n_tab, hidden_dim=64, dropout=0.22),
        ds.X_subtrain.values, ds.delta_subtrain, ds.X_val.values, ds.delta_val,
    )
    lstm_epochs, lstm_val_d = _select_torch_epochs_on_val(
        lambda: MacroLSTM(in_channels=n_seq_ch, hidden_dim=32, dropout=0.20),
        ds.X_seq_subtrain, ds.delta_subtrain, ds.X_seq_val, ds.delta_val,
    )
    tf_epochs, tf_val_d = _select_torch_epochs_on_val(
        lambda: InterpretableTemporalTransformer(in_channels=n_seq_ch, seq_len=seq_len, d_model=32, n_heads=4, dropout=0.20),
        ds.X_seq_subtrain, ds.delta_subtrain, ds.X_seq_val, ds.delta_val, lr=1.8e-3,
    )

    val_preds_cpi["MLP (Deep Net)"] = ds.anchor_val + mlp_val_d
    val_preds_cpi["LSTM (Sequence)"] = ds.anchor_val + lstm_val_d
    val_preds_cpi["Temporal Transformer"] = ds.anchor_val + tf_val_d

    # Select optimal epochs for Hybrid `XGB + Deep Net` strictly on pre-2019 validation set
    best_hyb_epochs = 75
    best_hyb_val_mse = float("inf")
    best_hyb_val_d = None
    for ep_cand in (45, 75, 105):
        hd_val, _, _, _, _ = _fit_and_predict_xgb_deep_net(
            ds.X_subtrain, ds.delta_subtrain, ds.X_val, best_xgb_cfg, horizon=h, epochs=ep_cand, seeds=(42,)
        )
        mse_h = float(mean_squared_error(ds.delta_val, hd_val))
        if mse_h < best_hyb_val_mse:
            best_hyb_val_mse = mse_h
            best_hyb_epochs = ep_cand
            best_hyb_val_d = hd_val
    val_preds_cpi["XGB + Deep Net"] = ds.anchor_val + best_hyb_val_d

    best_deep_val_name = min(
        ["MLP (Deep Net)", "LSTM (Sequence)", "Temporal Transformer"],
        key=lambda m_name: mean_squared_error(ds.y_val, val_preds_cpi[m_name]),
    )

    # ------------------------------------------------------------------
    # STAGE B: Causal Annual Expanding-Window Out-of-Sample Forecasting (2019–2026)
    # ------------------------------------------------------------------
    # Combine full pre-test + test arrays so expanding cutoffs can slice cleanly without leakage
    X_all_df = pd.concat([ds.X_train_full, ds.X_test], axis=0)
    # Note: between ds.dates_train_full[-1] and ds.dates_test[0], there are h-1 embargo rows in 2018;
    # here X_all_df contains all valid rows and we filter strictly by `realized_date < cutoff_date`!
    dates_origin_all = pd.DatetimeIndex(list(ds.dates_train_full) + list(ds.dates_test))
    realized_all = dates_origin_all + pd.DateOffset(months=h)
    X_seq_all = np.concatenate([ds.X_seq_train_full, ds.X_seq_test], axis=0)
    delta_all = np.concatenate([ds.delta_train_full, ds.delta_test], axis=0)

    n_test = len(ds.dates_test)
    model_names_core = [
        "Random Walk",
        "AR(p) Benchmark",
        "OLS Phillips Curve",
        "Ridge (L2)",
        "Elastic Net (L1+L2)",
        "Random Forest",
        "XGBoost",
        "MLP (Deep Net)",
        "LSTM (Sequence)",
        "Temporal Transformer",
        "XGB + Deep Net",
    ]
    test_preds_cpi: Dict[str, np.ndarray] = {m: np.zeros(n_test, dtype=np.float64) for m in model_names_core}
    test_preds_cpi["Random Walk"] = ds.anchor_test.copy()

    xgb_shap_contribs = np.zeros((n_test, n_tab + 1), dtype=np.float64)
    tf_test_attentions = np.zeros((n_test, seq_len), dtype=np.float64)
    tf_test_ig_attributions = np.zeros((n_test, seq_len, n_seq_ch), dtype=np.float64)
    hyb_test_gates = np.zeros(n_test, dtype=np.float64)
    hyb_test_feat_attr = np.zeros((n_test, n_tab), dtype=np.float64)

    fitted_latest: Dict[str, Any] = {
        "XGBoost_config": best_xgb_cfg,
        "RandomForest_config": best_rf_cfg,
        "best_deep_member": best_deep_val_name,
    }

    test_years = sorted(ds.dates_test.year.unique())
    alphas_ridge = np.logspace(-0.5, 3.5, 35)
    alphas_enet = np.logspace(-2.2, 0.8, 30)

    for yr in test_years:
        yr_start = pd.Timestamp(f"{yr}-01-01")
        # Test positions belonging to forecast origin year `yr`
        test_mask_yr = np.asarray(ds.dates_test.year == yr, dtype=bool)
        if not np.any(test_mask_yr):
            continue

        # Causal Expanding Training Set: ONLY historical origins whose h-step target realized BEFORE Jan 1 of `yr`!
        tr_mask = np.asarray(realized_all < yr_start, dtype=bool)
        X_tr_df = X_all_df.iloc[tr_mask]
        X_seq_tr = X_seq_all[tr_mask]
        d_tr = delta_all[tr_mask]

        X_te_yr_df = ds.X_test.iloc[test_mask_yr]
        X_seq_te_yr = ds.X_seq_test[test_mask_yr]
        anchor_yr = ds.anchor_test[test_mask_yr]

        # Soft winsorize extreme 4.5-sigma macro shocks at inference for unregularized linear stability
        tr_med = X_tr_df.mean()
        tr_sd = X_tr_df.std() + 1e-6
        X_te_yr_clipped = X_te_yr_df.clip(lower=tr_med - 4.5 * tr_sd, upper=tr_med + 4.5 * tr_sd, axis=1)

        # 2. AR(p)
        ar_m = LinearRegression().fit(X_tr_df[AR_BENCHMARK_FEATURES], d_tr)
        test_preds_cpi["AR(p) Benchmark"][test_mask_yr] = anchor_yr + ar_m.predict(X_te_yr_clipped[AR_BENCHMARK_FEATURES])

        # 3. OLS Phillips Curve
        pc_m = LinearRegression().fit(X_tr_df[PHILLIPS_CURVE_FEATURES], d_tr)
        test_preds_cpi["OLS Phillips Curve"][test_mask_yr] = anchor_yr + pc_m.predict(X_te_yr_clipped[PHILLIPS_CURVE_FEATURES])

        # Standardize for Ridge & Elastic Net
        sc = StandardScaler().fit(X_tr_df)
        X_tr_sc = sc.transform(X_tr_df)
        X_te_sc = np.clip(sc.transform(X_te_yr_df), -4.5, 4.5)

        # 4. Ridge (L2)
        ridge_m = RidgeCV(alphas=alphas_ridge, cv=tscv).fit(X_tr_sc, d_tr)
        test_preds_cpi["Ridge (L2)"][test_mask_yr] = anchor_yr + ridge_m.predict(X_te_sc)

        # 5. Elastic Net (L1 + L2)
        enet_m = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95],
            alphas=alphas_enet,
            cv=tscv,
            max_iter=6000,
            random_state=42,
        ).fit(X_tr_sc, d_tr)
        test_preds_cpi["Elastic Net (L1+L2)"][test_mask_yr] = anchor_yr + enet_m.predict(X_te_sc)

        # 6. Random Forest
        rf_m = RandomForestRegressor(**best_rf_cfg, random_state=42, n_jobs=1).fit(X_tr_df, d_tr)
        test_preds_cpi["Random Forest"][test_mask_yr] = anchor_yr + rf_m.predict(X_te_yr_df)

        # 7. XGBoost + Exact Native C++ TreeSHAP (`pred_contribs=True`)
        xgb_m = XGBRegressor(
            **best_xgb_cfg,
            subsample=0.80,
            colsample_bytree=0.75,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ).fit(X_tr_df, d_tr)
        xgb_d_pred = xgb_m.predict(X_te_yr_df)
        test_preds_cpi["XGBoost"][test_mask_yr] = anchor_yr + xgb_d_pred

        dmat_yr = xgb.DMatrix(X_te_yr_df, feature_names=list(X_te_yr_df.columns))
        xgb_shap_contribs[test_mask_yr] = xgb_m.get_booster().predict(dmat_yr, pred_contribs=True)

        # 8. MLP (Deep Net)
        mlp_d, mlp_ms, mlp_info = _fit_and_predict_torch(
            lambda: MacroMLP(in_features=n_tab, hidden_dim=64, dropout=0.22),
            X_tr_df.values, d_tr, X_te_yr_df.values, epochs=mlp_epochs,
        )
        test_preds_cpi["MLP (Deep Net)"][test_mask_yr] = anchor_yr + mlp_d

        # 9. LSTM (Sequence)
        lstm_d, lstm_ms, lstm_info = _fit_and_predict_torch(
            lambda: MacroLSTM(in_channels=n_seq_ch, hidden_dim=32, dropout=0.20),
            X_seq_tr, d_tr, X_seq_te_yr, epochs=lstm_epochs,
        )
        test_preds_cpi["LSTM (Sequence)"][test_mask_yr] = anchor_yr + lstm_d

        # 10. Temporal Transformer (Sequence + Lag Attention + 2D Integrated Gradients)
        tf_d, tf_ms, tf_info = _fit_and_predict_torch(
            lambda: InterpretableTemporalTransformer(in_channels=n_seq_ch, seq_len=seq_len, d_model=32, n_heads=4, dropout=0.20),
            X_seq_tr, d_tr, X_seq_te_yr, epochs=tf_epochs, lr=1.8e-3, return_attention=True,
        )
        test_preds_cpi["Temporal Transformer"][test_mask_yr] = anchor_yr + tf_d
        tf_test_attentions[test_mask_yr] = tf_info["attention"]
        if "ig_attributions" in tf_info:
            tf_test_ig_attributions[test_mask_yr] = tf_info["ig_attributions"]

        # 11. Hybrid `XGB + Deep Net` (Gated Deep Trend Network on XGBoost SHAP & Trend Features)
        hyb_d, hyb_g, hyb_imp, hyb_cols, hyb_raw24 = _fit_and_predict_xgb_deep_net(
            X_tr_df, d_tr, X_te_yr_df, best_xgb_cfg, horizon=h, epochs=best_hyb_epochs
        )
        test_preds_cpi["XGB + Deep Net"][test_mask_yr] = anchor_yr + hyb_d
        hyb_test_gates[test_mask_yr] = hyb_g
        hyb_test_feat_attr[test_mask_yr] = hyb_raw24

        # Store latest full-sample fitted objects for global parameter inspection
        fitted_latest["tabular_scaler"] = sc
        fitted_latest["Elastic Net (L1+L2)"] = enet_m
        fitted_latest["XGBoost"] = xgb_m
        fitted_latest["MLP (Deep Net)"] = (mlp_ms, mlp_info)
        fitted_latest["Temporal Transformer"] = (tf_ms, tf_info)
        fitted_latest["hybrid_feature_importance"] = hyb_imp
        fitted_latest["hybrid_feature_names"] = hyb_cols

    fitted_latest["xgb_test_shap_contribs"] = xgb_shap_contribs
    fitted_latest["tf_test_attention_weights"] = tf_test_attentions
    fitted_latest["tf_test_ig_attributions"] = tf_test_ig_attributions
    fitted_latest["hybrid_gate_weights"] = hyb_test_gates
    fitted_latest["hyb_test_feat_attr"] = hyb_test_feat_attr

    # 11. Cross-Family Forecast Combination (Linear + Boosted Tree + Best Pre-2019 Val Deep Model)
    test_preds_cpi["Forecast Combination"] = (
        test_preds_cpi["Elastic Net (L1+L2)"]
        + test_preds_cpi["XGBoost"]
        + test_preds_cpi[best_deep_val_name]
    ) / 3.0

    # Compute overall out-of-sample metrics
    y_rw = test_preds_cpi["Random Walk"]
    metric_rows = []
    for model_name, y_hat in test_preds_cpi.items():
        m = compute_forecast_metrics(
            y_true=ds.y_test,
            y_pred=y_hat,
            anchor=ds.anchor_test,
            y_bench_rw=y_rw,
            horizon=h,
        )
        metric_rows.append(
            {
                "Horizon_Months": h,
                "Model": model_name,
                "Family": MODEL_FAMILIES[model_name],
                **m,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    pred_df = pd.DataFrame(
        {
            "forecast_origin_date": ds.dates_test,
            "target_realization_date": ds.target_dates_test,
            "regime": ds.regimes_test.values,
            "current_cpi_yoy": ds.anchor_test,
            "actual_cpi_yoy": ds.y_test,
            **test_preds_cpi,
        }
    )

    regime_rows = []
    for reg_name, grp in pred_df.groupby("regime", sort=False):
        rw_reg_rmse = float(np.sqrt(mean_squared_error(grp["actual_cpi_yoy"], grp["Random Walk"])))
        for model_name in test_preds_cpi.keys():
            reg_rmse = float(np.sqrt(mean_squared_error(grp["actual_cpi_yoy"], grp[model_name])))
            reg_mae = float(mean_absolute_error(grp["actual_cpi_yoy"], grp[model_name]))
            regime_rows.append(
                {
                    "Horizon_Months": h,
                    "Regime": reg_name,
                    "N_Months": len(grp),
                    "Model": model_name,
                    "Family": MODEL_FAMILIES[model_name],
                    "RMSE": reg_rmse,
                    "MAE": reg_mae,
                    "Rel_RMSE_vs_RW": reg_rmse / rw_reg_rmse if rw_reg_rmse > 0 else 1.0,
                }
            )
    regime_metrics_df = pd.DataFrame(regime_rows)

    return HorizonRunArtifacts(
        horizon=h,
        metrics_df=metrics_df,
        predictions_df=pred_df,
        regime_metrics_df=regime_metrics_df,
        fitted_objects=fitted_latest,
    )


def run_cv_leakage_diagnostic(ds: HorizonDataset) -> pd.DataFrame:
    """
    Demonstrate why random K-Fold cross-validation creates severe look-ahead optimism
    on macroeconomic time series compared to chronological TimeSeriesSplit.
    """
    X = ds.X_train_full.values
    y = ds.delta_train_full

    models_to_check = {
        "Elastic Net (L1+L2)": ElasticNetCV(l1_ratio=0.5, cv=3, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=1),
    }

    kf_leaky = KFold(n_splits=5, shuffle=True, random_state=42)
    tscv_clean = TimeSeriesSplit(n_splits=5)

    rows = []
    for name, est in models_to_check.items():
        leaky_rmses = []
        for tr, va in kf_leaky.split(X):
            sc = StandardScaler().fit(X[tr])
            m = clone(est).fit(sc.transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[va]))
            leaky_rmses.append(np.sqrt(mean_squared_error(y[va], pred)))

        clean_rmses = []
        for tr, va in tscv_clean.split(X):
            tr_purged = tr[:-ds.horizon] if len(tr) > ds.horizon else tr
            sc = StandardScaler().fit(X[tr_purged])
            m = clone(est).fit(sc.transform(X[tr_purged]), y[tr_purged])
            pred = m.predict(sc.transform(X[va]))
            clean_rmses.append(np.sqrt(mean_squared_error(y[va], pred)))

        r_leaky = float(np.mean(leaky_rmses))
        r_clean = float(np.mean(clean_rmses))
        rows.append(
            {
                "Model": name,
                "Random_KFold_RMSE_Leaky": round(r_leaky, 3),
                "Chronological_Purged_TSCV_RMSE": round(r_clean, 3),
                "Artificial_Optimism_Bias_pct": round((1.0 - r_leaky / r_clean) * 100.0, 1),
            }
        )

    return pd.DataFrame(rows)
