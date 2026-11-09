"""
Model Explainability (XAI) across Linear, Tree Ensemble, and Deep Sequence Models.

Extracts:
1. Standardized Elastic Net coefficients (direction & sparsity)
2. Exact XGBoost Native TreeSHAP values (global importance, nonlinear dependence,
   and local historical episode attributions satisfying exact additivity)
3. Deep Neural Network (MLP) input-gradient attribution
4. Temporal Transformer lag-attention weights across macroeconomic regimes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src import TABLES_DIR
from src.features import FEATURE_LABELS, SEQUENCE_FEATURES, HorizonDataset
from src.models import HorizonRunArtifacts


@dataclass
class ExplainabilityBundle:
    importance_df: pd.DataFrame
    shap_values: np.ndarray                 # [N_test, D] exact TreeSHAP in percentage points
    shap_base_values: np.ndarray            # [N_test] exact causal baseline E[delta_hat]
    X_test_labeled: pd.DataFrame            # X_test with human-readable column names
    attention_by_regime_df: pd.DataFrame    # [Regime x Lag (t-11 .. t)]
    attention_matrix: np.ndarray            # [N_test, 12]
    episode_surge: Dict[str, object]
    episode_disinflation: Dict[str, object]


def compute_mlp_feature_importance(
    mlp_tuple,
    X_test: np.ndarray,
) -> np.ndarray:
    """Compute input-gradient attribution |d f / d x_j * z_j| averaged across seeds & test months."""
    models, stats_dict = mlp_tuple
    x_mean = stats_dict["x_mean"]
    x_std = stats_dict["x_std"]
    y_std = float(stats_dict["y_std"][0])

    X_s = np.clip((X_test - x_mean) / x_std, -4.5, 4.5).astype(np.float32)
    attributions = []
    for m in models:
        m.eval()
        xt = torch.tensor(X_s, dtype=torch.float32, requires_grad=True)
        out = m(xt).sum()
        out.backward()
        grad_x = xt.grad.detach().numpy() * y_std
        attributions.append(np.mean(np.abs(grad_x * X_s), axis=0))
    return np.mean(attributions, axis=0)


def extract_episode_breakdown(
    idx_pos: int,
    ds: HorizonDataset,
    shap_values: np.ndarray,
    base_vec: np.ndarray,
    xgb_preds_cpi: np.ndarray,
    title: str,
) -> Dict[str, object]:
    """Package a single historical month's anchored forecast + top exact TreeSHAP drivers."""
    origin_dt = ds.dates_test[idx_pos]
    target_dt = ds.target_dates_test[idx_pos]
    anchor_cpi = float(ds.anchor_test[idx_pos])
    actual_cpi = float(ds.y_test[idx_pos])
    pred_cpi = float(xgb_preds_cpi[idx_pos])
    base_val = float(base_vec[idx_pos])

    phi = shap_values[idx_pos]
    feat_vals = ds.X_test.iloc[idx_pos]
    contrib_df = pd.DataFrame(
        {
            "feature_code": ds.feature_names,
            "feature_label": [FEATURE_LABELS[f] for f in ds.feature_names],
            "feature_value": feat_vals.values,
            "shap_pp": phi,
            "abs_shap": np.abs(phi),
        }
    ).sort_values("abs_shap", ascending=False)

    return {
        "title": title,
        "origin_date": origin_dt.strftime("%Y-%m"),
        "target_date": target_dt.strftime("%Y-%m"),
        "anchor_cpi": anchor_cpi,
        "base_delta": base_val,
        "pred_cpi": pred_cpi,
        "actual_cpi": actual_cpi,
        "contributions": contrib_df,
    }


def run_explainability_analysis(
    ds: HorizonDataset,
    run_art: HorizonRunArtifacts,
) -> ExplainabilityBundle:
    """Run full multi-family explainability suite on the primary horizon run."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fitted = run_art.fitted_objects

    # 1. Elastic Net standardized coefficients
    enet = fitted["Elastic Net (L1+L2)"]
    enet_coefs = enet.coef_.copy()

    # 2. Exact Native C++ TreeSHAP values (`pred_contribs=True`) over out-of-sample test set
    contribs = fitted["xgb_test_shap_contribs"]
    shap_vals = contribs[:, :-1]
    base_vec = contribs[:, -1]
    xgb_mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # 3. Hybrid `XGB + MLP` (`XGB + Deep Net`) chain-rule feature attribution (in pp)
    if "hyb_test_feat_attr" in fitted:
        hyb_imp_24 = np.mean(fitted["hyb_test_feat_attr"], axis=0)
    else:
        hyb_imp_24 = compute_mlp_feature_importance(fitted["MLP (Deep Net)"], ds.X_test.values)

    imp_df = pd.DataFrame(
        {
            "Feature_Code": ds.feature_names,
            "Feature_Label": [FEATURE_LABELS[f] for f in ds.feature_names],
            "ElasticNet_Signed_Coef": enet_coefs,
            "ElasticNet_Abs_Coef": np.abs(enet_coefs),
            "XGBoost_Mean_Abs_SHAP_pp": xgb_mean_abs_shap,
            "XGB_MLP_Attribution_pp": hyb_imp_24,
        }
    )

    for col, share_col in [
        ("ElasticNet_Abs_Coef", "ElasticNet_Share_pct"),
        ("XGBoost_Mean_Abs_SHAP_pp", "XGBoost_Share_pct"),
        ("XGB_MLP_Attribution_pp", "XGB_MLP_Share_pct"),
    ]:
        total = imp_df[col].sum()
        imp_df[share_col] = (imp_df[col] / total * 100.0) if total > 0 else 0.0

    imp_df["Consensus_Rank_Score"] = (
        imp_df["ElasticNet_Share_pct"] + imp_df["XGBoost_Share_pct"] + imp_df["XGB_MLP_Share_pct"]
    ) / 3.0
    imp_df = imp_df.sort_values("Consensus_Rank_Score", ascending=False).reset_index(drop=True)
    imp_df.to_csv(TABLES_DIR / "explainability_feature_importance.csv", index=False)

    # 4. Temporal Transformer lag-attention weights [N_test, 12] across regimes
    attn_mat = fitted["tf_test_attention_weights"]
    lag_cols = [f"t-{11 - i}" if i < 11 else "t (Current)" for i in range(12)]
    attn_df = pd.DataFrame(attn_mat, index=ds.dates_test, columns=lag_cols)
    attn_df["Regime"] = ds.regimes_test.values
    attn_by_regime = attn_df.groupby("Regime", sort=False)[lag_cols].mean()
    attn_by_regime.to_csv(TABLES_DIR / "transformer_attention_by_regime.csv")

    # 5. Select two iconic macroeconomic episodes for Local SHAP Waterfall Explanations
    dates_s = pd.Series(np.arange(len(ds.dates_test)), index=ds.dates_test)
    pred_deltas = run_art.predictions_df["XGBoost"].values - ds.anchor_test

    surge_window = dates_s.loc["2021-01-01":"2022-05-01"]
    surge_idx = int(surge_window.iloc[int(np.argmax(pred_deltas[surge_window.values]))])

    disinf_window = dates_s.loc["2022-06-01":"2024-06-01"]
    disinf_idx = int(disinf_window.iloc[int(np.argmin(pred_deltas[disinf_window.values]))])

    xgb_preds_cpi = run_art.predictions_df["XGBoost"].values
    ep_surge = extract_episode_breakdown(
        surge_idx, ds, shap_vals, base_vec, xgb_preds_cpi, "Inflation Surge Episode"
    )
    ep_disinf = extract_episode_breakdown(
        disinf_idx, ds, shap_vals, base_vec, xgb_preds_cpi, "Disinflation Turning-Point Episode"
    )

    X_test_labeled = ds.X_test.rename(columns=FEATURE_LABELS)

    return ExplainabilityBundle(
        importance_df=imp_df,
        shap_values=shap_vals,
        shap_base_values=base_vec,
        X_test_labeled=X_test_labeled,
        attention_by_regime_df=attn_by_regime,
        attention_matrix=attn_mat,
        episode_surge=ep_surge,
        episode_disinflation=ep_disinf,
    )


MACRO_CHANNEL_BUCKETS: Dict[str, str] = {
    "base_3m_ann": "1. Base Effect & Price Inertia",
    "cpi_yoy": "1. Base Effect & Price Inertia",
    "cpi_mom_ann": "1. Base Effect & Price Inertia",
    "cpi_3m_ann": "1. Base Effect & Price Inertia",
    "cpi_yoy_lag3": "1. Base Effect & Price Inertia",
    "cpi_yoy_lag6": "1. Base Effect & Price Inertia",
    "core_headline_gap": "1. Base Effect & Price Inertia",
    "month_sin": "1. Base Effect & Price Inertia",
    "month_cos": "1. Base Effect & Price Inertia",
    "real_fedfunds": "2. Monetary, M2 & Equities",
    "fedfunds_chg_12m": "2. Monetary, M2 & Equities",
    "yield_spread_10y2y": "2. Monetary, M2 & Equities",
    "m2_yoy": "2. Monetary, M2 & Equities",
    "stock_yoy": "2. Monetary, M2 & Equities",
    "ppi_yoy": "3. Supply, Electricity & Oil",
    "oil_yoy": "3. Supply, Electricity & Oil",
    "elec_yoy": "3. Supply, Electricity & Oil",
    "indpro_yoy": "4. Real GDP, Output & Slack",
    "gdp_yoy": "4. Real GDP, Output & Slack",
    "unrate": "4. Real GDP, Output & Slack",
    "unrate_chg_12m": "4. Real GDP, Output & Slack",
    "umcsent_chg_12m": "4. Real GDP, Output & Slack",
    "mich_exp_1y": "5. Inflation Expectations (Mich.)",
    "mich_exp_chg3m": "5. Inflation Expectations (Mich.)",
}


def compute_policy_event_windows(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate 3-month-ahead forecast RMSE across 5 key macroeconomic shock & policy
    turning-point windows (2020–2026) to contrast ML anticipation against pure lag.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    p = pred_df.copy()
    p["target_realization_date"] = pd.to_datetime(p["target_realization_date"])

    windows = [
        ("1. COVID\nShock\n(2020 H1)", "1. COVID Lockdown (2020 H1)", "2020-01-01", "2020-06-30", "Exogenous Shock"),
        ("2. Reopening\nSurge\n(2021)", "2. Reopening Surge (2021)", "2021-01-01", "2021-12-31", "Base Roll-Off + Stimulus"),
        ("3. Ukraine\nWar Oil\n(2022 H1)", "3. Ukraine Energy Shock (2022 H1)", "2022-01-01", "2022-06-30", "Geopolitical War Surprise"),
        ("4. Fed 75bp\nHikes\n(22H2–23)", "4. Fed 75bp Hikes & Disinflation (22H2–23)", "2022-07-01", "2023-12-31", "Monetary Tightening Turn"),
        ("5. Soft Landing\n& Pivot\n(2024–26)", "5. Soft Landing & Rate Pivot (2024–26)", "2024-01-01", "2026-12-31", "Policy Normalization"),
    ]

    rows = []
    for plot_lbl, full_lbl, s_dt, e_dt, note in windows:
        sub = p[(p["target_realization_date"] >= s_dt) & (p["target_realization_date"] <= e_dt)]
        if len(sub) == 0:
            continue
        rw_rmse = float(np.sqrt(np.mean((sub["actual_cpi_yoy"] - sub["Random Walk"]) ** 2)))
        enet_rmse = float(np.sqrt(np.mean((sub["actual_cpi_yoy"] - sub["Elastic Net (L1+L2)"]) ** 2)))
        xgb_rmse = float(np.sqrt(np.mean((sub["actual_cpi_yoy"] - sub["XGBoost"]) ** 2)))
        hyb_rmse = float(np.sqrt(np.mean((sub["actual_cpi_yoy"] - sub["XGB + Deep Net"]) ** 2)))
        best_ml = min(enet_rmse, xgb_rmse, hyb_rmse)

        rows.append(
            {
                "Event_Window": full_lbl,
                "Plot_Label": plot_lbl,
                "Macro_Context": note,
                "Months": len(sub),
                "RandomWalk_PureLag_RMSE": round(rw_rmse, 3),
                "ElasticNet_RMSE": round(enet_rmse, 3),
                "XGBoost_RMSE": round(xgb_rmse, 3),
                "XGB_DeepNet_RMSE": round(hyb_rmse, 3),
                "Best_ML_vs_PureLag_pct": round((best_ml / rw_rmse - 1.0) * 100.0, 1),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLES_DIR / "policy_shock_windows_h3.csv", index=False)
    return out_df


def compute_horizon_channel_shap(
    horizon_runs: Dict[int, HorizonRunArtifacts],
    feature_names: list,
) -> pd.DataFrame:
    """
    Aggregate exact out-of-sample XGBoost TreeSHAP shares into 5 macroeconomic channels
    across horizons h = 1, 3, 6, 12 months ('The Macroeconomic Transmission Clock').
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, run_h in sorted(horizon_runs.items()):
        contribs = run_h.fitted_objects["xgb_test_shap_contribs"][:, :-1]
        mean_abs = np.mean(np.abs(contribs), axis=0)
        s = pd.Series(mean_abs, index=feature_names)
        grp = s.groupby(MACRO_CHANNEL_BUCKETS).sum()
        shares = grp / (grp.sum() + 1e-12) * 100.0
        row = {"Horizon_Months": h}
        for k, v in shares.items():
            row[k] = round(float(v), 2)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLES_DIR / "horizon_channel_shap_shares.csv", index=False)
    return out_df


def build_actionable_policy_playbook(imp_df: pd.DataFrame) -> pd.DataFrame:
    """Build a practitioner-facing table translating top XAI features into policy/decision rules."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    guidance = {
        "base_3m_ann": ("Mechanical Downward Roll-Off (-)", "Compute t-12..t-9 spike first before judging new CPI prints; separates calendar math from fresh inflation."),
        "ppi_yoy": ("Upstream Cost Push (+)", "Leading 1–3m retail pipeline warning: PPI > +5% YoY signals wholesale margin pass-through into goods CPI."),
        "m2_yoy": ("Liquidity Fuel (+)", "Medium-term monetary overhang gauge: double-digit M2 growth (>10%) sustains broad demand inflation across 3–12m."),
        "core_headline_gap": ("Mean-Reversion Anchor (+ toward Core)", "When Headline > Core (gap < -1.5 pp from oil/food spikes), fade transient headline extremes toward Core trend."),
        "real_fedfunds": ("Nonlinear Policy Brake (- above threshold)", "True monetary stance (Fed Rate - CPI): deeply negative (< -2%) fuels inflation; >= 0% to +2% triggers disinflation."),
        "mich_exp_1y": ("12m Wage-Price Anchor (+)", "Surges nearly 6x to 17.8% SHAP share at 12m: rising above 4.0% signals un-anchoring risk requiring hawkish policy."),
        "oil_yoy": ("Immediate Energy Shock (+)", "Strongest at 1m nowcast horizon; watch second-round spillover into Core & Michigan expectations."),
        "indpro_yoy": ("Real Output / Cyclical Heat (+)", "Contracting factory output (< 0% YoY) reinforces disinflationary pull alongside restrictive real rates."),
        "cpi_mom_ann": ("High-Frequency Speedometer (+)", "Annualized 1m pace catches sudden inflection points 2–3 months before 12-month YoY headline turns."),
        "umcsent_chg_12m": ("Household Demand Cooling (+)", "Sharp drops (< -10 pts YoY) signal consumer spending pullback and easing service-sector pricing power."),
        "elec_yoy": ("Sticky Utility Cost Push (+)", "Regulated grid electricity tariffs stay sticky long after crude oil falls, sustaining 3–6m service & retail CPI."),
        "stock_yoy": ("Financial Conditions & Wealth (+)", "Equity market rallies/drawdowns front-run household wealth effects and corporate pricing confidence."),
        "gdp_yoy": ("Broad Service + Goods Demand (+)", "Captures the 80% service-sector economy missing from factory-only INDPRO (especially helpful for linear Elastic Net)."),
    }

    rows = []
    for _, r in imp_df.iterrows():
        code = r["Feature_Code"]
        if code not in guidance:
            continue
        direction_tag, how_to_use = guidance[code]
        rows.append(
            {
                "Feature": r["Feature_Label"],
                "Consensus_Share_pct": round(float(r["Consensus_Rank_Score"]), 1),
                "XGB_SHAP_Share_pct": round(float(r["XGBoost_Share_pct"]), 1),
                "ElasticNet_Share_pct": round(float(r["ElasticNet_Share_pct"]), 1),
                "XGB_MLP_Share_pct": round(float(r["XGB_MLP_Share_pct"]), 1),
                "Economic_Mechanism": direction_tag,
                "How_Policymakers_and_Analysts_Use_It": how_to_use,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLES_DIR / "xai_policy_playbook.csv", index=False)
    return out_df


def run_transformer_explainability(
    ds_h3: HorizonDataset,
    run_h3: HorizonRunArtifacts,
    ds_h12: HorizonDataset,
    run_h12: HorizonRunArtifacts,
) -> Dict[str, pd.DataFrame]:
    """
    Full 2D Sequence Explainability for the `InterpretableTemporalTransformer`:
      1. Channel Attribution Share (%) at h=12m (#1 Overall Model) vs. h=3m via Integrated Gradients
      2. 2D Spatio-Temporal Attribution Matrix [14 Macro Channels x 12 Lookback Lags] at h=12m
      3. Explicit Multi-Head Temporal Lag Attention across Macroeconomic Regimes at h=12m
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    seq_labels = [FEATURE_LABELS[c] for c in SEQUENCE_FEATURES]
    lag_cols = [f"t-{11 - i}" if i < 11 else "t (Current)" for i in range(12)]

    ig_h3 = run_h3.fitted_objects["tf_test_ig_attributions"]      # [N_test_3, 12, 14] in pp
    ig_h12 = run_h12.fitted_objects["tf_test_ig_attributions"]    # [N_test_12, 12, 14] in pp

    # Channel totals (summed across 12 lookback lags, averaged across out-of-sample test months)
    ch_pp_h3 = np.mean(np.sum(np.abs(ig_h3), axis=1), axis=0)     # [14]
    ch_pp_h12 = np.mean(np.sum(np.abs(ig_h12), axis=1), axis=0)   # [14]

    sh_h3 = ch_pp_h3 / (ch_pp_h3.sum() + 1e-12) * 100.0
    sh_h12 = ch_pp_h12 / (ch_pp_h12.sum() + 1e-12) * 100.0

    # Signed directional correlation between contemporaneous channel value X_seq[:, -1, j] and net channel IG_j
    net_ig_h12 = np.sum(ig_h12, axis=1)  # [N_test_12, 14] signed pp
    x_curr_h12 = ds_h12.X_seq_test[:, -1, :]
    directions = []
    corr_vals = []
    for j in range(len(SEQUENCE_FEATURES)):
        if np.std(x_curr_h12[:, j]) > 1e-8 and np.std(net_ig_h12[:, j]) > 1e-8:
            r_val = float(np.corrcoef(x_curr_h12[:, j], net_ig_h12[:, j])[0, 1])
        else:
            r_val = 0.0
        corr_vals.append(round(r_val, 3))
        directions.append("Inflationary (+)" if r_val >= 0 else "Disinflationary (−)")

    channel_df = pd.DataFrame(
        {
            "Feature_Code": SEQUENCE_FEATURES,
            "Feature_Label": seq_labels,
            "TF_h12_Mean_Abs_IG_pp": np.round(ch_pp_h12, 4),
            "TF_h12_Share_pct": np.round(sh_h12, 2),
            "TF_h3_Share_pct": np.round(sh_h3, 2),
            "Signed_Corr_h12": corr_vals,
            "Dominant_Direction_h12": directions,
        }
    ).sort_values("TF_h12_Share_pct", ascending=False).reset_index(drop=True)
    channel_df.to_csv(TABLES_DIR / "transformer_channel_importance.csv", index=False)

    # 2D Spatio-Temporal Attribution Matrix [14 Channels x 12 Lags] (% of total 12m attribution)
    grid_pp = np.mean(np.abs(ig_h12), axis=0).T  # [14, 12]
    grid_pct = grid_pp / (grid_pp.sum() + 1e-12) * 100.0
    spatiotemporal_df = pd.DataFrame(grid_pct, index=seq_labels, columns=lag_cols)
    spatiotemporal_df.to_csv(TABLES_DIR / "transformer_spatiotemporal_h12.csv")

    # Regime-conditioned explicit lag attention at h=12m
    attn_h12 = run_h12.fitted_objects["tf_test_attention_weights"]  # [N_test_12, 12]
    attn_df_12 = pd.DataFrame(attn_h12, index=ds_h12.dates_test, columns=lag_cols)
    attn_df_12["Regime"] = ds_h12.regimes_test.values
    attn_reg_12 = attn_df_12.groupby("Regime", sort=False)[lag_cols].mean()
    attn_reg_12.to_csv(TABLES_DIR / "transformer_attention_by_regime_h12.csv")

    return {
        "channel_df": channel_df,
        "spatiotemporal_h12_df": spatiotemporal_df,
        "attention_regime_h12_df": attn_reg_12,
    }


def run_cross_paper_ablation(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Controlled causal ablation quantifying the value of unifying all 4 paper clusters
    from the ECO331 Drive folder:
      1. Core Macro Baseline (12 FRED series)
      2. + Electricity Price Index (Kuskaya & Bilgili, 2025)
      3. + US Stock Market Returns (Subasi 2021; Vamossy 2024)
      4. + Real GDP Growth (Yoon 2021; Maccarrone 2021, 1Q causal publication lag)
      5. Unified 4-Cluster Synthesis (All 15 FRED series)
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from src.features import CORE_12_TABULAR_FEATURES, prepare_horizon_dataset

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ds3 = prepare_horizon_dataset(feat_df, horizon=3)
    dates_all = pd.DatetimeIndex(list(ds3.dates_train_full) + list(ds3.dates_test))
    realized_all = dates_all + pd.DateOffset(months=3)
    delta_all = np.concatenate([ds3.delta_train_full, ds3.delta_test], axis=0)

    specs = [
        ("1. Core Macro\n(12 Series)", "1. Core Macro Baseline (12 FRED Series)", "CPI & NKPC Core (Singh & Bhoi 2022)", []),
        ("2. + Electricity\n(Kuskaya '25)", "2. + Electricity Price Index (CUSR0000SEHF01)", "Sticky Grid Utility Cost (Kuskaya & Bilgili 2025)", ["elec_yoy"]),
        ("3. + Equities\n(Vamossy '24)", "3. + US Stock Market Index (SPASTT01USM661N)", "Financial Wealth Effect (Subasi 2021; Vamossy 2024)", ["stock_yoy"]),
        ("4. + Real GDP\n(Yoon '21)", "4. + Real GDP Growth (GDPC1, Lagged 1Q)", "Broad Service+Goods Demand (Yoon 2021)", ["gdp_yoy"]),
        ("5. Unified\n(All 15 Series)", "5. Unified 4-Cluster Synthesis (All 15 Series)", "Full Course Literature Integration", ["elec_yoy", "stock_yoy", "gdp_yoy"]),
    ]

    rows = []
    base_enet = None
    for plot_lbl, full_lbl, lit_source, extra in specs:
        cols = CORE_12_TABULAR_FEATURES + extra
        X_ext_all = feat_df.loc[dates_all, cols]
        X_ext_te = feat_df.loc[ds3.dates_test, cols]
        enet_p = np.zeros(len(ds3.dates_test))
        xgb_p = np.zeros(len(ds3.dates_test))

        for yr in sorted(ds3.dates_test.year.unique()):
            yr_s = pd.Timestamp(f"{yr}-01-01")
            tm = np.asarray(ds3.dates_test.year == yr, dtype=bool)
            trm = np.asarray(realized_all < yr_s, dtype=bool)
            Xtr, dtr = X_ext_all.iloc[trm], delta_all[trm]
            Xte = X_ext_te.iloc[tm]
            anc = ds3.anchor_test[tm]

            sc = StandardScaler().fit(Xtr)
            em = ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.8, 0.95],
                alphas=np.logspace(-2.2, 0.8, 30),
                cv=TimeSeriesSplit(n_splits=5),
                max_iter=6000,
                random_state=42,
            ).fit(sc.transform(Xtr), dtr)
            enet_p[tm] = anc + em.predict(np.clip(sc.transform(Xte), -4.5, 4.5))

            xm = XGBRegressor(
                max_depth=3,
                learning_rate=0.03,
                n_estimators=200,
                min_child_weight=6,
                reg_alpha=0.5,
                reg_lambda=4.0,
                subsample=0.80,
                colsample_bytree=0.75,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ).fit(Xtr, dtr)
            xgb_p[tm] = anc + xm.predict(Xte)

        rmse_e = float(np.sqrt(np.mean((ds3.y_test - enet_p) ** 2)))
        rmse_x = float(np.sqrt(np.mean((ds3.y_test - xgb_p) ** 2)))
        rmse_avg = 0.5 * (rmse_e + rmse_x)
        if base_enet is None:
            base_enet = rmse_e

        rows.append(
            {
                "Plot_Label": plot_lbl,
                "Feature_Set": full_lbl,
                "Course_Literature_Link": lit_source,
                "Num_FRED_Series": 12 + len(extra),
                "ElasticNet_h3_RMSE": round(rmse_e, 3),
                "XGBoost_h3_RMSE": round(rmse_x, 3),
                "Mean_ML_h3_RMSE": round(rmse_avg, 3),
                "ElasticNet_Gain_vs_Core_pct": round((rmse_e / base_enet - 1.0) * 100.0, 1),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLES_DIR / "cross_paper_channel_ablation.csv", index=False)
    return out_df
