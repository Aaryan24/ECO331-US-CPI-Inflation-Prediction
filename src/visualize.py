"""
Publication-grade data storytelling, forecast evaluation, and XAI visualizations.

Generates 8 high-resolution (300 DPI) figures designed for both classroom presentation
slides (16:9 friendly aspect ratios) and written academic reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import FIGURES_DIR
from src.explainability import ExplainabilityBundle
from src.features import FEATURE_LABELS
from src.models import HorizonRunArtifacts

# Clean academic styling
PALETTE_FAMILY = {
    "1. Econometric Benchmark": "#64748B",   # Slate gray
    "2. Regularized Linear": "#2563EB",      # Royal blue
    "3. Tree Ensemble": "#0D9488",           # Teal
    "4. Deep Learning": "#7C3AED",           # Violet
    "5. Hybrid Tree + Deep Net": "#E11D48",  # Crimson-rose highlight
    "6. Cross-Family Ensemble": "#EA580C",   # Amber-orange
}


def apply_academic_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFCFF",
        }
    )


def plot_fig01_eda_storytelling(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Slide 8 Storytelling: Rolling mean/volatility regimes + Phillips Curve & Cost-Push scatters."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.5, 8.2), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.36, wspace=0.24)

    # Panel A: 47-Year Headline & Core CPI + 36m Rolling Mean & Volatility Band
    ax1 = fig.add_subplot(gs[0, :])
    roll_mean = df["cpi_yoy"].rolling(36, min_periods=12).mean()
    roll_std = df["cpi_yoy"].rolling(36, min_periods=12).std()

    # Shade macro regimes lightly
    regime_spans = [
        ("1979-07-01", "1984-01-01", "#FEE2E2", "Volcker\nDisinflation"),
        ("1984-01-01", "2008-01-01", "#F1F5F9", "Great Moderation"),
        ("2008-01-01", "2019-01-01", "#EFF6FF", "Post-GFC\nLow Inflation"),
        ("2020-03-01", "2023-01-01", "#FEF3C7", "Supply\nShock"),
        ("2023-01-01", "2026-09-01", "#ECFDF5", "Disinflation\n(2023–26)"),
    ]
    for s_dt, e_dt, col, lbl in regime_spans:
        ax1.axvspan(pd.Timestamp(s_dt), pd.Timestamp(e_dt), color=col, alpha=0.65, zorder=0)
        mid_dt = pd.Timestamp(s_dt) + (pd.Timestamp(e_dt) - pd.Timestamp(s_dt)) / 2
        ax1.text(mid_dt, 14.3, lbl, ha="center", va="top", fontsize=8.2, fontweight="bold", color="#334155")

    ax1.fill_between(
        df.index,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color="#93C5FD",
        alpha=0.30,
        label="3Y Rolling Mean ± 1σ Band",
    )
    ax1.plot(df.index, df["cpi_yoy"], color="#0F172A", lw=1.8, label="Headline CPI YoY (%)")
    ax1.plot(df.index, df["core_cpi_yoy"], color="#2563EB", lw=1.35, ls="--", label="Core CPI YoY (%)")
    ax1.plot(df.index, roll_mean, color="#DC2626", lw=1.6, label="3Y Rolling Average")
    ax1.axvline(pd.Timestamp("2019-01-01"), color="#B91C1C", lw=1.8, ls=":", label="Out-of-Sample Split (2019-01)")
    ax1.axhline(2.0, color="#475569", lw=1.0, ls="-.", alpha=0.7)

    ax1.set_ylim(-2.8, 17.8)
    ax1.set_xlim(df.index.min(), df.index.max())
    ax1.set_ylabel("Inflation Rate (% YoY)")
    ax1.set_title("A. US CPI Inflation Regimes, Rolling Mean & Volatility (1979–2026)")
    ax1.legend(loc="upper center", ncol=5, frameon=True, facecolor="white", framealpha=0.95, fontsize=8.4)

    # Panel B: Regime-Dependent Phillips Curve Scatter (Unemployment vs CPI YoY)
    ax2 = fig.add_subplot(gs[1, 0])
    era_groups = [
        ("1979–1983 Stagflation", df.index < "1984-01-01", "#DC2626"),
        ("1984–2019 Great Mod. & Post-GFC", (df.index >= "1984-01-01") & (df.index < "2020-03-01"), "#2563EB"),
        ("2020–2022 Supply Shock", (df.index >= "2020-03-01") & (df.index < "2023-01-01"), "#D97706"),
        ("2023–2026 Disinflation", df.index >= "2023-01-01", "#059669"),
    ]
    for label, mask, color in era_groups:
        sub = df.loc[mask]
        ax2.scatter(sub["unrate"], sub["cpi_yoy"], c=color, alpha=0.62, s=24, label=label, edgecolors="none")
        if len(sub) > 8:
            xs = np.linspace(sub["unrate"].min(), sub["unrate"].max(), 50)
            coef = np.polyfit(sub["unrate"], sub["cpi_yoy"], deg=1)
            ax2.plot(xs, np.polyval(coef, xs), color=color, lw=1.8, alpha=0.9)

    ax2.set_xlabel("Civilian Unemployment Rate (%)")
    ax2.set_ylabel("Headline CPI Inflation (% YoY)")
    ax2.set_title("B. Shifting Phillips Curve Across Macro Regimes")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel C: Upstream Cost-Push Pass-Through colored by 1Y Inflation Expectations
    ax3 = fig.add_subplot(gs[1, 1])
    valid_sc = df.dropna(subset=["ppi_yoy", "target_cpi_h3", "mich_exp_1y"])
    sc = ax3.scatter(
        valid_sc["ppi_yoy"],
        valid_sc["target_cpi_h3"],
        c=valid_sc["mich_exp_1y"],
        cmap="viridis",
        s=24,
        alpha=0.75,
        edgecolors="none",
    )
    # Quadratic smooth curve illustrating nonlinear pass-through at high PPI surges
    xs_p = np.linspace(valid_sc["ppi_yoy"].min(), valid_sc["ppi_yoy"].max(), 100)
    poly2 = np.polyfit(valid_sc["ppi_yoy"], valid_sc["target_cpi_h3"], deg=2)
    ax3.plot(xs_p, np.polyval(poly2, xs_p), color="#DC2626", lw=2.0, label="Nonlinear Quadratic Fit")

    cbar = fig.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("1Y Mich. Inflation Expect. (%)", fontsize=9.5)
    ax3.set_xlabel("Upstream PPI Producer Inflation (% YoY)")
    ax3.set_ylabel("Future CPI Inflation at t+3 (% YoY)")
    ax3.set_title("C. Upstream Cost-Push & Expectations Synergy")
    ax3.legend(loc="upper left", frameon=True, facecolor="white")

    out_path = out_dir / "fig01_eda_storytelling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig02_correlation_heatmap(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Slide 8 Heatmap: Lower-triangle correlation matrix of key macroeconomic channels."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "target_cpi_h3",
        "cpi_yoy",
        "core_cpi_yoy",
        "cpi_3m_ann",
        "core_headline_gap",
        "mich_exp_1y",
        "ppi_yoy",
        "elec_yoy",
        "oil_yoy",
        "gdp_yoy",
        "unrate",
        "real_fedfunds",
        "m2_yoy",
        "stock_yoy",
    ]
    labels = [
        "Future CPI (t+3)",
        "Headline CPI (t)",
        "Core CPI (t)",
        "CPI 3m Momentum",
        "Core - Headline Gap",
        "1Y Inflation Expect.",
        "PPI Producer Infl.",
        "Electricity CPI Infl.",
        "WTI Oil Infl.",
        "Real GDP Growth",
        "Unemployment Rate",
        "Real Fed Funds Rate",
        "M2 Money Growth",
        "US Stock Market",
    ]
    corr = df[cols].dropna().corr()
    corr.index = labels
    corr.columns = labels

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11.4, 8.8), dpi=300)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-0.85,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.7},
        linewidths=0.6,
        cbar_kws={"shrink": 0.82, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Cross-Correlation Heatmap of Inflation & Macroeconomic Drivers (1979–2026)", pad=14)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    out_path = out_dir / "fig02_correlation_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _sig_stars(p_val: float, dm_stat: float) -> str:
    if dm_stat >= 0:
        return ""
    if p_val < 0.01:
        return "***"
    if p_val < 0.05:
        return "**"
    if p_val < 0.10:
        return "*"
    return ""


def plot_fig03_model_comparison_h3(run_h3: HorizonRunArtifacts, out_dir: Path = FIGURES_DIR) -> Path:
    """Slide 9 Flagship Results: Out-of-Sample RMSE Leaderboard + 2019-2026 Forecast Tracking."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.8, 6.4), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.25)

    # Left: RMSE Leaderboard sorted best (top) to worst (bottom)
    ax1 = fig.add_subplot(gs[0, 0])
    mdf = run_h3.metrics_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(mdf))
    colors = [PALETTE_FAMILY[fam] for fam in mdf["Family"]]

    bars = ax1.barh(y_pos, mdf["RMSE"], color=colors, height=0.68, edgecolor="none", alpha=0.92)
    rw_rmse = float(mdf.loc[mdf["Model"] == "Random Walk", "RMSE"].iloc[0])
    ax1.axvline(rw_rmse, color="#DC2626", ls="--", lw=1.6, label=f"Random Walk ({rw_rmse:.2f} pp)")

    for idx, row in mdf.iterrows():
        stars = _sig_stars(row["DM_p_value"], row["DM_Stat_vs_RW"])
        ax1.text(
            row["RMSE"] + 0.018,
            idx,
            f"{row['RMSE']:.3f}{stars}",
            va="center",
            fontsize=8.8,
            fontweight="bold" if idx < 3 else "normal",
            color="#0F172A",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(mdf["Model"])
    ax1.invert_yaxis()
    ax1.set_xlim(0, mdf["RMSE"].max() * 1.20)
    ax1.set_xlabel("Out-of-Sample RMSE (Percentage Points, h=3 Months)")
    ax1.set_title("A. Model Accuracy Ranking (2019–2026 Out-of-Sample)\n(* p<0.10, ** p<0.05, *** p<0.01 DM Test vs. Random Walk)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    # Right: Realized vs Predicted Inflation Trajectory across COVID Surge & Disinflation
    ax2 = fig.add_subplot(gs[0, 1])
    pdf = run_h3.predictions_df.copy()
    tdates = pd.to_datetime(pdf["target_realization_date"])

    ax2.axvspan(pd.Timestamp("2021-03-01"), pd.Timestamp("2023-01-01"), color="#FEF3C7", alpha=0.55, label="Supply-Shock Surge")
    ax2.plot(tdates, pdf["actual_cpi_yoy"], color="#0F172A", lw=2.5, label="Actual Headline CPI YoY", zorder=5)
    ax2.plot(tdates, pdf["Random Walk"], color="#94A3B8", lw=1.4, ls=":", label="Random Walk Benchmark")
    ax2.plot(tdates, pdf["Elastic Net (L1+L2)"], color="#2563EB", lw=1.5, ls="--", label="Elastic Net (Linear)")
    ax2.plot(tdates, pdf["XGBoost"], color="#0D9488", lw=1.7, label="XGBoost (Boosted Trees)")
    ax2.plot(tdates, pdf["XGB + Deep Net"], color="#E11D48", lw=2.0, label="XGB + Deep Net (Hybrid)")
    ax2.plot(tdates, pdf["Temporal Transformer"], color="#7C3AED", lw=1.5, ls="-.", label="Temporal Transformer")

    ax2.set_ylabel("Headline CPI Inflation (% YoY)")
    ax2.set_xlabel("Target Realization Month (t + 3)")
    ax2.set_title("B. Out-of-Sample 3-Month-Ahead Inflation Forecasts (2019–2026)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", fontsize=8.5)

    out_path = out_dir / "fig03_model_comparison_h3.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig04_multi_horizon_and_regimes(
    all_metrics_df: pd.DataFrame,
    regime_h3_df: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Multi-horizon scaling (h=1,3,6,12) and regime-specific stress-test performance."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.8), dpi=300)

    # Panel A: Relative RMSE vs Random Walk across Horizons h = 1, 3, 6, 12
    highlight_models = [
        ("OLS Phillips Curve", "#64748B", "s", "--"),
        ("Elastic Net (L1+L2)", "#2563EB", "o", "-"),
        ("XGBoost", "#0F766E", "D", "-"),
        ("MLP (Deep Net)", "#A855F7", "v", ":"),
        ("Temporal Transformer", "#6D28D9", "P", "-."),
        ("XGB + Deep Net", "#E11D48", "h", "-"),
        ("Forecast Combination", "#EA580C", "*", "--"),
    ]
    ax1.axhline(1.0, color="#DC2626", lw=1.6, ls=":", label="Random Walk Parity (U = 1.00)")
    for m_name, col, marker, ls in highlight_models:
        sub = all_metrics_df[all_metrics_df["Model"] == m_name].sort_values("Horizon_Months")
        ax1.plot(
            sub["Horizon_Months"],
            sub["Rel_RMSE_vs_RW"],
            color=col,
            marker=marker,
            ls=ls,
            lw=2.3 if m_name in ("XGB + Deep Net", "XGBoost", "Elastic Net (L1+L2)") else 1.5,
            markersize=7,
            label=m_name,
        )

    ax1.set_xticks([1, 3, 6, 12])
    ax1.set_xticklabels(["1m Ahead\n(Nowcast)", "3m Ahead\n(1 Quarter)", "6m Ahead\n(Half-Year)", "12m Ahead\n(1 Year)"])
    ax1.set_ylabel("Relative RMSE vs. Random Walk (Theil's U < 1 Beats RW)")
    ax1.set_title("A. Multi-Horizon Forecast Skill Across h = 1, 3, 6, 12 Months")
    ax1.legend(loc="best", frameon=True, facecolor="white", fontsize=8.2)

    # Panel B: Regime-Specific RMSE Heatmap at h=3
    selected_models = [
        "Random Walk",
        "OLS Phillips Curve",
        "Elastic Net (L1+L2)",
        "Random Forest",
        "XGBoost",
        "MLP (Deep Net)",
        "LSTM (Sequence)",
        "Temporal Transformer",
        "XGB + Deep Net",
        "Forecast Combination",
    ]
    piv = (
        regime_h3_df[regime_h3_df["Model"].isin(selected_models)]
        .pivot(index="Model", columns="Regime", values="RMSE")
        .reindex(selected_models)
    )
    short_reg_names = {
        c: (
            "1. Pre-COVID\n(2019)"
            if "Pre-COVID" in c
            else ("2. Supply Shock\n(2020–2022)" if "Pandemic" in c else "3. Disinflation\n(2023–2026)")
        )
        for c in piv.columns
    }
    ordered_cols = [
        "1. Pre-COVID\n(2019)",
        "2. Supply Shock\n(2020–2022)",
        "3. Disinflation\n(2023–2026)",
    ]
    piv = piv.rename(columns=short_reg_names)[ordered_cols]

    sns.heatmap(
        piv,
        cmap="YlGnBu_r",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5, "weight": "bold"},
        linewidths=0.8,
        cbar_kws={"label": "RMSE (Percentage Points)"},
        ax=ax2,
    )
    ax2.grid(False)
    ax2.set_title("B. Regime Stress-Test Accuracy (RMSE at h=3 Months)")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.sca(ax2)
    plt.xticks(rotation=0)

    plt.tight_layout()
    out_path = out_dir / "fig04_multi_horizon_and_regimes.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig05_xai_cross_family_importance(xai: ExplainabilityBundle, out_dir: Path = FIGURES_DIR) -> Path:
    """Compare Linear (Elastic Net coefficients) vs Tree (XGBoost SHAP) vs Deep Net (MLP attribution)."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_df = xai.importance_df.head(12).iloc[::-1].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.2), dpi=300, sharey=True)

    # Panel A: Signed Standardized Elastic Net Coefficients (shows economic direction!)
    coef_colors = ["#DC2626" if v > 0 else "#2563EB" for v in top_df["ElasticNet_Signed_Coef"]]
    ax1.barh(top_df["Feature_Label"], top_df["ElasticNet_Signed_Coef"], color=coef_colors, height=0.66, alpha=0.9)
    ax1.axvline(0.0, color="#0F172A", lw=1.1)
    ax1.set_xlabel("Standardized Coefficient on Δπ(t+3) (Red = Inflationary, Blue = Disinflationary)")
    ax1.set_title("A. Linear Direction & Sparsity (Elastic Net)")

    # Panel B: Side-by-side Relative Importance Share (%) across Elastic Net, XGBoost, and XGB + MLP
    y = np.arange(len(top_df))
    h_bar = 0.26
    ax2.barh(y + h_bar, top_df["ElasticNet_Share_pct"], height=h_bar, color="#2563EB", label="Elastic Net (Linear |β|)")
    ax2.barh(y, top_df["XGBoost_Share_pct"], height=h_bar, color="#0D9488", label="XGBoost (Exact TreeSHAP)")
    ax2.barh(y - h_bar, top_df["XGB_MLP_Share_pct"], height=h_bar, color="#E11D48", label="XGB + MLP (Hybrid)")

    ax2.set_yticks(y)
    ax2.set_yticklabels(top_df["Feature_Label"])
    ax2.set_xlabel("Normalized Feature Importance Share (%)")
    ax2.set_title("B. Top-Model Driver Consensus (Elastic Net vs. XGBoost vs. XGB + MLP)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white")

    plt.tight_layout()
    out_path = out_dir / "fig05_xai_cross_family_importance.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig06_shap_global_nonlinear(xai: ExplainabilityBundle, out_dir: Path = FIGURES_DIR) -> Path:
    """Publication-grade XGBoost TreeSHAP Beeswarm + Top-2 Nonlinear SHAP Dependence Curves."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.0, 6.4), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.28, hspace=0.38)

    # Left (spanning both rows): Custom crisp TreeSHAP Beeswarm for top 10 features
    ax_bee = fig.add_subplot(gs[:, 0])
    mean_abs = np.mean(np.abs(xai.shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[-10:]  # ascending so largest is on top
    rng = np.random.default_rng(42)

    cmap = plt.get_cmap("coolwarm")
    for row_pos, feat_i in enumerate(top_idx):
        sv = xai.shap_values[:, feat_i]
        fv = xai.X_test_labeled.iloc[:, feat_i].values
        # Percentile-normalize feature values [0, 1] for clean low-to-high color mapping
        p_low, p_high = np.percentile(fv, [5, 95])
        norm_fv = np.clip((fv - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
        jitter = rng.normal(0.0, 0.13, size=len(sv))
        ax_bee.scatter(
            sv,
            np.full_like(sv, row_pos) + jitter,
            c=norm_fv,
            cmap=cmap,
            s=26,
            alpha=0.82,
            edgecolors="none",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax_bee, pad=0.02)
    cbar.set_ticks([0.05, 0.5, 0.95])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.set_label("Feature Value Percentile", fontsize=9.5)

    ax_bee.axvline(0.0, color="#0F172A", lw=1.2, ls="--")
    ax_bee.set_yticks(np.arange(len(top_idx)))
    ax_bee.set_yticklabels([xai.X_test_labeled.columns[i] for i in top_idx])
    ax_bee.set_xlabel("SHAP Impact on 3-Month Inflation Trajectory (Percentage Points)")
    ax_bee.set_title("A. Global XGBoost TreeSHAP Summary (2019–2026 Out-of-Sample)")

    # Right Top & Right Bottom: Nonlinear SHAP Dependence Plots for Top 2 Economic Drivers
    top2_indices = top_idx[::-1][:2]
    panel_tags = ["B", "C"]
    for k, feat_i in enumerate(top2_indices):
        ax_dep = fig.add_subplot(gs[k, 1])
        feat_name = xai.X_test_labeled.columns[feat_i]
        fv = xai.X_test_labeled.iloc[:, feat_i].values
        sv = xai.shap_values[:, feat_i]
        anchor_vals = xai.X_test_labeled["Headline CPI YoY (%)"].values

        sc = ax_dep.scatter(
            fv,
            sv,
            c=anchor_vals,
            cmap="plasma",
            s=32,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.3,
        )
        # Smooth binned trend line
        order = np.argsort(fv)
        if len(fv) >= 12:
            xs_sorted = fv[order]
            ys_sorted = pd.Series(sv[order]).rolling(11, center=True, min_periods=3).mean().values
            ax_dep.plot(xs_sorted, ys_sorted, color="#0F172A", lw=2.0, label="Nonlinear SHAP Trend")

        ax_dep.axhline(0.0, color="#64748B", lw=1.0, ls=":")
        cb = fig.colorbar(sc, ax=ax_dep, pad=0.02)
        cb.set_label("Current CPI (%)", fontsize=8.5)
        ax_dep.set_xlabel(feat_name)
        ax_dep.set_ylabel("SHAP Effect (pp)")
        ax_dep.set_title(f"{panel_tags[k]}. Nonlinear Dependence: {feat_name}")
        ax_dep.legend(loc="best", frameon=True, facecolor="white", fontsize=8.0)

    out_path = out_dir / "fig06_shap_global_nonlinear.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_episode_waterfall(ax: plt.Axes, ep: Dict[str, object], panel_prefix: str) -> None:
    """Draw a crystal-clear anchored forecast attribution waterfall on `ax`."""
    cdf: pd.DataFrame = ep["contributions"]
    top_k = 6
    top_rows = cdf.iloc[:top_k].copy()
    other_shap = float(cdf.iloc[top_k:]["shap_pp"].sum()) + float(ep["base_delta"])

    labels = [f"Anchor CPI ({ep['origin_date']})"]
    deltas = [float(ep["anchor_cpi"])]
    starts = [0.0]
    running = float(ep["anchor_cpi"])

    for _, r in top_rows.iterrows():
        starts.append(running)
        d = float(r["shap_pp"])
        deltas.append(d)
        labels.append(f"{r['feature_label']}\n({r['feature_value']:.1f})")
        running += d

    starts.append(running)
    deltas.append(other_shap)
    labels.append("Other Features + Base")
    running += other_shap

    y_pos = np.arange(len(labels))
    for i, (st, dl) in enumerate(zip(starts, deltas)):
        if i == 0:
            col = "#475569"
            ax.barh(i, dl, left=0.0, color=col, height=0.60, alpha=0.88)
            ax.text(dl + 0.08, i, f"{dl:.2f}%", va="center", fontsize=8.8, fontweight="bold")
        else:
            col = "#DC2626" if dl >= 0 else "#2563EB"
            left = min(st, st + dl)
            width = abs(dl)
            ax.barh(i, width, left=left, color=col, height=0.60, alpha=0.90)
            txt = f"{dl:+.2f} pp"
            x_txt = (st + dl + 0.08) if dl >= 0 else (max(left - 0.08, 0.05))
            ha = "left" if dl >= 0 else "right"
            if x_txt < 0.8:
                x_txt = left + width + 0.08
                ha = "left"
            ax.text(x_txt, i, txt, va="center", ha=ha, fontsize=8.5, fontweight="bold", color=col)

    ax.axvline(float(ep["pred_cpi"]), color="#0D9488", lw=2.0, ls="--", label=f"XGBoost Forecast ({ep['pred_cpi']:.2f}%)")
    ax.axvline(float(ep["actual_cpi"]), color="#0F172A", lw=2.0, ls=":", label=f"Actual Realized ({ep['actual_cpi']:.2f}%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.7)
    ax.invert_yaxis()
    max_x = max(float(ep["anchor_cpi"]), float(ep["pred_cpi"]), float(ep["actual_cpi"]), running) + 1.4
    ax.set_xlim(0.0, max_x)
    ax.set_xlabel("Headline CPI Inflation (% YoY)")
    ax.set_title(f"{panel_prefix}. {ep['title']}\nOrigin {ep['origin_date']} → Target {ep['target_date']}")
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.3)


def plot_fig07_shap_episode_waterfalls(xai: ExplainabilityBundle, out_dir: Path = FIGURES_DIR) -> Path:
    """Local SHAP case-study explanations for the inflation surge vs disinflation turning point."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0), dpi=300)
    _draw_episode_waterfall(ax1, xai.episode_surge, "A")
    _draw_episode_waterfall(ax2, xai.episode_disinflation, "B")

    plt.tight_layout()
    out_path = out_dir / "fig07_shap_episode_waterfalls.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig08_transformer_attention_and_leakage(
    xai: ExplainabilityBundle,
    leakage_df: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Inside Sequence Deep Learning (Temporal Attention lags) + CV Data Leakage Proof."""
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.5), dpi=300)

    # Panel A: Temporal Transformer Lag Attention Profiles across Regimes
    attn_reg = xai.attention_by_regime_df.copy()
    short_names = []
    for r in attn_reg.index:
        if "Pre-COVID" in r:
            short_names.append("Pre-COVID (2019)")
        elif "Pandemic" in r:
            short_names.append("Supply Shock (2020–22)")
        else:
            short_names.append("Disinflation (2023–26)")
    attn_reg.index = short_names

    sns.heatmap(
        attn_reg * 100.0,
        cmap="Purples",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.6,
        cbar_kws={"label": "Attention Weight (%)"},
        ax=ax1,
    )
    ax1.grid(False)
    ax1.set_title("A. Temporal Transformer Attention Weights Across 12 Lookback Lags (%)")
    ax1.set_xlabel("Lookback Lag Month Entering Sequence Window [t-11 ... t]")
    ax1.set_ylabel("")
    plt.sca(ax1)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # Panel B: Methodological Rigor — Random K-Fold Leakage vs Purged TimeSeriesSplit
    x = np.arange(len(leakage_df))
    w = 0.34
    b1 = ax2.bar(
        x - w / 2,
        leakage_df["Random_KFold_RMSE_Leaky"],
        width=w,
        color="#EF4444",
        alpha=0.88,
        label="Random 5-Fold CV (Leaky / Look-Ahead)",
    )
    b2 = ax2.bar(
        x + w / 2,
        leakage_df["Chronological_Purged_TSCV_RMSE"],
        width=w,
        color="#1E3A8A",
        alpha=0.90,
        label="Purged Chronological TimeSeriesSplit (Causal)",
    )

    for idx, row in leakage_df.iterrows():
        ax2.text(
            idx - w / 2,
            row["Random_KFold_RMSE_Leaky"] + 0.03,
            f"{row['Random_KFold_RMSE_Leaky']:.2f}",
            ha="center",
            fontsize=8.8,
            color="#B91C1C",
            fontweight="bold",
        )
        ax2.text(
            idx + w / 2,
            row["Chronological_Purged_TSCV_RMSE"] + 0.03,
            f"{row['Chronological_Purged_TSCV_RMSE']:.2f}",
            ha="center",
            fontsize=8.8,
            color="#1E3A8A",
            fontweight="bold",
        )
        ax2.annotate(
            f"Fake gain:\n-{row['Artificial_Optimism_Bias_pct']:.0f}%",
            xy=(idx - w / 2, row["Random_KFold_RMSE_Leaky"] / 2),
            ha="center",
            va="center",
            fontsize=8.0,
            color="white",
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(leakage_df["Model"])
    ax2.set_ylim(0, leakage_df["Chronological_Purged_TSCV_RMSE"].max() * 1.25)
    ax2.set_ylabel("Validation RMSE on Δπ(t+3) (Percentage Points)")
    ax2.set_title("B. Why Chronological CV Matters: Random K-Fold Leakage Audit")
    ax2.legend(loc="upper left", frameon=True, facecolor="white")

    plt.tight_layout()
    out_path = out_dir / "fig08_transformer_attention_and_leakage.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig09_hybrid_xgb_deepnet(
    all_metrics_df: pd.DataFrame,
    run_h3: HorizonRunArtifacts,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Dedicated analysis of Hybrid `XGB + Deep Net`:
    Panel A: Grouped comparison vs Standalone XGBoost, MLP, and Transformer across horizons.
    Panel B: Inside the Hybrid Deep Trend Network — top XGBoost-SHAP, Trend, and Raw inputs.
    """
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.8), dpi=300)

    # Panel A: Relative RMSE vs RW across h = 1, 3, 6, 12 for Standalone vs Hybrid
    compare_models = [
        ("MLP (Deep Net)", "#A855F7"),
        ("Temporal Transformer", "#6D28D9"),
        ("XGBoost", "#0D9488"),
        ("XGB + Deep Net", "#E11D48"),
    ]
    horizons = [1, 3, 6, 12]
    x = np.arange(len(horizons))
    w = 0.19

    ax1.axhline(1.0, color="#64748B", lw=1.5, ls=":", label="Random Walk (1.00)")
    for k, (m_name, col) in enumerate(compare_models):
        sub = all_metrics_df[all_metrics_df["Model"] == m_name].sort_values("Horizon_Months")
        vals = sub["Rel_RMSE_vs_RW"].values
        offset = (k - 1.5) * w
        bars = ax1.bar(x + offset, vals, width=w, color=col, alpha=0.92, label=m_name)
        if m_name == "XGB + Deep Net":
            for bx, v in zip(x + offset, vals):
                ax1.text(bx, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#BE123C")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["h = 1m\n(Nowcast)", "h = 3m\n(1 Quarter)", "h = 6m\n(Half-Year)", "h = 12m\n(1 Year)"])
    ax1.set_ylim(0.68, 1.18)
    ax1.set_ylabel("Relative RMSE vs. Random Walk (Lower = Better)")
    ax1.set_title("A. Why Feeding XGBoost Features into a Deep Net Works\n(Standalone Deep Nets vs. XGBoost vs. Hybrid XGB + Deep Net)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel B: Top 12 Hybrid Features inside `XGB + Deep Net`
    hyb_imp = run_h3.fitted_objects["hybrid_feature_importance"]
    hyb_cols = run_h3.fitted_objects["hybrid_feature_names"]

    # Make human-readable labels for the hybrid feature names
    pretty_cols = []
    feat_types = []
    for c in hyb_cols:
        if c.startswith("Raw: "):
            code = c.replace("Raw: ", "")
            pretty_cols.append(f"Raw: {FEATURE_LABELS.get(code, code)}")
            feat_types.append("Raw Macro Input")
        elif c.startswith("XGB-SHAP: "):
            code = c.replace("XGB-SHAP: ", "")
            pretty_cols.append(f"XGB Nonlinear: {FEATURE_LABELS.get(code, code)}")
            feat_types.append("XGBoost Nonlinear SHAP")
        else:
            pretty_cols.append(c)
            feat_types.append("XGBoost Extracted Trend")

    hyb_df = pd.DataFrame(
        {"Label": pretty_cols, "Type": feat_types, "Imp": hyb_imp / (hyb_imp.sum() + 1e-8) * 100.0}
    ).sort_values("Imp", ascending=False).head(12).iloc[::-1]

    type_colors = {
        "XGBoost Extracted Trend": "#E11D48",
        "XGBoost Nonlinear SHAP": "#0D9488",
        "Raw Macro Input": "#2563EB",
    }
    bar_cols = [type_colors[t] for t in hyb_df["Type"]]
    ax2.barh(hyb_df["Label"], hyb_df["Imp"], color=bar_cols, height=0.66, alpha=0.92)

    # Legend patches for feature categories
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=col, label=lbl) for lbl, col in type_colors.items()]
    ax2.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", fontsize=8.5)
    ax2.set_xlabel("Hybrid Deep Network Attribution Share (%)")
    ax2.set_title("B. What Inside `XGB + Deep Net` Drives Trend Refinement (h=3m)")

    plt.tight_layout()
    out_path = out_dir / "fig09_hybrid_xgb_deepnet.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig10_change_and_policy_shocks(
    pred_h3_df: pd.DataFrame,
    event_windows_df: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Answer 'Are Models Just Lagging?' by plotting 3-Month Inflation CHANGES
    (pi_{t+3} - pi_t, where Pure-Lag Random Walk = 0) across annotated policy & shock
    events, paired with Event-Window RMSE and a 4-Quadrant Turning-Point Map.
    """
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    p = pred_h3_df.copy()
    tdates = pd.to_datetime(p["target_realization_date"])
    act_d = p["actual_cpi_yoy"] - p["current_cpi_yoy"]
    xgb_d = p["XGBoost"] - p["current_cpi_yoy"]
    enet_d = p["Elastic Net (L1+L2)"] - p["current_cpi_yoy"]
    hyb_d = p["XGB + Deep Net"] - p["current_cpi_yoy"]

    fig = plt.figure(figsize=(15.4, 8.8), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.18, 1.0], hspace=0.40, wspace=0.28)

    # Panel A: 3-Month Inflation CHANGE (Eliminates Visual Level-Lag Illusion!)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(tdates, 0.0, act_d, color="#CBD5E1", alpha=0.45, label="Actual 3m Inflation Shift: π(t+3) − π(t)")
    ax1.axhline(0.0, color="#64748B", lw=2.0, ls=":", label="Random Walk / Pure Lag (= 0.00 Change Always)")
    ax1.plot(tdates, act_d, color="#0F172A", lw=2.3, label="Actual 3m Change (pp)", zorder=4)
    ax1.plot(tdates, enet_d, color="#2563EB", lw=1.6, ls="--", label="Elastic Net Predicted Change")
    ax1.plot(tdates, xgb_d, color="#0D9488", lw=2.1, label="XGBoost Predicted Change", zorder=5)
    ax1.plot(tdates, hyb_d, color="#E11D48", lw=1.8, ls="-.", label="XGB + Deep Net Predicted Change")

    # Annotated Policy & Shock Milestones (cleanly spaced below top legend band)
    events = [
        ("2020-04-01", "COVID Lockdown\n(Surprise Shock)", -2.42, "#DC2626"),
        ("2021-04-01", "Stimulus + Base\nSurge (+Anticipated!)", 2.35, "#D97706"),
        ("2022-03-01", "Fed Liftoff &\nUkraine Oil Shock", 2.48, "#7C3AED"),
        ("2022-11-01", "Peak 75bp Fed Hikes\n(Disinflation Anticipated!)", -2.35, "#059669"),
        ("2024-09-01", "Fed Rate-Cut\nPivot (5.3%→4.8%)", 1.85, "#1E40AF"),
    ]
    for dt_str, txt, y_pos, col in events:
        dt = pd.Timestamp(dt_str)
        ax1.axvline(dt, color=col, lw=1.2, ls="--", alpha=0.75)
        ax1.text(
            dt,
            y_pos,
            txt,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=col,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": col, "alpha": 0.94},
            zorder=6,
        )

    ax1.set_ylim(-3.05, 4.45)
    ax1.set_ylabel("3-Month CPI Change: π(t+3) − π(t) (pp)")
    ax1.set_title("A. Predicted vs. Actual 3-Month Inflation CHANGES Across Policy & Shock Events (Pure Lag = Flat 0 Line)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper center", ncol=3, frameon=True, facecolor="white", fontsize=8.3)

    # Panel B: RMSE Across the 5 Specific Shock / Policy Windows
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(event_windows_df))
    w = 0.19
    ax2.bar(x - 1.5 * w, event_windows_df["RandomWalk_PureLag_RMSE"], width=w, color="#94A3B8", label="Random Walk (Pure Lag)")
    ax2.bar(x - 0.5 * w, event_windows_df["ElasticNet_RMSE"], width=w, color="#2563EB", label="Elastic Net")
    ax2.bar(x + 0.5 * w, event_windows_df["XGBoost_RMSE"], width=w, color="#0D9488", label="XGBoost")
    ax2.bar(x + 1.5 * w, event_windows_df["XGB_DeepNet_RMSE"], width=w, color="#E11D48", label="XGB + Deep Net")

    for idx, r in event_windows_df.iterrows():
        top_y = max(r["RandomWalk_PureLag_RMSE"], r["ElasticNet_RMSE"], r["XGBoost_RMSE"], r["XGB_DeepNet_RMSE"])
        pct = r["Best_ML_vs_PureLag_pct"]
        tag = f"{pct:+.0f}% vs Lag"
        tag_col = "#047857" if pct < 0 else "#B91C1C"
        ax2.text(idx, top_y + 0.05, tag, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=tag_col)

    ax2.set_xticks(x)
    ax2.set_xticklabels(event_windows_df["Plot_Label"], fontsize=8.4)
    ax2.set_ylim(0, event_windows_df[["RandomWalk_PureLag_RMSE", "ElasticNet_RMSE"]].values.max() * 1.26)
    ax2.set_ylabel("Out-of-Sample RMSE (pp)")
    ax2.set_title("B. Forecast RMSE by Policy Era (ML Beats Lag in 4 of 5)")
    ax2.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", fontsize=8.0)

    # Panel C: 4-Quadrant Directional Turning-Point Scatter (XGBoost vs Actual Change)
    ax3 = fig.add_subplot(gs[1, 1])
    lim_min, lim_max = -2.6, 3.2
    # Shade Q1 (+/+) and Q3 (-/-) correct directional quadrants softly green
    ax3.axvspan(0, lim_max, ymin=(0 - lim_min) / (lim_max - lim_min), ymax=1.0, color="#DCFCE7", alpha=0.45, zorder=0)
    ax3.axvspan(lim_min, 0, ymin=0.0, ymax=(0 - lim_min) / (lim_max - lim_min), color="#DCFCE7", alpha=0.45, zorder=0)
    ax3.axhline(0.0, color="#475569", lw=1.1, ls="--")
    ax3.axvline(0.0, color="#475569", lw=1.1, ls="--")
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], color="#0F172A", lw=1.2, ls=":", alpha=0.7, label="Perfect 45° Line")

    correct = np.sign(act_d) == np.sign(xgb_d)
    ax3.scatter(act_d[correct], xgb_d[correct], c="#0D9488", s=34, alpha=0.85, label=f"Correct Direction ({correct.mean()*100:.1f}%)")
    ax3.scatter(act_d[~correct], xgb_d[~correct], c="#EF4444", s=30, alpha=0.75, marker="x", label="Missed Surprise Shock")

    ax3.text(1.75, 2.55, "Correctly Anticipated\nAcceleration (+/+)", fontsize=8.2, fontweight="bold", color="#065F46", ha="center")
    ax3.text(-1.45, -2.15, "Correctly Anticipated\nDisinflation (−/−)", fontsize=8.2, fontweight="bold", color="#065F46", ha="center")

    ax3.set_xlim(lim_min, lim_max)
    ax3.set_ylim(lim_min, lim_max)
    ax3.set_xlabel("Actual 3-Month Inflation Change (pp)")
    ax3.set_ylabel("XGBoost Predicted 3-Month Change (pp)")
    ax3.set_title("C. Turning-Point Quadrant Test (Pure Lag Stuck at y = 0)")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.0)

    out_path = out_dir / "fig10_change_and_policy_shocks.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig11_xai_policy_playbook(
    imp_df: pd.DataFrame,
    channel_horizon_df: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Practical XAI Implications:
    Panel A: Non-Mechanical Economic Levers (re-normalized excluding calendar base effect)
             showing exact % importance across Linear, TreeSHAP, and Deep MLP.
    Panel B: The Macroeconomic Transmission Clock across Horizons h = 1, 3, 6, 12m.
    """
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0), dpi=300)

    # Panel A: Pure Economic Signals (excluding base_3m_ann) for Elastic Net, XGBoost & XGB + MLP ONLY
    econ_df = imp_df[imp_df["Feature_Code"] != "base_3m_ann"].copy()
    for col in ["ElasticNet_Share_pct", "XGBoost_Share_pct", "XGB_MLP_Share_pct"]:
        econ_df[col] = econ_df[col] / (econ_df[col].sum() + 1e-12) * 100.0
    econ_df["Econ_Consensus"] = (
        econ_df["ElasticNet_Share_pct"] + econ_df["XGBoost_Share_pct"] + econ_df["XGB_MLP_Share_pct"]
    ) / 3.0
    top_econ = econ_df.sort_values("Econ_Consensus", ascending=False).head(10).iloc[::-1]

    y = np.arange(len(top_econ))
    h_b = 0.26
    ax1.barh(y + h_b, top_econ["ElasticNet_Share_pct"], height=h_b, color="#2563EB", label="Elastic Net (Linear)")
    ax1.barh(y, top_econ["XGBoost_Share_pct"], height=h_b, color="#0D9488", label="XGBoost (Exact TreeSHAP)")
    ax1.barh(y - h_b, top_econ["XGB_MLP_Share_pct"], height=h_b, color="#E11D48", label="XGB + MLP (Hybrid)")

    for pos, (_, r) in enumerate(top_econ.iterrows()):
        max_bar = max(r["ElasticNet_Share_pct"], r["XGBoost_Share_pct"], r["XGB_MLP_Share_pct"])
        ax1.text(
            max_bar + 0.4,
            pos,
            f"Avg: {r['Econ_Consensus']:.1f}%",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#0F172A",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(top_econ["Feature_Label"])
    ax1.set_xlim(0, top_econ[["ElasticNet_Share_pct", "XGBoost_Share_pct", "XGB_MLP_Share_pct"]].values.max() * 1.24)
    ax1.set_xlabel("Share of Non-Base Economic Attribution (%)")
    ax1.set_title("A. Top Economic Levers: Elastic Net vs. XGBoost vs. XGB + MLP\n(PPI, Sticky Electricity, M2 & Real Rates Beat Unemployment)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.4)

    # Panel B: Stacked Bar 'Macroeconomic Transmission Clock' across h = 1, 3, 6, 12m
    ch_cols = [c for c in channel_horizon_df.columns if c != "Horizon_Months"]
    ch_colors = ["#475569", "#2563EB", "#D97706", "#0D9488", "#E11D48"]
    x_h = np.arange(len(channel_horizon_df))
    bottom = np.zeros(len(channel_horizon_df))

    for col, c_hex in zip(ch_cols, ch_colors):
        vals = channel_horizon_df[col].values
        ax2.bar(x_h, vals, bottom=bottom, width=0.56, color=c_hex, alpha=0.92, label=col)
        for ix, v in enumerate(vals):
            if v >= 6.5:
                ax2.text(
                    ix,
                    bottom[ix] + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    fontweight="bold",
                    color="white",
                )
        bottom += vals

    ax2.set_xticks(x_h)
    ax2.set_xticklabels(["h = 1m\n(Electricity, Oil &\nPPI Hit First: 28.6%)", "h = 3m\n(Base + Sticky\nTariffs Build Up)", "h = 6m\n(Monetary Policy &\nGDP Demand Peak!)", "h = 12m\n(Expectations Jump\n10× to 16.7% Share!)"])
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Share of Total XGBoost SHAP Attribution (%)")
    ax2.set_title("B. The Macroeconomic Transmission Clock Across Forecast Horizons\n(Which Indicator Policymakers Should Watch by Horizon)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", fontsize=8.3)

    plt.tight_layout()
    out_path = out_dir / "fig11_xai_policy_playbook.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig12_cross_paper_synthesis(
    feat_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Show how unifying all 4 ECO331 literature clusters (Electricity Price Index,
    US Stock Market, Real GDP Growth + CPI Core) strengthens both economic insight & accuracy:
      Panel A: Sticky Electricity Tariffs vs. Volatile Crude Oil vs. Headline CPI (2019–2026)
      Panel B: Controlled Out-of-Sample Ablation Across the 4 Course Paper Clusters (h = 3m)
    """
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.9), dpi=300)

    # Panel A: Sticky Electricity CPI vs Volatile Crude Oil vs Headline CPI (2019–2026)
    sub = feat_df.loc[feat_df.index >= pd.Timestamp("2019-01-01")].copy()
    ax1.plot(sub.index, sub["cpi_yoy"], color="#0F172A", lw=2.4, label="Headline CPI YoY (%) [Left]", zorder=4)
    ax1.plot(sub.index, sub["elec_yoy"], color="#D97706", lw=2.3, ls="-", label="Electricity CPI YoY (%) — Sticky Utility Cost [Left]", zorder=5)
    ax1.plot(sub.index, sub["gdp_yoy"], color="#2563EB", lw=1.7, ls="--", label="Real GDP Growth YoY (%, Lag 1Q) [Left]", zorder=3)
    ax1.axhline(2.0, color="#64748B", lw=1.0, ls=":")

    ax1_r = ax1.twinx()
    ax1_r.fill_between(sub.index, 0.0, sub["oil_yoy"], color="#94A3B8", alpha=0.22, label="WTI Crude Oil YoY (%) — Noisy [Right]")
    ax1_r.plot(sub.index, sub["stock_yoy"], color="#7C3AED", lw=1.35, ls=":", alpha=0.85, label="US Stock Market YoY (%) [Right]")
    ax1_r.set_ylabel("Crude Oil & Stock Market YoY (%)", color="#475569")
    ax1_r.set_ylim(-68, 138)
    ax1_r.grid(False)

    ax1.set_ylim(-8.2, 16.8)
    ax1.set_ylabel("CPI, Electricity CPI & Real GDP YoY (%)")
    ax1.set_title("A. Why Electricity Beats Oil at 3–6m: Sticky Tariffs vs. Noisy Oil\n(Uniting Kuskaya 2025 Electricity, Vamossy 2024 Stocks & Yoon 2021 GDP)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor="white", fontsize=7.9)

    # Panel B: Controlled Ablation Bar Chart (h = 3m Out-of-Sample RMSE)
    x = np.arange(len(ablation_df))
    w = 0.34
    ax2.bar(x - w / 2, ablation_df["ElasticNet_h3_RMSE"], width=w, color="#2563EB", alpha=0.92, label="Elastic Net (Regularized Linear)")
    ax2.bar(x + w / 2, ablation_df["XGBoost_h3_RMSE"], width=w, color="#0D9488", alpha=0.92, label="XGBoost (Boosted Trees)")
    ax2.axhline(1.004, color="#64748B", lw=1.4, ls=":", label="Random Walk Benchmark (1.004 pp)")

    for idx, r in ablation_df.iterrows():
        ax2.text(
            idx - w / 2,
            r["ElasticNet_h3_RMSE"] + 0.010,
            f"{r['ElasticNet_h3_RMSE']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#1E40AF",
        )
        ax2.text(
            idx + w / 2,
            r["XGBoost_h3_RMSE"] + 0.010,
            f"{r['XGBoost_h3_RMSE']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#0F766E",
        )
        if idx == len(ablation_df) - 1:
            ax2.text(
                idx,
                max(r["ElasticNet_h3_RMSE"], r["XGBoost_h3_RMSE"]) + 0.058,
                f"ENet {r['ElasticNet_Gain_vs_Core_pct']:+.1f}% vs Core!",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color="#1E40AF",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#EFF6FF", "edgecolor": "#2563EB", "alpha": 0.96},
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(ablation_df["Plot_Label"], fontsize=8.4)
    ax2.set_ylim(0.55, 1.09)
    ax2.set_ylabel("Out-of-Sample RMSE at h = 3m (pp, Lower = Better)")
    ax2.set_title("B. Controlled Ablation Across the 4 Course Literature Clusters\n(Adding Electricity + Equities + Real GDP Cuts Linear Error to 0.777 pp)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.3)

    plt.tight_layout()
    out_path = out_dir / "fig12_cross_paper_synthesis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig13_transformer_explainability(
    tf_xai: Dict[str, pd.DataFrame],
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Dedicated 2D Sequence Explainability for the `Temporal Transformer`
    (explaining why it wins #1 overall at h = 12m with RMSE = 2.384 pp):
      Panel A: Integrated-Gradients Channel Share (%) at h=12m (#1 Winner) vs. h=3m (All 14 Channels)
      Panel B: 2D Spatio-Temporal Attribution Heatmap [Top 8 Channels x 12 Lookback Lags] at h=12m
      Panel C: Regime-Adaptive U-Shaped Self-Attention Lag Weights at h=12m
    """
    apply_academic_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_df = tf_xai["channel_df"].iloc[::-1].copy()  # All 14 sequence channels!
    sp_df = tf_xai["spatiotemporal_h12_df"].copy()
    attn_12 = tf_xai["attention_regime_h12_df"].copy()

    fig = plt.figure(figsize=(16.2, 7.8), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.28], height_ratios=[1.25, 0.90], wspace=0.52, hspace=0.44)

    # Panel A: Channel Attribution Share (%) at h=12m vs h=3m across all 14 sequence channels
    ax1 = fig.add_subplot(gs[:, 0])
    y = np.arange(len(ch_df))
    hb = 0.36
    ax1.barh(y + hb / 2, ch_df["TF_h12_Share_pct"], height=hb, color="#6D28D9", alpha=0.92, label="h = 12m (#1 Overall: 2.384 pp)")
    ax1.barh(y - hb / 2, ch_df["TF_h3_Share_pct"], height=hb, color="#94A3B8", alpha=0.88, label="h = 3m (Short-Horizon Compare)")

    for pos, (_, r) in enumerate(ch_df.iterrows()):
        mx = max(r["TF_h12_Share_pct"], r["TF_h3_Share_pct"])
        ax1.text(
            mx + 0.35,
            pos,
            f"{r['TF_h12_Share_pct']:.1f}%",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color="#5B21B6",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(ch_df["Feature_Label"], fontsize=8.6)
    ax1.set_xlim(0, ch_df[["TF_h12_Share_pct", "TF_h3_Share_pct"]].values.max() * 1.22)
    ax1.set_xlabel("Integrated-Gradients Attribution Share (%)")
    ax1.set_title("A. Which Macro Channels Drive the Transformer?\n(Integrated Gradients: h = 12m vs. h = 3m)")
    ax1.legend(loc="center right", frameon=True, facecolor="white", fontsize=8.2)

    # Panel B: 2D Spatio-Temporal Attribution Heatmap [Top 8 Channels x 12 Lookback Lags] at h=12m
    ax2 = fig.add_subplot(gs[0, 1])
    top8_labels = tf_xai["channel_df"]["Feature_Label"].head(8).tolist()
    sub_sp = sp_df.loc[top8_labels]
    sns.heatmap(
        sub_sp,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7.8, "weight": "bold"},
        linewidths=0.5,
        cbar_kws={"label": "% of Total 12m Attribution"},
        ax=ax2,
    )
    ax2.grid(False)
    ax2.set_title("B. 2D Spatio-Temporal Map (h = 12m): Feature × Lag Month")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.sca(ax2)
    plt.xticks(rotation=28, ha="right", fontsize=8.2)
    plt.yticks(rotation=0, fontsize=8.3)

    # Panel C: Regime Lag Attention Weights at h = 12m
    ax3 = fig.add_subplot(gs[1, 1])
    short_reg = []
    for r_lbl in attn_12.index:
        if "Pre-COVID" in r_lbl:
            short_reg.append("Pre-COVID (2019)")
        elif "Pandemic" in r_lbl:
            short_reg.append("Supply Shock (2020–22)")
        else:
            short_reg.append("Disinflation (2023–26)")
    attn_12.index = short_reg

    sns.heatmap(
        attn_12 * 100.0,
        cmap="Purples",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8.0, "weight": "bold"},
        linewidths=0.5,
        cbar_kws={"label": "Attention Weight (%)"},
        ax=ax3,
    )
    ax3.grid(False)
    ax3.set_title("C. Self-Attention Lag Weights at h = 12m Across Regimes (%)")
    ax3.set_xlabel("Lookback Lag Month Entering 12-Month Sequence [t-11 ... t]")
    ax3.set_ylabel("")
    plt.sca(ax3)
    plt.xticks(rotation=28, ha="right", fontsize=8.2)
    plt.yticks(rotation=0, fontsize=8.4)

    out_path = out_dir / "fig13_transformer_explainability.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

