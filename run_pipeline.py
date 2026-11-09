"""
Master end-to-end runner for the ECO331 Explainable Inflation Forecasting project.

Usage:
    uv run python run_pipeline.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src import FIGURES_DIR, OUTPUTS_DIR, TABLES_DIR
from src.build_presentation import build_course_presentation
from src.data_loader import generate_data_dictionary_and_adf, load_macro_dataset
from src.explainability import (
    build_actionable_policy_playbook,
    compute_horizon_channel_shap,
    compute_policy_event_windows,
    run_cross_paper_ablation,
    run_explainability_analysis,
    run_transformer_explainability,
)
from src.features import HORIZONS, build_feature_matrix, prepare_horizon_dataset
from src.models import (
    HorizonRunArtifacts,
    run_cv_leakage_diagnostic,
    set_global_seed,
    train_and_evaluate_horizon,
)
from src.visualize import (
    plot_fig01_eda_storytelling,
    plot_fig02_correlation_heatmap,
    plot_fig03_model_comparison_h3,
    plot_fig04_multi_horizon_and_regimes,
    plot_fig05_xai_cross_family_importance,
    plot_fig06_shap_global_nonlinear,
    plot_fig07_shap_episode_waterfalls,
    plot_fig08_transformer_attention_and_leakage,
    plot_fig09_hybrid_xgb_deepnet,
    plot_fig10_change_and_policy_shocks,
    plot_fig11_xai_policy_playbook,
    plot_fig12_cross_paper_synthesis,
    plot_fig13_transformer_explainability,
)


def _img_to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_self_contained_html_report(
    run_h3: HorizonRunArtifacts,
    all_metrics_df: pd.DataFrame,
    adf_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Create a clean, standalone HTML presentation dashboard with embedded figures & tables."""
    mdf = run_h3.metrics_df.sort_values("RMSE").copy()
    for col in ["RMSE", "MAE", "Rel_RMSE_vs_RW", "R2", "DM_Stat_vs_RW"]:
        mdf[col] = mdf[col].map(lambda v: f"{v:.3f}")
    mdf["Directional_Acc_pct"] = mdf["Directional_Acc_pct"].map(lambda v: f"{v:.1f}%")
    mdf["DM_p_value"] = mdf["DM_p_value"].map(lambda v: f"{v:.4f}")

    figs = [
        ("1. Macroeconomic Regimes, Phillips Curve Shifts & Cost-Push Nonlinearity", FIGURES_DIR / "fig01_eda_storytelling.png"),
        ("2. Cross-Correlation Heatmap of Macro Channels", FIGURES_DIR / "fig02_correlation_heatmap.png"),
        ("3. Flagship 3-Month-Ahead Model Comparison & Out-of-Sample Tracking (2019–2026)", FIGURES_DIR / "fig03_model_comparison_h3.png"),
        ("4. Are Models Just Lagging? 3-Month Change-Space & Policy/Shock Event Audit", FIGURES_DIR / "fig10_change_and_policy_shocks.png"),
        ("5. Multi-Horizon Forecast Skill (1–12m) & Regime Stress-Test", FIGURES_DIR / "fig04_multi_horizon_and_regimes.png"),
        ("6. Hybrid Innovation: Feeding XGBoost Features & Trends into a Deep Net", FIGURES_DIR / "fig09_hybrid_xgb_deepnet.png"),
        ("7. Cross-Paper Synthesis: Adding Sticky Electricity, US Equities & Real GDP", FIGURES_DIR / "fig12_cross_paper_synthesis.png"),
        ("8. Explainable AI I: Elastic Net vs. XGBoost TreeSHAP vs. XGB + MLP Consensus", FIGURES_DIR / "fig05_xai_cross_family_importance.png"),
        ("9. Explainable AI II: Global XGBoost TreeSHAP Summary & Nonlinear Thresholds", FIGURES_DIR / "fig06_shap_global_nonlinear.png"),
        ("10. Explainable AI III: Local Case-Study Waterfalls (Surge vs. Disinflation)", FIGURES_DIR / "fig07_shap_episode_waterfalls.png"),
        ("11. Inside Sequence Deep Learning I: Temporal Attention Lags & CV Leakage Audit", FIGURES_DIR / "fig08_transformer_attention_and_leakage.png"),
        ("12. Inside Sequence Deep Learning II: Temporal Transformer 2D Explainability (Why It Wins at h=12m)", FIGURES_DIR / "fig13_transformer_explainability.png"),
        ("13. Policy Implications of XAI: Non-Base Economic Levers & Macroeconomic Transmission Clock", FIGURES_DIR / "fig11_xai_policy_playbook.png"),
    ]

    fig_blocks = "\n".join(
        f"""
        <section class="card">
          <h2>{title}</h2>
          <img src="{_img_to_data_uri(fp)}" alt="{title}" />
        </section>
        """
        for title, fp in figs
    )

    event_html = ""
    if (TABLES_DIR / "policy_shock_windows_h3.csv").exists():
        ev_df = pd.read_csv(TABLES_DIR / "policy_shock_windows_h3.csv").drop(columns=["Plot_Label"], errors="ignore")
        event_html = f"""
    <section class="card">
      <h2>Policy &amp; Shock Window Audit (h = 3 Months Ahead: ML vs. Pure Lag)</h2>
      {ev_df.to_html(index=False, border=0)}
    </section>"""

    playbook_html = ""
    if (TABLES_DIR / "xai_policy_playbook.csv").exists():
        pb_df = pd.read_csv(TABLES_DIR / "xai_policy_playbook.csv")
        playbook_html = f"""
    <section class="card">
      <h2>Actionable Explainability Playbook: Which Indicators Matter, How Much &amp; How to Use Them</h2>
      {pb_df.to_html(index=False, border=0)}
    </section>"""

    ablation_html = ""
    if (TABLES_DIR / "cross_paper_channel_ablation.csv").exists():
        ab_df = pd.read_csv(TABLES_DIR / "cross_paper_channel_ablation.csv").drop(columns=["Plot_Label"], errors="ignore")
        ablation_html = f"""
    <section class="card">
      <h2>Cross-Paper Literature Synthesis Ablation: Core Macro vs. +Electricity, +Equities &amp; +Real GDP</h2>
      {ab_df.to_html(index=False, border=0)}
    </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>ECO331 — Explainable Macroeconomic Inflation Forecasting</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #F8FAFC; color: #0F172A; margin: 0; padding: 0;
    }}
    header {{
      background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
      color: white; padding: 36px 48px; border-bottom: 5px solid #0D9488;
    }}
    header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    header p {{ margin: 0; color: #CBD5E1; font-size: 15px; line-height: 1.5; }}
    main {{ max-width: 1240px; margin: 28px auto; padding: 0 24px 60px; }}
    .card {{
      background: white; border-radius: 12px; padding: 24px 28px;
      margin-bottom: 26px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
      border: 1px solid #E2E8F0;
    }}
    h2 {{ margin-top: 0; font-size: 19px; color: #0F172A; border-left: 4px solid #0D9488; padding-left: 12px; }}
    img {{ width: 100%; height: auto; border-radius: 6px; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 12px; }}
    th {{ background: #0F172A; color: white; text-align: left; padding: 10px 12px; }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #E2E8F0; }}
    tr:nth-child(even) {{ background: #F8FAFC; }}
    tr:first-child td {{ font-weight: 700; background: #ECFDF5; }}
  </style>
</head>
<body>
  <header>
    <h1>Explainable Macroeconomic Inflation Forecasting (ECO331 — 2026)</h1>
    <p>Econometric Benchmarks vs. Regularized Linear, Gradient Boosted Trees &amp; Deep Sequence Models (MLP, LSTM, Temporal Transformer) on US Monthly FRED Data (1978–2026)</p>
  </header>
  <main>
    <section class="card">
      <h2>Flagship Out-of-Sample Leaderboard (h = 3 Months Ahead, 2019–2026)</h2>
      {mdf.to_html(index=False, border=0)}
    </section>
    {event_html}
    {ablation_html}
    {playbook_html}
    {fig_blocks}
    <section class="card">
      <h2>Methodological Audit: Random K-Fold Leakage vs. Purged Chronological CV</h2>
      {leakage_df.to_html(index=False, border=0)}
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    t0 = time.time()
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    set_global_seed(42)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading FRED monthly macroeconomic dataset (1978–2026)...")
    raw_panel = load_macro_dataset(force_refresh=False)
    feat_df = build_feature_matrix(raw_panel)
    adf_df = generate_data_dictionary_and_adf(raw_panel, feat_df)
    print(f"      Loaded {len(raw_panel)} raw monthly rows; {len(feat_df)} stationary feature rows.")

    print("[2/6] Generating Exploratory Data Storytelling plots (Slide 8)...")
    plot_fig01_eda_storytelling(feat_df)
    plot_fig02_correlation_heatmap(feat_df)

    print("[3/6] Training & evaluating 11 models across horizons h = [1, 3, 6, 12] months...")
    horizon_runs: Dict[int, HorizonRunArtifacts] = {}
    horizon_datasets = {}
    all_metrics = []

    for h in HORIZONS:
        ds_h = prepare_horizon_dataset(feat_df, horizon=h)
        horizon_datasets[h] = ds_h
        run_h = train_and_evaluate_horizon(ds_h)
        horizon_runs[h] = run_h
        all_metrics.append(run_h.metrics_df)
        best_m = run_h.metrics_df.sort_values("RMSE").iloc[0]
        rw_m = run_h.metrics_df.loc[run_h.metrics_df["Model"] == "Random Walk"].iloc[0]
        print(
            f"      h={h:2d}m | Train={len(ds_h.X_train_full)} | Test={len(ds_h.X_test)} | "
            f"RW RMSE={rw_m['RMSE']:.3f} pp -> Best={best_m['Model']} ({best_m['RMSE']:.3f} pp, "
            f"Rel={best_m['Rel_RMSE_vs_RW']:.3f}, DM p={best_m['DM_p_value']:.4f})"
        )

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    run_h3 = horizon_runs[3]
    ds_h3 = horizon_datasets[3]

    # Save tabular outputs
    run_h3.metrics_df.sort_values("RMSE").to_csv(TABLES_DIR / "metrics_h3_flagship.csv", index=False)
    all_metrics_df.to_csv(TABLES_DIR / "metrics_all_horizons.csv", index=False)
    run_h3.regime_metrics_df.to_csv(TABLES_DIR / "regime_metrics_h3.csv", index=False)
    run_h3.predictions_df.to_csv(TABLES_DIR / "predictions_h3_out_of_sample.csv", index=False)

    print("[4/6] Running Multi-Family Explainability & 4-Cluster Course Literature Ablation...")
    xai_h3 = run_explainability_analysis(ds_h3, run_h3)
    tf_xai = run_transformer_explainability(ds_h3, run_h3, horizon_datasets[12], horizon_runs[12])
    event_windows_df = compute_policy_event_windows(run_h3.predictions_df)
    channel_horizon_df = compute_horizon_channel_shap(horizon_runs, ds_h3.feature_names)
    playbook_df = build_actionable_policy_playbook(xai_h3.importance_df)
    ablation_df = run_cross_paper_ablation(feat_df)
    leakage_df = run_cv_leakage_diagnostic(ds_h3)
    leakage_df.to_csv(TABLES_DIR / "cv_leakage_diagnostic.csv", index=False)

    print("[5/6] Rendering empirical & explainability figures (300 DPI)...")
    plot_fig03_model_comparison_h3(run_h3)
    plot_fig04_multi_horizon_and_regimes(all_metrics_df, run_h3.regime_metrics_df)
    plot_fig05_xai_cross_family_importance(xai_h3)
    plot_fig06_shap_global_nonlinear(xai_h3)
    plot_fig07_shap_episode_waterfalls(xai_h3)
    plot_fig08_transformer_attention_and_leakage(xai_h3, leakage_df)
    plot_fig09_hybrid_xgb_deepnet(all_metrics_df, run_h3)
    plot_fig10_change_and_policy_shocks(run_h3.predictions_df, event_windows_df)
    plot_fig11_xai_policy_playbook(xai_h3.importance_df, channel_horizon_df)
    plot_fig12_cross_paper_synthesis(feat_df, ablation_df)
    plot_fig13_transformer_explainability(tf_xai)

    print("[6/6] Building 22-slide ECO331 PowerPoint deck (with Speaker Notes) & HTML report...")
    pptx_path = build_course_presentation(run_h3, all_metrics_df, xai_h3, leakage_df, ablation_df=ablation_df)
    html_path = build_self_contained_html_report(
        run_h3, all_metrics_df, adf_df, leakage_df, OUTPUTS_DIR / "interactive_report.html"
    )

    summary_payload = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "raw_months": len(raw_panel),
        "feature_months": len(feat_df),
        "flagship_horizon_months": 3,
        "h3_train_months": len(ds_h3.X_train_full),
        "h3_test_months": len(ds_h3.X_test),
        "h3_leaderboard": run_h3.metrics_df.sort_values("RMSE").to_dict(orient="records"),
        "top_consensus_features": xai_h3.importance_df.head(8).to_dict(orient="records"),
        "episode_surge": {
            k: v for k, v in xai_h3.episode_surge.items() if k != "contributions"
        },
        "episode_disinflation": {
            k: v for k, v in xai_h3.episode_disinflation.items() if k != "contributions"
        },
        "policy_shock_windows_h3": event_windows_df.drop(columns=["Plot_Label"], errors="ignore").to_dict(orient="records"),
        "cross_paper_channel_ablation": ablation_df.drop(columns=["Plot_Label"], errors="ignore").to_dict(orient="records"),
        "horizon_channel_shap_shares": channel_horizon_df.to_dict(orient="records"),
        "xai_policy_playbook": playbook_df.to_dict(orient="records"),
        "cv_leakage_audit": leakage_df.to_dict(orient="records"),
        "presentation_pptx": str(pptx_path),
        "interactive_html": str(html_path),
    }
    (OUTPUTS_DIR / "metrics_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.1f}s! Presentation saved to: {pptx_path}")


if __name__ == "__main__":
    main()
