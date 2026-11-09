"""
Generate and execute an interactive, presentation-ready Jupyter Notebook walkthrough.
"""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

from src import NOTEBOOKS_DIR, PROJECT_ROOT


def build_and_execute_notebook() -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb_path = NOTEBOOKS_DIR / "ECO331_Explainable_Inflation_Forecasting.ipynb"

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# ECO331 (2026): Explainable Macroeconomic Inflation Forecasting\n"
            "**Comparing Econometric Benchmarks, Regularized Linear Models, Gradient Boosted Trees (XGBoost), "
            "and Deep Sequence Architectures (MLP, LSTM, Temporal Transformer)**\n\n"
            "This notebook walks end-to-end through:\n"
            "1. **FRED Macroeconomic Data & Stationarity Checks** (`1978–2026`, 12 monthly channels)\n"
            "2. **Storytelling Exploratory Data Analysis** (Regimes, Phillips Curve shifts, Cost-Push nonlinearity, Correlation Matrix)\n"
            "3. **Multi-Horizon & Regime Model Comparison** ($h \\in \\{1, 3, 6, 12\\}$ months ahead, Diebold-Mariano HAC tests)\n"
            "4. **Explainable AI (XAI)**: Standardized Elastic Net $\\beta$, Exact XGBoost `TreeSHAP`, Local Historical Episode Waterfalls, and `Temporal Transformer` Lag Attention + CV Leakage Audit."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import os, sys\n"
            "os.environ.setdefault('OMP_NUM_THREADS', '1')\n"
            "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src import FIGURES_DIR, TABLES_DIR\n"
            "pd.set_option('display.max_columns', 20)\n"
            "pd.set_option('display.width', 140)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Macroeconomic Data Dictionary & Unit Root (ADF) Stationarity\n"
            "Raw price and output indices (`CPIAUCSL`, `CPILFESL`, `PPIACO`, `INDPRO`, `M2SL`) are non-stationary $I(1)$ series. "
            "Transforming them into year-over-year percentage rates, annualized momentum, and spreads yields stationary predictors."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "adf_df = pd.read_csv(TABLES_DIR / 'data_dictionary_and_adf.csv')\n"
            "adf_df[['FRED_ID', 'Variable_Name', 'Economic_Channel', 'Raw_ADF_p', 'Model_Transformation', 'Transformed_ADF_p']]"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Exploratory Data Storytelling (Regimes, Nonlinearity & Correlations)\n"
            "Notice in **Panel B** below how the empirical Phillips Curve slope shifts dramatically across stagflation, "
            "the Great Moderation, and the 2020–2022 supply-shock surge—motivating nonlinear ML and regime-aware ensembles."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES_DIR / 'fig01_eda_storytelling.png'), width=950))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig02_correlation_heatmap.png'), width=760))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Out-of-Sample Model Comparison ($h = 3$ Months Ahead, 2019–2026)\n"
            "All 11 specifications are trained strictly on pre-2019 observations (with an explicit $h$-month embargo so zero 2019+ targets leak into training) "
            "using the anchored decomposition:\n"
            "$$\\hat{\\pi}_{t+h} = \\pi_t + f_\\theta(X_t)$$"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "h3_df = pd.read_csv(TABLES_DIR / 'metrics_h3_flagship.csv')\n"
            "h3_df.style.format({\n"
            "    'RMSE': '{:.3f}', 'MAE': '{:.3f}', 'Rel_RMSE_vs_RW': '{:.3f}',\n"
            "    'R2': '{:.3f}', 'Directional_Acc_pct': '{:.1f}%',\n"
            "    'DM_Stat_vs_RW': '{:.2f}', 'DM_p_value': '{:.4f}'\n"
            "})"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES_DIR / 'fig03_model_comparison_h3.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig10_change_and_policy_shocks.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig04_multi_horizon_and_regimes.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig09_hybrid_xgb_deepnet.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig12_cross_paper_synthesis.png'), width=960))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### 3B. Policy Shock Audit & 4-Cluster Course Literature Ablation ($h = 3$m)\n"
            "Evaluating forecast errors across 5 policy/shock windows AND controlled ablation adding Electricity (`CUSR0000SEHF01`), Equities (`SPASTT01USM661N`), and 1Q-lagged Real GDP (`GDPC1`):"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(pd.read_csv(TABLES_DIR / 'policy_shock_windows_h3.csv').drop(columns=['Plot_Label'], errors='ignore'))\n"
            "pd.read_csv(TABLES_DIR / 'cross_paper_channel_ablation.csv').drop(columns=['Plot_Label'], errors='ignore')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Multi-Horizon Summary Table ($h \\in \\{1, 3, 6, 12\\}$ Months)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "all_h = pd.read_csv(TABLES_DIR / 'metrics_all_horizons.csv')\n"
            "piv_rmse = all_h.pivot(index='Model', columns='Horizon_Months', values='RMSE')\n"
            "piv_rmse.columns = [f'RMSE h={c}m' for c in piv_rmse.columns]\n"
            "piv_rmse.sort_values('RMSE h=3m').round(3)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Explainable AI (XAI): Opening the Black Box\n"
            "We inspect **four complementary explainability lenses**:\n"
            "1. **Top-Model Driver Consensus** (`Elastic Net` signed $\\beta$ vs. `XGBoost` exact `TreeSHAP` vs. `XGB + MLP` Hybrid attribution)\n"
            "2. **Global Nonlinear SHAP Summary & Threshold Curves**\n"
            "3. **Local Case-Study Episode Waterfalls** (Inflation Surge vs. Disinflation Turning Point)\n"
            "4. **Temporal Transformer 2D Sequence Explainability (`h = 12m` #1 Winner) & Random K-Fold Leakage Audit**"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "imp_df = pd.read_csv(TABLES_DIR / 'explainability_feature_importance.csv')\n"
            "imp_df[['Feature_Label', 'ElasticNet_Signed_Coef', 'ElasticNet_Share_pct', 'XGBoost_Mean_Abs_SHAP_pp', 'XGBoost_Share_pct', 'XGB_MLP_Share_pct']].head(10).round(3)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES_DIR / 'fig05_xai_cross_family_importance.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig06_shap_global_nonlinear.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig07_shap_episode_waterfalls.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig08_transformer_attention_and_leakage.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig13_transformer_explainability.png'), width=960))\n"
            "display(Image(filename=str(FIGURES_DIR / 'fig11_xai_policy_playbook.png'), width=960))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### 5B. Actionable Policy Playbook & Macroeconomic Transmission Clock Across Horizons"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(pd.read_csv(TABLES_DIR / 'horizon_channel_shap_shares.csv'))\n"
            "pd.read_csv(TABLES_DIR / 'xai_policy_playbook.csv')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. Methodological Check: Why Random K-Fold Leaks Future Inflation"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "pd.read_csv(TABLES_DIR / 'cv_leakage_diagnostic.csv')"
        )
    )

    nb["cells"] = cells
    nbf.write(nb, str(nb_path))

    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    nbf.write(nb, str(nb_path))
    return nb_path


if __name__ == "__main__":
    out = build_and_execute_notebook()
    print(f"Executed notebook saved to: {out}")
