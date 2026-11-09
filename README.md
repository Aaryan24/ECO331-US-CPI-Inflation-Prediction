# ECO331 Course Project: US CPI Inflation Prediction

Forecasting US consumer-price inflation with traditional economic models and machine learning, and explaining the indicators behind the predictions.

The project compares **12 models** at **1, 3, 6 and 12 months ahead**, using **15 economic series from FRED**. The included dataset spans **1978–2026**, with forecast evaluation on **2019–2026**.

## Start here

- **[Final presentation](presentation/ECO331_Inflation_Forecasting_Revised.pptx)** — the latest manually edited PowerPoint. Download it to view the slides and presenter notes.
- **[Research notebook](notebooks/ECO331_Explainable_Inflation_Forecasting.ipynb)** — the analysis walkthrough.
- **[Results dashboard](outputs/interactive_report.html)** — download and open locally for a standalone report with embedded charts.
- **[Results tables](outputs/tables/)** and **[figures](outputs/figures/)** — the underlying comparisons and visual evidence.
- **[Literature review](docs/drive_literature_review.md)** — course readings and the motivation for the project.

## Research questions

1. Which models predict inflation most accurately at each horizon?
2. Can they anticipate rises and falls, rather than simply follow recent inflation?
3. Do electricity prices, stock returns and GDP improve the forecasts?
4. Which indicators explain a prediction, and how does their importance change with the horizon?

## Main results

The saved three-month evaluation reports:

| Model | RMSE (percentage points) | Reduction versus no change |
| --- | ---: | ---: |
| Elastic Net | 0.777 | 22.7% |
| XGBoost | 0.783 | 22.1% |
| Forecast Combination | 0.793 | 21.0% |
| XGBoost + Deep Net | 0.829 | 17.4% |
| Random Walk (no-change forecast) | 1.004 | — |

RMSE measures forecast error and gives larger mistakes more weight. Lower is better. Full metrics, including statistical comparisons, are in [the three-month results](outputs/tables/metrics_h3_flagship.csv).

The best model varies by horizon: the hybrid leads at one month, Elastic Net at three months, the autoregressive benchmark at six months, and the Transformer at twelve months. Adding electricity prices, equities and GDP together reduces Elastic Net's three-month error from 0.825 to 0.777 percentage points in the saved comparison.

These results describe the included experiment. They do not establish the causal effects of changing interest rates or other inputs.

## Run the interactive predictor

Install [uv](https://docs.astral.sh/uv/) and clone this repository. From the project directory:

```bash
uv sync --locked
uv run python app.py --port 8000
```

Open **http://127.0.0.1:8000**. The app compares model predictions across horizons and explains XGBoost's forecast contributions. A saved model bundle is included in `outputs/demo_predictor_bundle.pkl`; the app trains a bundle if none is present.

Python requirements and dependencies are recorded in `pyproject.toml`, `.python-version` and `uv.lock`.

## Reproduce the analysis

```bash
uv run python run_pipeline.py
uv run python -m src.build_notebook
```

The pipeline builds the features, evaluates models, and regenerates figures, tables, the HTML report and the **original generated presentation**. The notebook command rebuilds the notebook.

**The manually edited final deck is a separate deliverable.** Keep `presentation/ECO331_Inflation_Forecasting_Revised.pptx` as the presentation of record. The generator does not reproduce its manual edits. The separate Markdown speaker guides predate those manual edits and may differ from the final slides.

## Project structure

```text
.
├── app.py                    # Local interactive predictor server
├── run_pipeline.py           # End-to-end research pipeline
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── src/
│   ├── data_loader.py        # FRED downloads and cached data
│   ├── features.py           # Feature preparation and forecast targets
│   ├── models.py             # Model training and evaluation
│   ├── explainability.py     # Forecast explanations and comparison experiments
│   ├── visualize.py          # Research figures
│   ├── predictor_service.py  # Saved models and interactive inference
│   ├── build_presentation.py # Original PowerPoint generator
│   ├── build_notebook.py     # Notebook generation
│   └── web/                  # Demo HTML, CSS and JavaScript
├── data/
│   ├── raw/                  # Cached FRED series
│   └── processed/            # Prepared feature dataset
├── notebooks/                # Research notebook
├── outputs/
│   ├── figures/              # Research charts
│   ├── tables/               # CSV results and diagnostics
│   ├── metrics_summary.json  # Machine-readable summary
│   ├── interactive_report.html
│   └── demo_predictor_bundle.pkl
├── presentation/             # Final deck, previous drafts and speaker guides
└── docs/                     # Literature review
```

## Methods and limitations

Models include Random Walk, autoregression, a Phillips-curve regression, Ridge, Elastic Net, Random Forest, XGBoost, MLP, LSTM, a Transformer, a tree–neural hybrid and a forecast combination. Explanations include SHAP, linear coefficients and neural-network attribution methods.

Training and evaluation follow time order. The pipeline separates training targets from later test periods and delays GDP inputs to account for publication. However, some historical series include subsequent revisions, so this is not a complete real-time vintage-data evaluation. The monthly sample is limited, unexpected shocks can cause large errors, and model explanations should not be interpreted as causal policy estimates.

## Data and references

Economic data come from the [Federal Reserve Bank of St. Louis FRED database](https://fred.stlouisfed.org/). Series definitions are recorded in the [data dictionary](outputs/tables/data_dictionary_and_adf.csv). The [literature review](docs/drive_literature_review.md) documents the course papers on inflation, electricity prices, equities and GDP forecasting. Cached source data retain their original providers' terms.
