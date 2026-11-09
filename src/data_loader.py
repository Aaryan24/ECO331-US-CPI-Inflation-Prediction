"""
Data acquisition, local caching, causal gap handling, and ADF stationarity checks.

Downloads 12 core monthly US macroeconomic series from FRED (Federal Reserve
Economic Data, St. Louis Fed) spanning 1978-01 to 2026-09 without requiring API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import adfuller

from src import DATA_RAW_DIR, TABLES_DIR


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    name: str
    channel: str
    units: str
    stationary_transform: str


FRED_SERIES: List[SeriesSpec] = [
    SeriesSpec(
        "CPIAUCSL",
        "Headline CPI (All Urban Consumers)",
        "Target / Price Inertia",
        "Index (1982-84=100, SA)",
        "YoY % Change & 3m Annualized %",
    ),
    SeriesSpec(
        "CPILFESL",
        "Core CPI (Less Food & Energy)",
        "Underlying Trend Inflation",
        "Index (1982-84=100, SA)",
        "YoY % Change & Core-Headline Spread",
    ),
    SeriesSpec(
        "MICH",
        "Univ. of Michigan 1Y Inflation Expectations",
        "Expectations Channel (NKPC)",
        "Percent (%)",
        "Level (%) & 3m Change",
    ),
    SeriesSpec(
        "UNRATE",
        "Civilian Unemployment Rate",
        "Phillips Curve / Labor Slack",
        "Percent (%, SA)",
        "Level (%) & 12m Change (Sahm-style)",
    ),
    SeriesSpec(
        "INDPRO",
        "Industrial Production Index",
        "Real Activity / Output Growth",
        "Index (2017=100, SA)",
        "YoY % Growth",
    ),
    SeriesSpec(
        "PPIACO",
        "Producer Price Index (All Commodities)",
        "Upstream Cost-Push Supply Shock",
        "Index (1982=100, NSA)",
        "YoY % Change",
    ),
    SeriesSpec(
        "WTISPLC",
        "Spot Crude Oil Price (WTI)",
        "Energy / Commodity Shock",
        "USD per Barrel",
        "YoY % Change",
    ),
    SeriesSpec(
        "FEDFUNDS",
        "Effective Federal Funds Rate",
        "Monetary Policy Stance",
        "Percent (%)",
        "Level (%), Real Policy Rate, 12m Change",
    ),
    SeriesSpec(
        "GS10",
        "10-Year Treasury Constant Maturity Rate",
        "Long-Term Bond Market Yields",
        "Percent (%)",
        "10Y-2Y Term Spread (%)",
    ),
    SeriesSpec(
        "GS2",
        "2-Year Treasury Constant Maturity Rate",
        "Near-Term Policy Path Expectations",
        "Percent (%)",
        "10Y-2Y Term Spread (%)",
    ),
    SeriesSpec(
        "M2SL",
        "M2 Broad Money Stock",
        "Monetarist / Liquidity Channel",
        "Billions of USD (SA)",
        "YoY % Growth",
    ),
    SeriesSpec(
        "UMCSENT",
        "Univ. of Michigan Consumer Sentiment",
        "Household Demand Confidence",
        "Index (1966:Q1=100)",
        "12m Change & Normalized Level",
    ),
    SeriesSpec(
        "CUSR0000SEHF01",
        "CPI: Electricity Price Index",
        "Sticky Grid Energy Pass-Through (Kuskaya 2025)",
        "Index (1982-84=100, SA)",
        "YoY % Change",
    ),
    SeriesSpec(
        "SPASTT01USM661N",
        "US Total Equity / Stock Price Index",
        "Financial Conditions & Wealth (Vamossy 2024)",
        "Index (2015=100, OECD)",
        "YoY % Return",
    ),
    SeriesSpec(
        "GDPC1",
        "Real Gross Domestic Product",
        "Aggregate Output Growth (Yoon 2021)",
        "Billions of Chained 2017 USD (Quarterly)",
        "YoY % Growth (1Q Causal Publication Lag)",
    ),
]

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
START_DATE = "1978-01-01"
END_DATE = "2026-09-01"


def fetch_single_fred_series(series_id: str, cache_dir: Path = DATA_RAW_DIR, force_refresh: bool = False) -> pd.Series:
    """Download a single monthly FRED CSV series with local disk caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
    else:
        url = FRED_CSV_URL.format(series_id=series_id)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        df = pd.read_csv(StringIO(resp.text))

    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].sort_index()
    s.name = series_id
    return s


def load_macro_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Assemble the monthly US macroeconomic panel (1978-01 to 2026-09).

    Applies strictly causal imputation (using only t-1 and t-2) if an isolated
    single-month survey release gap occurs (specifically 2025-10 BLS release gap),
    avoiding any look-ahead bias from t+1.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    monthly_index = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    panel = pd.DataFrame(index=monthly_index)
    panel.index.name = "date"

    for spec in FRED_SERIES:
        s = fetch_single_fred_series(spec.series_id, force_refresh=force_refresh)
        if spec.series_id == "GDPC1":
            # BEA Advance GDP for Q1 (dated Jan 1) is published late April -> shift +3m for zero look-ahead
            s_lagged = s.copy()
            s_lagged.index = s_lagged.index + pd.DateOffset(months=3)
            panel[spec.series_id] = s_lagged.reindex(monthly_index).ffill()
        else:
            panel[spec.series_id] = s.reindex(monthly_index)

    panel["is_imputed_gap"] = panel[["CPIAUCSL", "CPILFESL", "UNRATE"]].isna().any(axis=1)

    # Strictly causal imputation (zero look-ahead from future t+1 observations)
    price_index_cols = [
        "CPIAUCSL",
        "CPILFESL",
        "INDPRO",
        "PPIACO",
        "WTISPLC",
        "M2SL",
        "CUSR0000SEHF01",
        "SPASTT01USM661N",
        "GDPC1",
    ]
    for col in panel.columns:
        if col == "is_imputed_gap":
            continue
        missing_dates = panel.index[panel[col].isna()]
        for dt in missing_dates:
            loc = panel.index.get_loc(dt)
            if col in price_index_cols and loc >= 2:
                prev_1 = panel[col].iloc[loc - 1]
                prev_2 = panel[col].iloc[loc - 2]
                panel.iloc[loc, panel.columns.get_loc(col)] = prev_1 * (prev_1 / prev_2)
            elif loc >= 1:
                panel.iloc[loc, panel.columns.get_loc(col)] = panel[col].iloc[loc - 1]

    combined_cache = DATA_RAW_DIR / "fred_macro_panel_1978_2026.csv"
    panel.to_csv(combined_cache)
    return panel


def run_adf_test(series: pd.Series) -> Dict[str, float]:
    """Run Augmented Dickey-Fuller unit root test with AIC lag selection."""
    clean = series.dropna()
    try:
        stat, pval, usedlag, nobs, _, _ = adfuller(clean, autolag="AIC", result_object=False)
    except TypeError:
        stat, pval, usedlag, nobs, _, _ = adfuller(clean, autolag="AIC")
    return {
        "adf_stat": float(stat),
        "p_value": float(pval),
        "lags": int(usedlag),
        "n_obs": int(nobs),
    }


def generate_data_dictionary_and_adf(panel: pd.DataFrame, transformed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a publication-ready summary & ADF stationarity table comparing raw FRED
    levels against the stationary macroeconomic transformations used in modeling.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    transform_map = {
        "CPIAUCSL": "cpi_yoy",
        "CPILFESL": "core_cpi_yoy",
        "MICH": "mich_exp_1y",
        "UNRATE": "unrate_chg_12m",
        "INDPRO": "indpro_yoy",
        "PPIACO": "ppi_yoy",
        "WTISPLC": "oil_yoy",
        "FEDFUNDS": "fedfunds_chg_12m",
        "GS10": "yield_spread_10y2y",
        "GS2": "yield_spread_10y2y",
        "M2SL": "m2_yoy",
        "UMCSENT": "umcsent_chg_12m",
        "CUSR0000SEHF01": "elec_yoy",
        "SPASTT01USM661N": "stock_yoy",
        "GDPC1": "gdp_yoy",
    }

    rows = []
    for spec in FRED_SERIES:
        raw_s = fetch_single_fred_series(spec.series_id, force_refresh=False)
        raw_s = raw_s[(raw_s.index >= pd.Timestamp(START_DATE)) & (raw_s.index <= pd.Timestamp(END_DATE))]
        raw_adf = run_adf_test(raw_s)
        t_col = transform_map[spec.series_id]
        trans_adf = run_adf_test(transformed_df[t_col])

        rows.append(
            {
                "FRED_ID": spec.series_id,
                "Variable_Name": spec.name,
                "Economic_Channel": spec.channel,
                "Units": spec.units,
                "Raw_Mean": round(float(raw_s.mean()), 2),
                "Raw_Std": round(float(raw_s.std()), 2),
                "Raw_ADF_Stat": round(raw_adf["adf_stat"], 2),
                "Raw_ADF_p": round(raw_adf["p_value"], 4),
                "Raw_Stationary_5pct": "Yes" if raw_adf["p_value"] < 0.05 else "No (Unit Root)",
                "Model_Transformation": spec.stationary_transform,
                "Transformed_ADF_Stat": round(trans_adf["adf_stat"], 2),
                "Transformed_ADF_p": round(trans_adf["p_value"], 4),
                "Transformed_Stationary_5pct": "Yes" if trans_adf["p_value"] < 0.05 else "Borderline",
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLES_DIR / "data_dictionary_and_adf.csv", index=False)
    return out_df
