# Literature & Folder Audit: `ECO331 2026 / 2. Machine learning topics`

**Google Drive Folder Inspected**: `https://drive.google.com/drive/folders/1iwS56r4OlcEUXsAG-cALVxrOSt6FHyfa`  
**Parent Course Folder**: `ECO331 2026` (`1. Finance and climate change`, `2. Machine learning topics`, `3. Sanctions and Harm`, `4. Current issues proposals`)

---

## 1. Inventory Summary (21 Files Total)

The folder contains **1 mandatory presentation template (`Fomat of PPT.pptx`)** and **20 PDF papers** (**18 unique research papers** plus **2 exact duplicates**). Grouping them by macroeconomic domain reveals four clear clusters:

| Cluster | Unique Papers (Files) | Representative Papers in Folder | Core Models Studied |
| :--- | :---: | :--- | :--- |
| **A. Inflation & CPI Forecasting** *(Dominant Theme)* | **8** *(9 PDFs)* | **#3** Singh & Bhoi (RBI, 2022) *(+ duplicate `2INFLATION.pdf`)*; **#19** Singh & Rathi (RBI, 2024); **#17** Li, Pai & Lin (2022); **#7** Mamedli & Shibitov (2021); **#13** Ülke (2018); **#4** Araujo (2020); **#6** Mirza (2022); **#8** Choudhary (2022) | Random Walk, ARIMA/SARIMA, OLS Phillips Curve (NKPC), Ridge, Lasso, Elastic Net, Random Forest, XGBoost, SVR, ANN/MLP, LSTM, Forecast Combinations |
| **B. Real GDP Growth Forecasting** | **2** *(2 PDFs)* | **#14** Yoon (2021); **#16** Maccarrone et al. (2021) | Gradient Boosting, Random Forest, Linear/AR benchmarks vs. IMF/BOJ institutional forecasts |
| **C. Stock Market Price Prediction** | **5** *(6 PDFs)* | **#2** Subasi & Amir (2021); **#9** Parmar (2020); **#10** Vijh et al. (2020) *(+ duplicate `1-s2.0-S1877050920307924-main.pdf`)*; **#12** Vamossy (2024); **#15** Polamuri (2022); **#18** Kurani (2020) | Linear Regression, Random Forest, ANN, SVM, LSTM, GAN hybrids |
| **D. Sectoral & Commodity Prices** | **2** *(2 PDFs)* | **#1** Soumic Ray (2023, agricultural prices); **#20** Kuşkaya & Bilgili (2025, FRED electricity price index) | ARIMA-LSTM hybrids, XGBoost, Random Forest, Chronological `TimeSeriesSplit` vs. leaky `K-Fold` CV |

---

## 2. Key Methodological Insights Extracted from the Folder

1. **Beating the Random Walk Requires Medium Horizons or Nonlinear Regime Signals** (*Singh & Bhoi 2022; Mamedli & Shibitov 2021; Li et al. 2022*):
   - At $h = 1$ month ahead, CPI inertia is extreme, making univariate benchmarks competitive.
   - At policy-relevant horizons ($h = 3, 6, 12$ months ahead) and across supply-shock regimes, **Regularized Linear (`Ridge`, `Elastic Net`)**, **Gradient Boosted Trees (`XGBoost`)**, and **Cross-Family Forecast Combinations** deliver statistically significant RMSE gains.
2. **Expectations + Upstream Cost-Push Signals Matter** (*Singh & Bhoi 2022; Singh & Rathi 2024*):
   - Structural New Keynesian Phillips Curve (`NKPC`) predictors—survey inflation expectations, labor slack, upstream producer prices (`PPI`), crude oil, monetary rates, and money growth—provide strong economic grounding.
3. **Strict Temporal Validation Prevents Artificial Optimism** (*Kuşkaya & Bilgili 2025*):
   - Applying standard shuffled `K-Fold` cross-validation on autocorrelated macroeconomic price series leaks adjacent months (`t-1` and `t+1`) into training folds, fabricating artificially low validation RMSE. Chronological `TimeSeriesSplit` with an explicit **$h$-month target embargo** is essential.
4. **Course Presentation Template (`Fomat of PPT.pptx`, 13 Required Sections)**:
   - `1. Title` → `2. Contents` → `3. Introduction` → `4. Motivation` → `5. Objectives` → `6. Research methods (with numerical expressions)` → `7. Data source` → `8. Graphical presentation of data (rolling average, heatmap, scatter plot)` → `9. Empirical findings` → `10. Conclusions` → `11. Limitations` → `12. Resources (Data, Code, GitHub)` → `13. References`.

---

## 3. Why We Built Explainable Macroeconomic CPI Inflation Forecasting

Inflation forecasting represents **nearly half of the papers in the course folder**, connects directly to macroeconomic theory (`ECO331`), naturally benchmarks **Linear Models vs. Tree Ensembles (`XGBoost`/`RF`) vs. Deep Sequence Nets (`MLP`/`LSTM`/`Temporal Transformer`)**, and allows rich **Explainable AI (`TreeSHAP`, standardized `Elastic Net` coefficients, local historical episode waterfalls, and `Transformer` lag attention)** so every prediction translates into clear economic intuition.
