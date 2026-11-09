# ECO331 Presentation Speaker Guide & Slide-by-Slide Talking Points

> Tip: All notes below are ALSO embedded directly inside the PowerPoint **Presenter View** of `ECO331_Inflation_Forecasting_Presentation.pptx`!

## Slide 1: Title Slide
- **Opening Hook**: *"Today we ask: can modern Machine Learning beat traditional central-bank inflation models — AND still explain WHY in plain economics?"*
- **Key Speaking Points**:
  - We synthesize all four literature clusters from the ECO331 folder: CPI inflation, Electricity prices, Stock markets, and Real GDP.
  - We benchmark 12 models — from Random Walk and Phillips Curves to Elastic Net, XGBoost, LSTM, Transformers, and a Hybrid XGB + Deep Net.
  - Most importantly, we open the black box using exact TreeSHAP and showcase a live interactive web predictor at the end.

## Slide 2: Presentation Roadmap
- **Opening Hook**: *"Our talk follows the exact 13-section ECO331 structure in two clean halves."*
- **Key Speaking Points**:
  - Part I sets up the economic puzzle: why linear Phillips Curves broke during COVID and how we engineered 15 leak-free FRED series.
  - Part II shows our empirical horse-race, proves our models aren't just lagging, translates SHAP into a central-bank playbook, and demos our live web app.

## Slide 3: Introduction
- **Opening Hook**: *"Monetary policy is like steering an oil tanker: today's rate hike only hits inflation two to four quarters later."*
- **Key Speaking Points**:
  - In calm decades, beating a naive Random Walk ('inflation in 3 months equals inflation today') was famously difficult.
  - Between 2020 and 2026, inflation went from 0.3% to 9.0% and back to ~2.5% — creating the ultimate stress test for ML vs. econometrics.
  - Our thesis: ML wins only if it is regularized against small macro sample sizes AND fully explainable to policymakers.

## Slide 4: Motivation from the Course Folder
- **Opening Hook**: *"Instead of picking just one paper from the course folder, we noticed all 18 papers fall into four connected macro channels."*
- **Key Speaking Points**:
  - Cluster 1 is CPI inflation (RBI papers by Singh & Bhoi, Li et al.). Cluster 2 is Electricity prices (Kuskaya & Bilgili 2025).
  - Cluster 3 is Stock markets (Vamossy, Subasi), and Cluster 4 is Real GDP growth (Yoon 2021).
  - We integrate all four into a single unified US inflation framework AND test why hybridizing XGBoost with a Deep Net solves the 585-month macro sample limit.

## Slide 5: Research Objectives
- **Opening Hook**: *"We structured the project around four concrete, falsifiable questions."*
- **Key Speaking Points**:
  - First: which model family wins at 1, 3, 6, and 12 months? Second: are models truly anticipating turning points or just lagging?
  - Third: do Electricity prices, Stock returns, and Real GDP measurably cut forecast error?
  - Fourth: which exact indicators should central bankers watch at each horizon, and how much do they move CPI in percentage points?

## Slide 6: Research Methods & Equations
- **Opening Hook**: *"The single smartest design choice in our math is the Anchored Forecast equation in Box 1."*
- **Key Speaking Points**:
  - Instead of predicting raw CPI levels from scratch, every model predicts how much inflation will CHANGE from today: pi-hat(t+h) = pi(t) + f(X_t).
  - Why does that matter? First, if f=0, we recover the exact Random Walk benchmark. Second, every SHAP value phi_j is literally in percentage points of inflation push or pull!
  - Box 4 shows our Hybrid innovation: a gated neural net that blends XGBoost's tree prediction with smooth 1-month and 3-month SHAP trends.

## Slide 7: Data Source & Zero Look-Ahead Design
- **Opening Hook**: *"We pull 48 years of US data (585 months, 1978–2026) directly from the St. Louis Fed FRED database with zero look-ahead bias."*
- **Key Speaking Points**:
  - Notice two crucial econometric details on the right card: First, raw CPI and GDP have unit roots, so we transform everything into stationary growth rates, spreads, and momentum.
  - Second, Real GDP is reported quarterly with a release delay — Q1 GDP comes out at the end of April. We lag GDP by 3 months so our models only use information a real-time forecaster actually had!
  - And before every test year from 2019 to 2026, we purge h months at the boundary so future inflation realizations never leak into training.

## Slide 8: Data Storytelling I: Regimes & Nonlinearity
- **Opening Hook**: *"If anyone asks WHY we need Machine Learning instead of a textbook OLS Phillips Curve, point straight to Panel B and Panel C."*
- **Key Speaking Points**:
  - In Panel B, each color is a different era. From 1984 to 2019, the unemployment-inflation slope was practically flat. Then in 2021–2022 it jumped vertically.
  - Panel C shows why: upstream Producer Price inflation (PPI) has a curved, nonlinear relationship with future CPI — gentle in normal times, steep above 8–10%.
  - A single straight OLS line averages across all regimes and misses both the calm flat zone and the steep shock zone.

## Slide 9: Data Storytelling II: Correlation Heatmap
- **Opening Hook**: *"This correlation matrix reveals two huge insights before we even run a single ML model."*
- **Key Speaking Points**:
  - First, compare Electricity CPI (+0.64) vs Crude Oil (+0.52) and Unemployment (+0.02) in the first column: sticky utility rates and expectations (+0.87) co-move strongly with future CPI while static unemployment has near-zero unconditional correlation!
  - Second, look at the dark red cluster among CPI, Core CPI, Expectations, Electricity, and PPI — correlations of 0.64 to 0.94. That collinearity is toxic for standard OLS, which is why Ridge and Elastic Net shine.

## Slide 10: Flagship 3-Month Model Comparison
- **Opening Hook**: *"Here is our main 12-model horse race at the 1-quarter (3-month) policy horizon over 2019–2026."*
- **Key Speaking Points**:
  - At the bottom of Panel A, the Random Walk benchmark has an error of 1.004 percentage points, and the textbook OLS Phillips Curve essentially ties it.
  - At the top, Elastic Net (L1+L2) (0.777 pp), Elastic Net (0.777 pp), and XGBoost (0.783 pp) slash forecast error by over 20%!
  - Notice Panel B on the right: our models track both the 2021–22 climb to 9% and the 2023–24 disinflation — which brings up an obvious question on the next slide: are they just lagging?

## Slide 11: Are Models Just Lagging? Change-Space & Policy Shock Audit
- **Opening Hook**: *"When you look at a level chart, every forecast looks slightly shifted to the right during sudden shocks. So we tested this rigorously!"*
- **Key Speaking Points**:
  - In Panel A we subtract today's CPI and plot the 3-month CHANGE. Here, a pure-lag Random Walk is stuck flat at zero forever.
  - Look at two huge policy moments: in Spring 2021 (stimulus + base surge) and late 2022 (peak 75bp Fed rate hikes), Elastic Net and XGBoost swung strongly positive and negative ON TIME, cutting pure-lag error by 28% and 65%!
  - The ONLY window where pure lag won was H1 2022 — because no macro dataset in January 2022 knew Russia would invade Ukraine in February and spike oil to $120.

## Slide 12: Horizons (1–12m) & Macro Regimes
- **Opening Hook**: *"Central banks don't just forecast 3 months ahead — they forecast 1, 3, 6, and 12 months across calm and crisis regimes."*
- **Key Speaking Points**:
  - Panel A plots Relative RMSE below the dotted 1.00 Random Walk line. Every horizon beats Random Walk, led by XGB + Deep Net at 1m and Temporal Transformer at 12m.
  - Panel B shows a great insight from Singh & Bhoi (RBI, 2022): nonlinear trees win during violent supply shocks (col 2), whereas linear shrinkage wins during smooth disinflation (col 3). Combining them hedges regime risk!

## Slide 13: Hybrid XGBoost + Deep Trend Network
- **Opening Hook**: *"A natural question is: why didn't a standalone Transformer crush everything after 48 years of data?"*
- **Key Speaking Points**:
  - 48 years of MONTHLY data is only 585 rows! Deep learning thrives on tens of thousands of examples, while trees handle 500 noisy tabular rows effortlessly.
  - However, tree predictions are flat step-functions that cannot extrapolate smooth slopes. By feeding XGBoost's exact SHAP features and 1m/3m trends into a Gated Deep Network, our Hybrid beats standalone neural nets across every horizon!

## Slide 14: Cross-Paper Synthesis: Electricity, Stock Market & Real GDP
- **Opening Hook**: *"This slide connects ALL FOUR paper topics from our ECO331 Drive folder in a single controlled experiment."*
- **Key Speaking Points**:
  - Look at the orange line in Panel A: while Crude Oil crashed in 2023, regulated Grid Electricity prices (from Kuskaya & Bilgili 2025) stayed high at 4–7%, keeping restaurant, grocery, and housing utility inflation sticky!
  - In Panel B, adding Electricity, US Equities, and causally lagged Real GDP cuts Elastic Net error from 0.825 down to 0.777 pp and XGBoost to 0.770 pp.

## Slide 15: Explainable AI I: Winner-Model Consensus (Elastic Net vs XGBoost vs XGB+MLP)
- **Opening Hook**: *"Now we open the black box across our three winning architectures: Elastic Net, XGBoost, and Hybrid XGB + MLP."*
- **Key Speaking Points**:
  - Mostly they agree — and where they diverge is super revealing! First, all three agree that static Unemployment accounts for under 1% of predictive power.
  - Second, notice Electricity CPI and Real Fed Funds Rate: linear Elastic Net shrinks them to zero because of collinearity, while XGBoost and XGB+MLP give them heavy nonlinear weight. Slide 16 shows WHY.

## Slide 16: Explainable AI II: Global TreeSHAP & Nonlinear Thresholds
- **Opening Hook**: *"This slide explains why XGBoost beats linear models at monetary turning points."*
- **Key Speaking Points**:
  - Look at Panel B: monetary policy does NOT work like a dimmer switch with a constant linear slope — it works like a threshold brake!
  - When Real Fed Funds is negative (-3% to -7% in 2021–22), policy is accommodative. Only when the Fed pushes rates near or above inflation (0% to +2% real rate) does the disinflationary brake bite sharply.

## Slide 17: Local Historical Episode Waterfalls
- **Opening Hook**: *"Imagine briefing a central bank governor in March 2021 or December 2022 — these waterfall charts turn ML numbers into an instant story."*
- **Key Speaking Points**:
  - On the left (2021-01), CPI was still low at 1.37%, yet XGBoost warned of a sharp jump because low pandemic base prices were rolling off while PPI and M2 were booming.
  - On the right (2022-12), CPI was 6.40%, and XGBoost called the drop to ~4.92% by summing base roll-off, tightening real rates, and cooling output.

## Slide 18: Sequence Lag Attention & K-Fold Leakage Audit
- **Opening Hook**: *"Two important methodological takeaways here — one on sequence neural nets, and one warning every student doing time-series ML."*
- **Key Speaking Points**:
  - Panel A shows our Temporal Transformer dynamically shifts its attention onto lags t and t-1 when inflation volatility spikes.
  - Panel B proves Kuskaya & Bilgili's (2025) warning: if you run standard scikit-learn shuffled KFold on CPI, XGBoost looks 47% more accurate than reality because adjacent months leak the answer!

## Slide 19: Temporal Transformer 2D Sequence Explainability (h=12m #1 Winner)
- **Opening Hook**: *"Remember our Temporal Transformer jumped to #1 overall at the 1-year (12-month) horizon (2.384 pp, -15% vs Random Walk) — so how do we explain a Transformer?"*
- **Key Speaking Points**:
  - We combine exact Integrated Gradients (the deep-learning counterpart to SHAP) across all 14 channels x 12 months with explicit self-attention weights.
  - Look at the bottom of Panel A first: Base-Effect CPI drops 5x from 14.6% at 3m down to 2.9% at 12m, while M2 Money Growth (11.8%), Real Fed Funds (7.3%), and Sticky Electricity (doubling to 7.2%) take over!
  - And look at Panel C during Supply Shocks: the Transformer forms a U-shaped 'barbell' attention (10.5% on month t and 10.1% on month t-11), literally comparing today's economy against a year ago!

## Slide 20: Policy Playbook & Macroeconomic Transmission Clock
- **Opening Hook**: *"If a central banker asks 'Okay, which indicator should I actually watch and when?', Panel B is our favorite discovery in the whole project."*
- **Key Speaking Points**:
  - Panel A compares ONLY our three winning models — Elastic Net, XGBoost, and Hybrid XGB+MLP — proving sticky Electricity, PPI, M2, and Real Fed Funds dominate static Unemployment.
  - Panel B is the Macroeconomic Transmission Clock: Supply & Electricity peak at 1m (28.6%), Monetary Policy & GDP Demand peak at 6m, and Inflation Expectations jump 10x to 16.7% at 12m!

## Slide 21: Live Multi-Model Predictor Demo
- **Opening Hook**: *"To make our models tangible rather than static slides, we built a lightweight local web app running all 12 trained models in 20 milliseconds."*
- **Key Speaking Points**:
  - We can click Dec 2022 to watch the models call the post-hike disinflation live, or drag Real Fed Funds and Electricity sliders to watch the exact TreeSHAP bars flip from red to green in real time.

## Slide 22: Conclusions & Limitations
- **Opening Hook**: *"Wrapping up: ML works for macroeconomic forecasting when you respect economic structure, sample size, and causal validation."*
- **Key Speaking Points**:
  - Our four main takeaways are on the left: real turning-point anticipation, cross-paper gains from sticky electricity/stocks/GDP, our Hybrid XGB+Deep Net, and the Horizon Transmission Clock.
  - On the right, we stay honest about limitations: exogenous wars cannot be predicted 3 months early, and scaling standalone Transformers further would require multi-country panel data.

## Slide 23: Resources & References
- **Opening Hook**: *"Everything — data, models, 12 figures, speaker notes, and our live web predictor — reproduces in one command. Thank you, and we'd love your questions!"*
- **Key Speaking Points**:
  - Invite the professor/audience to suggest a 'What-If' macro scenario (e.g., an oil + electricity spike or aggressive Fed rate cuts) to test live in the predictor app!
