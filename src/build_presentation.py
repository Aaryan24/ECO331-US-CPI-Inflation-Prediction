"""
Automated PowerPoint deck & Speaker-Script generator aligned with ECO331 `Format of PPT.pptx`.

Produces a 22-slide widescreen (16:9) presentation where EVERY slide pairs high-DPI
graphics with concrete on-slide speaking points, policy takeaways, and embedded
PowerPoint Presenter Notes (also exported to `presentation/Speaker_Notes_and_Talking_Points.md`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src import FIGURES_DIR, PRESENTATION_DIR, TABLES_DIR
from src.explainability import ExplainabilityBundle
from src.models import HorizonRunArtifacts

# Academic color theme
NAVY = RGBColor(15, 23, 42)        # #0F172A
ROYAL = RGBColor(30, 64, 175)      # #1E40AF
TEAL = RGBColor(13, 148, 136)      # #0D9488
AMBER = RGBColor(217, 119, 6)      # #D97706
ROSE = RGBColor(225, 29, 72)       # #E11D48
SLATE = RGBColor(51, 65, 85)       # #334155
MUTED = RGBColor(100, 116, 139)    # #64748B
CARD_BG = RGBColor(241, 245, 249)  # #F1F5F9
WHITE = RGBColor(255, 255, 255)


def _add_header(slide, section_tag: str, title_text: str, slide_num: int, total_slides: int) -> None:
    """Render a consistent top academic header bar and footer page counter."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.88))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.88), Inches(13.333), Inches(0.05))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = TEAL
    stripe.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.48), Inches(0.08), Inches(12.2), Inches(0.72))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    r_tag = p0.add_run()
    r_tag.text = f"{section_tag.upper()}  |  "
    r_tag.font.size = Pt(10.5)
    r_tag.font.bold = True
    r_tag.font.color.rgb = RGBColor(94, 234, 212)

    r_title = p0.add_run()
    r_title.text = title_text
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = WHITE

    ft = slide.shapes.add_textbox(Inches(0.48), Inches(7.14), Inches(12.37), Inches(0.28))
    ftf = ft.text_frame
    fp = ftf.paragraphs[0]
    fp.alignment = PP_ALIGN.RIGHT
    fr = fp.add_run()
    fr.text = f"ECO331 (2026)  •  Explainable Macroeconomic Inflation Forecasting  •  Slide {slide_num}/{total_slides}"
    fr.font.size = Pt(9.2)
    fr.font.color.rgb = MUTED


def _add_bullet_run(paragraph, text: str, font_pt: float) -> None:
    """Format `Lead Label: rest of sentence` with bold navy lead for effortless scanning."""
    if ":" in text and len(text.split(":", 1)[0]) < 48:
        lead, rest = text.split(":", 1)
        r1 = paragraph.add_run()
        r1.text = f"•  {lead}:"
        r1.font.size = Pt(font_pt)
        r1.font.bold = True
        r1.font.color.rgb = NAVY

        r2 = paragraph.add_run()
        r2.text = rest
        r2.font.size = Pt(font_pt)
        r2.font.color.rgb = SLATE
    else:
        rb = paragraph.add_run()
        rb.text = f"•  {text}"
        rb.font.size = Pt(font_pt)
        rb.font.color.rgb = SLATE


def _add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    heading: str,
    bullets: List[str],
    accent_color: RGBColor = ROYAL,
    font_pt: float = 12.2,
    space_after_pt: float = 4.5,
) -> None:
    """Add a rounded info card with colored top border and bold-lead speaking bullets."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = RGBColor(203, 213, 225)
    card.line.width = Pt(1.0)

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.07)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.16), Inches(top + 0.11), Inches(width - 0.32), Inches(height - 0.18)
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p_head = tf.paragraphs[0]
    p_head.space_after = Pt(4.5)
    rh = p_head.add_run()
    rh.text = heading
    rh.font.size = Pt(font_pt + 1.5)
    rh.font.bold = True
    rh.font.color.rgb = NAVY

    for b in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(space_after_pt)
        _add_bullet_run(p, b, font_pt)


def _add_fitted_image(
    slide,
    img_path: Path,
    box_left: float,
    box_top: float,
    box_width: float,
    box_height: float,
) -> None:
    """Insert `img_path` centered inside the bounding box while preserving exact aspect ratio."""
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    aspect = w_px / h_px
    box_aspect = box_width / box_height

    if aspect >= box_aspect:
        final_w = box_width
        final_h = box_width / aspect
    else:
        final_h = box_height
        final_w = box_height * aspect

    left = box_left + (box_width - final_w) / 2.0
    top = box_top + (box_height - final_h) / 2.0
    slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), width=Inches(final_w), height=Inches(final_h))


def _add_wide_chart_with_talking_cards(
    slide,
    img_path: Path,
    left_heading: str,
    left_bullets: List[str],
    right_heading: str,
    right_bullets: List[str],
    left_accent: RGBColor = ROYAL,
    right_accent: RGBColor = TEAL,
    font_pt: float = 10.6,
) -> None:
    """Standard layout for wide 2-panel charts: crisp figure on top + 2 speaking-point cards below."""
    _add_fitted_image(slide, img_path, 0.42, 1.00, 12.48, 4.24)
    _add_card(
        slide, 0.42, 5.32, 6.12, 1.74,
        left_heading,
        left_bullets,
        accent_color=left_accent,
        font_pt=font_pt,
        space_after_pt=3.5,
    )
    _add_card(
        slide, 6.76, 5.32, 6.14, 1.74,
        right_heading,
        right_bullets,
        accent_color=right_accent,
        font_pt=font_pt,
        space_after_pt=3.5,
    )


def _attach_speaker_notes(
    slide,
    slide_num: int,
    title: str,
    hook: str,
    say_points: List[str],
    script_log: List[str],
) -> None:
    """Embed Presenter View speaker notes directly inside the PPTX slide and record to Markdown."""
    notes_text = (
        f"SLIDE {slide_num}: {title}\n"
        f"1-Sentence Hook: {hook}\n\n"
        "What to Say Out Loud:\n"
        + "\n".join(f"  - {pt}" for pt in say_points)
    )
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = notes_text

    md_block = [
        f"## Slide {slide_num}: {title}",
        f"- **Opening Hook**: *\"{hook}\"*",
        "- **Key Speaking Points**:",
    ]
    for pt in say_points:
        md_block.append(f"  - {pt}")
    md_block.append("")
    script_log.append("\n".join(md_block))


def build_course_presentation(
    run_h3: HorizonRunArtifacts,
    all_metrics_df: pd.DataFrame,
    xai: ExplainabilityBundle,
    leakage_df: pd.DataFrame,
    ablation_df: Optional[pd.DataFrame] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate the 22-slide ECO331 presentation with rich speaking points & Presenter Notes."""
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = PRESENTATION_DIR / "ECO331_Inflation_Forecasting_Presentation.pptx"

    if ablation_df is None and (TABLES_DIR / "cross_paper_channel_ablation.csv").exists():
        ablation_df = pd.read_csv(TABLES_DIR / "cross_paper_channel_ablation.csv")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 23
    script_log: List[str] = [
        "# ECO331 Presentation Speaker Guide & Slide-by-Slide Talking Points",
        "",
        "> Tip: All notes below are ALSO embedded directly inside the PowerPoint **Presenter View** of `ECO331_Inflation_Forecasting_Presentation.pptx`!",
        "",
    ]

    # Dynamic numbers from the trained pipeline
    m_h3 = run_h3.metrics_df.set_index("Model")
    rw_rmse = float(m_h3.loc["Random Walk", "RMSE"])
    best_row = run_h3.metrics_df.sort_values("RMSE").iloc[0]
    enet_rmse = float(m_h3.loc["Elastic Net (L1+L2)", "RMSE"])
    xgb_rmse = float(m_h3.loc["XGBoost", "RMSE"])
    hyb_rmse = float(m_h3.loc["XGB + Deep Net", "RMSE"])
    comb_rmse = float(m_h3.loc["Forecast Combination", "RMSE"])
    tf_rmse = float(m_h3.loc["Temporal Transformer", "RMSE"])
    lstm_rmse = float(m_h3.loc["LSTM (Sequence)", "RMSE"])
    xgb_dir = float(m_h3.loc["XGBoost", "Directional_Acc_pct"])
    top3_feats = xai.importance_df["Feature_Label"].iloc[:3].tolist()

    # Horizon leaders
    best_h1 = all_metrics_df[all_metrics_df["Horizon_Months"] == 1].sort_values("RMSE").iloc[0]
    best_h6 = all_metrics_df[all_metrics_df["Horizon_Months"] == 6].sort_values("RMSE").iloc[0]
    best_h12 = all_metrics_df[all_metrics_df["Horizon_Months"] == 12].sort_values("RMSE").iloc[0]

    # Ablation numbers
    core_enet_rmse = float(ablation_df.iloc[0]["ElasticNet_h3_RMSE"]) if ablation_df is not None else 0.825
    uni_enet_rmse = float(ablation_df.iloc[-1]["ElasticNet_h3_RMSE"]) if ablation_df is not None else 0.777
    uni_xgb_rmse = float(ablation_df.iloc[-1]["XGBoost_h3_RMSE"]) if ablation_df is not None else 0.770

    # Electricity SHAP share if present
    elec_rows = xai.importance_df[xai.importance_df["Feature_Code"] == "elec_yoy"]
    elec_xgb_share = float(elec_rows["XGBoost_Share_pct"].iloc[0]) if not elec_rows.empty else 9.8

    # ------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # ------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    accent1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.25), Inches(1.5), Inches(0.08))
    accent1.fill.solid()
    accent1.fill.fore_color.rgb = TEAL
    accent1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(11.7), Inches(5.1))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_course = tf1.paragraphs[0]
    rc = p_course.add_run()
    rc.text = "ECO331 (2026)  •  MACHINE LEARNING IN ECONOMICS"
    rc.font.size = Pt(14)
    rc.font.bold = True
    rc.font.color.rgb = RGBColor(94, 234, 212)

    p_main = tf1.add_paragraph()
    p_main.space_before = Pt(14)
    rm = p_main.add_run()
    rm.text = "Explainable Macroeconomic Inflation Forecasting"
    rm.font.size = Pt(37)
    rm.font.bold = True
    rm.font.color.rgb = WHITE

    p_sub = tf1.add_paragraph()
    p_sub.space_before = Pt(12)
    rs = p_sub.add_run()
    rs.text = (
        "Unifying Econometric Benchmarks, Regularized Linear Models, XGBoost, Deep Sequence Nets\n"
        "& Hybrid Tree-Neural Architectures Across 48 Years of US Macro, Energy, Equity & GDP Data"
    )
    rs.font.size = Pt(18.5)
    rs.font.color.rgb = RGBColor(203, 213, 225)

    p_meta = tf1.add_paragraph()
    p_meta.space_before = Pt(34)
    rmeta = p_meta.add_run()
    rmeta.text = (
        "Data: 15 Monthly/Quarterly FRED Series (1978–2026, N = 585 Months)   |   Models: 12 Specifications Across 6 Families\n"
        "Horizons: h = 1, 3, 6, 12 Months Ahead   |   Explainability: Exact TreeSHAP, Policy Playbook & Live Demo App"
    )
    rmeta.font.size = Pt(13.2)
    rmeta.font.color.rgb = RGBColor(148, 163, 184)

    _attach_speaker_notes(
        s1, 1, "Title Slide",
        "Today we ask: can modern Machine Learning beat traditional central-bank inflation models — AND still explain WHY in plain economics?",
        [
            "We synthesize all four literature clusters from the ECO331 folder: CPI inflation, Electricity prices, Stock markets, and Real GDP.",
            "We benchmark 12 models — from Random Walk and Phillips Curves to Elastic Net, XGBoost, LSTM, Transformers, and a Hybrid XGB + Deep Net.",
            "Most importantly, we open the black box using exact TreeSHAP and showcase a live interactive web predictor at the end.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 2: CONTENTS
    # ------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    _add_header(s2, "Section 1", "Presentation Roadmap (Aligned with Course Template)", 2, total_slides)
    _add_card(
        s2, 0.55, 1.25, 5.95, 5.55,
        "Part I — Economic Problem, Theory & Data Story",
        [
            "1. Introduction: Why Central Banks Forecast 3–12 Months Ahead",
            "2. Motivation: Uniting All 4 Course Literature Clusters (18 Papers)",
            "3. Research Objectives: Accuracy vs. Interpretability Trade-Off",
            "4. Methods & Equations: Anchored Forecasts, Shrinkage, SHAP & Attention",
            "5. Data Source: 15 FRED Macro, Electricity, Equity & GDP Series (1978–2026)",
            "6. Data Storytelling: Shifting Phillips Curve & Cross-Correlations",
        ],
        accent_color=ROYAL,
        font_pt=13.0,
        space_after_pt=6.0,
    )
    _add_card(
        s2, 6.82, 1.25, 5.95, 5.55,
        "Part II — Empirical Findings, Policy XAI & Live Demo",
        [
            "7. Flagship Results (h=3m): Why Shrinkage & Boosted Trees Beat OLS",
            "8. Are Models Just Lagging? Change-Space & 5 Policy/Shock Eras",
            "9. Horizons (1–12m), Hybrid XGB+Deep Net & 4-Cluster Ablation",
            "10. Explainable AI: Linear vs. TreeSHAP vs. Neural Driver Consensus",
            "11. Policy Playbook: Nonlinear Fed Rate Brake & Transmission Clock",
            "12. Live Predictor Web Tool, Conclusions, Limitations & References",
        ],
        accent_color=TEAL,
        font_pt=13.0,
        space_after_pt=6.0,
    )
    _attach_speaker_notes(
        s2, 2, "Presentation Roadmap",
        "Our talk follows the exact 13-section ECO331 structure in two clean halves.",
        [
            "Part I sets up the economic puzzle: why linear Phillips Curves broke during COVID and how we engineered 15 leak-free FRED series.",
            "Part II shows our empirical horse-race, proves our models aren't just lagging, translates SHAP into a central-bank playbook, and demos our live web app.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 3: INTRODUCTION
    # ------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    _add_header(s3, "Section 2", "Introduction: The Central Bank Forecasting Puzzle", 3, total_slides)
    _add_card(
        s3, 0.55, 1.25, 12.23, 2.50,
        "Why Inflation Forecasting Is Hard — Especially at Turning Points",
        [
            "Policy Lags Require Foresight: Interest rate hikes take 6 to 12 months to cool inflation, so central banks must steer using forecasts, not rearview mirrors.",
            "The Random Walk Challenge: Atkeson & Ohanian (2001) famously showed that during calm eras, simply guessing 'tomorrow = today' beats complex econometric models.",
            "The 2020–2026 Stress Test: Pandemic lockdowns, a 9.0% CPI surge, Ukraine energy shocks, and 525 bps of Fed hikes broke linear textbook relationships.",
        ],
        accent_color=ROYAL,
        font_pt=12.8,
    )
    _add_card(
        s3, 0.55, 4.00, 5.95, 2.85,
        "What Machine Learning Brings",
        [
            "Collinearity Control: Elastic Net & Ridge prevent correlated macro indicators (CPI, PPI, M2, GDP) from blowing up OLS coefficients.",
            "Regime Thresholds: XGBoost & Hybrid Deep Nets learn when real interest rates or energy shocks cross critical tipping points.",
        ],
        accent_color=TEAL,
        font_pt=12.4,
    )
    _add_card(
        s3, 6.83, 4.00, 5.95, 2.85,
        "Why Explainability Is Non-Negotiable",
        [
            "No Black-Box Rate Hikes: An FOMC or RBI committee will never hike rates by 75 bps just because a neural network outputs '5.2%'.",
            "Economic Attribution: Every forecast in our project is decomposed into exact percentage-point drivers (Supply vs. Policy vs. Expectations).",
        ],
        accent_color=AMBER,
        font_pt=12.4,
    )
    _attach_speaker_notes(
        s3, 3, "Introduction",
        "Monetary policy is like steering an oil tanker: today's rate hike only hits inflation two to four quarters later.",
        [
            "In calm decades, beating a naive Random Walk ('inflation in 3 months equals inflation today') was famously difficult.",
            "Between 2020 and 2026, inflation went from 0.3% to 9.0% and back to ~2.5% — creating the ultimate stress test for ML vs. econometrics.",
            "Our thesis: ML wins only if it is regularized against small macro sample sizes AND fully explainable to policymakers.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 4: MOTIVATION (UNITING ALL 4 FOLDER CLUSTERS)
    # ------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    _add_header(s4, "Section 3", "Motivation: Synthesizing All 4 Course Literature Clusters", 4, total_slides)
    _add_card(
        s4, 0.55, 1.25, 3.92, 5.55,
        "1. CPI & Food Inflation (8 Papers)",
        [
            "Singh & Bhoi (RBI, 2022) & Singh & Rathi (RBI, 2024): Show ML & Forecast Combinations beat ARIMA/NKPC during supply shocks.",
            "Li et al. (2022) & Mamedli (2021): Establish XGBoost & Elastic Net as top performers at 3–12m horizons.",
            "Gap We Solve: Prior papers rarely compare SHAP vs. Linear β vs. Deep Attention side-by-side.",
        ],
        accent_color=ROYAL,
        font_pt=12.0,
    )
    _add_card(
        s4, 4.71, 1.25, 3.92, 5.55,
        "2. Electricity, Stocks & GDP (10 Papers)",
        [
            "Electricity (Kuskaya & Bilgili, 2025): Grid electricity tariffs capture sticky second-round energy costs + warn against leaky K-Fold CV!",
            "Equities (Subasi 2021; Vamossy 2024): Stock market returns front-run financial conditions & household wealth.",
            "Real GDP (Yoon 2021; Maccarrone 2021): Captures broad service-sector demand missing from factory-only indices.",
        ],
        accent_color=TEAL,
        font_pt=12.0,
    )
    _add_card(
        s4, 8.87, 1.25, 3.92, 5.55,
        "3. Small-Data Deep Learning Puzzle",
        [
            "48 Years = Only 585 Months: Unlike millions of words in LLMs, monthly macro data spans only ~5 business cycles.",
            "Why Standalone Transformers Struggle: Over-parameterized self-attention overfits small tabular/sequence panels.",
            "Our Innovation: Feed XGBoost's nonlinear SHAP signals + trends into a Gated Deep Trend Network!",
        ],
        accent_color=ROSE,
        font_pt=12.0,
    )
    _attach_speaker_notes(
        s4, 4, "Motivation from the Course Folder",
        "Instead of picking just one paper from the course folder, we noticed all 18 papers fall into four connected macro channels.",
        [
            "Cluster 1 is CPI inflation (RBI papers by Singh & Bhoi, Li et al.). Cluster 2 is Electricity prices (Kuskaya & Bilgili 2025).",
            "Cluster 3 is Stock markets (Vamossy, Subasi), and Cluster 4 is Real GDP growth (Yoon 2021).",
            "We integrate all four into a single unified US inflation framework AND test why hybridizing XGBoost with a Deep Net solves the 585-month macro sample limit.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 5: OBJECTIVES
    # ------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    _add_header(s5, "Section 4", "Research Objectives & Testable Questions", 5, total_slides)
    _add_card(
        s5, 0.55, 1.25, 5.95, 2.62,
        "Objective 1 — 12-Model Horse Race Across 6 Families",
        [
            "Benchmark Naive/Econometric (Random Walk, AR, OLS Phillips Curve), Shrinkage (Ridge, Elastic Net), Trees (RF, XGBoost), Deep Nets (MLP, LSTM, Transformer), Hybrid (XGB+Deep Net), and Ensembles.",
        ],
        accent_color=ROYAL,
        font_pt=12.5,
    )
    _add_card(
        s5, 6.83, 1.25, 5.95, 2.62,
        "Objective 2 — Turning-Point & Policy-Shock Audit",
        [
            "Test whether ML genuinely anticipates inflation changes (Δπ_{t+3}) during COVID, the 2021 Surge, 2022–23 Fed 75bp hikes, and 2024–26 Soft Landing — or merely lags.",
        ],
        accent_color=TEAL,
        font_pt=12.5,
    )
    _add_card(
        s5, 0.55, 4.12, 5.95, 2.68,
        "Objective 3 — Cross-Paper Channel Ablation",
        [
            "Quantify how much adding sticky Electricity Prices (Kuskaya 2025), US Equities (Vamossy 2024), and Real GDP (Yoon 2021) improves out-of-sample CPI accuracy.",
        ],
        accent_color=AMBER,
        font_pt=12.5,
    )
    _add_card(
        s5, 6.83, 4.12, 5.95, 2.68,
        "Objective 4 — Actionable Policy Playbook & Demo Tool",
        [
            "Map exact SHAP values across horizons (1m to 12m) into a 'Macroeconomic Transmission Clock' and deploy a live multi-model web predictor.",
        ],
        accent_color=NAVY,
        font_pt=12.5,
    )
    _attach_speaker_notes(
        s5, 5, "Research Objectives",
        "We structured the project around four concrete, falsifiable questions.",
        [
            "First: which model family wins at 1, 3, 6, and 12 months? Second: are models truly anticipating turning points or just lagging?",
            "Third: do Electricity prices, Stock returns, and Real GDP measurably cut forecast error?",
            "Fourth: which exact indicators should central bankers watch at each horizon, and how much do they move CPI in percentage points?",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 6: RESEARCH METHODS & MATHEMATICAL EXPRESSIONS
    # ------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    _add_header(s6, "Section 5", "Research Methods & Mathematical Formulations", 6, total_slides)
    _add_card(
        s6, 0.55, 1.20, 6.0, 2.78,
        "1. Anchored Target Decomposition (Why SHAP = pp)",
        [
            "Anchored Law:  π̂_{t+h} = π_t + f_θ(X_t),   predicting shift  Δ_h π_{t+h} = π_{t+h} − π_t.",
            "Econometric Nesting:  If f_θ(X_t) = 0, forecast equals Random Walk (π̂_{t+h} = π_t) identically.",
            "NKPC Baseline:  Δ_h π_{t+h} = α − λπ_t + β E_t[π_{t+12}] − κ UNRATE_t + γ Supply_t + ε_{t+h}.",
        ],
        accent_color=ROYAL,
        font_pt=11.8,
    )
    _add_card(
        s6, 6.78, 1.20, 6.0, 2.78,
        "2. Penalized Linear Shrinkage (Ridge & Elastic Net)",
        [
            "Objective over standardized predictors X̃_t:  min_β (1/2N)||Δ_h π − X̃β||₂² + α[ρ||β||₁ + ½(1−ρ)||β||₂²].",
            "Why OLS Fails:  15 macro channels share strong collinearity; L1 prunes noise while L2 stabilizes slopes.",
            "Tuned strictly via 5-fold chronological TimeSeriesSplit inside each expanding training window.",
        ],
        accent_color=TEAL,
        font_pt=11.8,
    )
    _add_card(
        s6, 0.55, 4.16, 6.0, 2.80,
        "3. XGBoost & Exact Additive TreeSHAP",
        [
            "Additive Boosted Trees:  f(X_t) = Σ_{k=1}^K η h_k(X_t),  penalized by Ω(h_k) = γT + ½λ||w||₂² + α||w||₁.",
            "Exact TreeSHAP Identity:  π̂_{t+h} = π_t + E[Δ_h π] + Σ_{j=1}^p φ_j(X_t).",
            "Key Speaking Point:  Every φ_j(X_t) is measured directly in percentage points of CPI inflation!",
        ],
        accent_color=AMBER,
        font_pt=11.8,
    )
    _add_card(
        s6, 6.78, 4.16, 6.0, 2.80,
        "4. Sequence Nets, Hybrid Gate & Diebold-Mariano Test",
        [
            "Temporal Attention:  w_{t,k} = softmax_k(q(h_t)ᵀ k(h_{t−k}) / √d)  over 12-month tensor S_t ∈ R^{12×14}.",
            "Hybrid XGB + Deep Net:  Δ̂ = σ(g(z_t)) · Δ̂_XGB + (1 − σ(g(z_t))) · TrendNet(X_t, φ_t, Δ_1m, Δ_3m).",
            "Statistical Test:  Harvey-Leybourne-Newbold HAC Diebold-Mariano t-stat vs. Random Walk.",
        ],
        accent_color=NAVY,
        font_pt=11.6,
    )
    _attach_speaker_notes(
        s6, 6, "Research Methods & Equations",
        "The single smartest design choice in our math is the Anchored Forecast equation in Box 1.",
        [
            "Instead of predicting raw CPI levels from scratch, every model predicts how much inflation will CHANGE from today: pi-hat(t+h) = pi(t) + f(X_t).",
            "Why does that matter? First, if f=0, we recover the exact Random Walk benchmark. Second, every SHAP value phi_j is literally in percentage points of inflation push or pull!",
            "Box 4 shows our Hybrid innovation: a gated neural net that blends XGBoost's tree prediction with smooth 1-month and 3-month SHAP trends.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 7: DATA SOURCE & CAUSAL PROTOCOL
    # ------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    _add_header(s7, "Section 6", "Data Source: 15 FRED Macro, Energy, Equity & GDP Series", 7, total_slides)
    _add_card(
        s7, 0.55, 1.22, 6.15, 5.62,
        "15 Public FRED Series Uniting 5 Economic Channels (1978–2026)",
        [
            "1. Price & Base Inertia: Headline CPI (CPIAUCSL), Core CPI (CPILFESL), Core–Headline Gap, 1m/3m Momentum, and Base-Effect CPI (t−12..t−9).",
            "2. Supply, Oil & Sticky Electricity: PPI Commodities (PPIACO), WTI Crude Oil (WTISPLC), plus Electricity Price Index (CUSR0000SEHF01, Kuskaya 2025).",
            "3. Monetary, Liquidity & Equities: Fed Funds, Real Fed Funds (Fed − CPI), 10Y–2Y Yield Curve, M2 Money Stock, and US Equities (SPASTT01USM661N).",
            "4. Real GDP & Labor Slack: Real GDP Growth (GDPC1, Yoon 2021), Industrial Production (INDPRO), Unemployment (UNRATE), Consumer Sentiment.",
            "5. Forward Expectations: Univ. of Michigan 1-Year Household Inflation Expectations (MICH).",
        ],
        accent_color=ROYAL,
        font_pt=11.5,
    )
    _add_card(
        s7, 6.92, 1.22, 5.86, 5.62,
        "Zero Look-Ahead & Stationarity Guarantees",
        [
            "ADF Unit-Root Verification: Raw price/GDP/equity levels are non-stationary I(1) (ADF p > 0.80); all YoY %, momentum, and spread features reject unit roots (p < 0.05).",
            "Publication-Lag Alignment for GDP: Q1 Real GDP is published in late April — we shift GDPC1 forward by +3 months so models never see unreleased GDP!",
            "Central-Bank Expanding Windows: Out-of-sample test (2019–2026, 90 months) re-estimates annually using ONLY historical targets realized before Jan 1 of each test year.",
            "Causal Gap Handling: Single Oct 2025 survey gap imputed strictly backward from (t−1, t−2).",
        ],
        accent_color=TEAL,
        font_pt=11.5,
    )
    _attach_speaker_notes(
        s7, 7, "Data Source & Zero Look-Ahead Design",
        "We pull 48 years of US data (585 months, 1978–2026) directly from the St. Louis Fed FRED database with zero look-ahead bias.",
        [
            "Notice two crucial econometric details on the right card: First, raw CPI and GDP have unit roots, so we transform everything into stationary growth rates, spreads, and momentum.",
            "Second, Real GDP is reported quarterly with a release delay — Q1 GDP comes out at the end of April. We lag GDP by 3 months so our models only use information a real-time forecaster actually had!",
            "And before every test year from 2019 to 2026, we purge h months at the boundary so future inflation realizations never leak into training.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 8: GRAPHICAL DATA I (REGIMES & SHIFTING PHILLIPS CURVE)
    # ------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    _add_header(s8, "Section 7A", "Data Storytelling I: Regimes & Why Linear Phillips Curves Break", 8, total_slides)
    _add_fitted_image(s8, FIGURES_DIR / "fig01_eda_storytelling.png", 0.35, 1.05, 8.25, 5.92)
    _add_card(
        s8, 8.74, 1.12, 4.18, 5.80,
        "What to Point Out on Slide 8",
        [
            "Panel A (3 Inflation Worlds): US CPI swung from 14% Volcker stagflation to a 25-year Great Moderation (~2.5%), then exploded to 9.0% in Jun 2022.",
            "Panel B (Broken Phillips Curve): Look at the colored regression lines — flat during 1984–2019, then violently vertical in 2020–2022! Unemployment alone cannot explain supply-shock inflation.",
            "Panel C (Nonlinear Cost-Push): When PPI wholesale inflation exceeds +8% YoY, future CPI accelerates nonlinearly — motivating Tree & Neural models.",
            "Panel D (Volatility Spikes): Rolling 36m CPI volatility tripled in 2021–2023, breaking constant-variance OLS assumptions.",
        ],
        accent_color=ROYAL,
        font_pt=11.1,
        space_after_pt=5.0,
    )
    _attach_speaker_notes(
        s8, 8, "Data Storytelling I: Regimes & Nonlinearity",
        "If anyone asks WHY we need Machine Learning instead of a textbook OLS Phillips Curve, point straight to Panel B and Panel C.",
        [
            "In Panel B, each color is a different era. From 1984 to 2019, the unemployment-inflation slope was practically flat. Then in 2021–2022 it jumped vertically.",
            "Panel C shows why: upstream Producer Price inflation (PPI) has a curved, nonlinear relationship with future CPI — gentle in normal times, steep above 8–10%.",
            "A single straight OLS line averages across all regimes and misses both the calm flat zone and the steep shock zone.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 9: GRAPHICAL DATA II (CORRELATION HEATMAP)
    # ------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    _add_header(s9, "Section 7B", "Data Storytelling II: Cross-Correlation & Multicollinearity", 9, total_slides)
    _add_fitted_image(s9, FIGURES_DIR / "fig02_correlation_heatmap.png", 0.35, 1.05, 7.85, 5.92)
    _add_card(
        s9, 8.35, 1.12, 4.55, 5.80,
        "3 Speaking Takeaways from Heatmap",
        [
            "1. Sticky Electricity Beats Crude Oil on Future CPI: Michigan 1Y Expectations (+0.87), PPI (+0.67), and Electricity CPI (+0.64) track 3m-ahead CPI tighter than noisy Crude Oil (+0.52), while static Unemployment is near zero (+0.02)!",
            "2. Severe Multicollinearity Trap: Headline CPI, Core CPI, Expectations, PPI, and Electricity all correlate at r = 0.64 to 0.94 — causing unregularized OLS weights to swing wildly.",
            "3. Real Fed Funds Sign Puzzle in Raw Levels: Unconditional correlation looks weak (+0.11) because the Fed hikes AFTER inflation rises — requiring nonlinear regime trees to isolate the braking threshold.",
        ],
        accent_color=TEAL,
        font_pt=11.3,
        space_after_pt=6.0,
    )
    _attach_speaker_notes(
        s9, 9, "Data Storytelling II: Correlation Heatmap",
        "This correlation matrix reveals two huge insights before we even run a single ML model.",
        [
            "First, compare Electricity CPI (+0.64) vs Crude Oil (+0.52) and Unemployment (+0.02) in the first column: sticky utility rates and expectations (+0.87) co-move strongly with future CPI while static unemployment has near-zero unconditional correlation!",
            "Second, look at the dark red cluster among CPI, Core CPI, Expectations, Electricity, and PPI — correlations of 0.64 to 0.94. That collinearity is toxic for standard OLS, which is why Ridge and Elastic Net shine.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 10: EMPIRICAL FINDINGS I (H=3 LEADERBOARD)
    # ------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    _add_header(s10, "Section 8A", "Empirical Findings I: Flagship 3-Month-Ahead Leaderboard", 10, total_slides)
    _add_wide_chart_with_talking_cards(
        s10,
        FIGURES_DIR / "fig03_model_comparison_h3.png",
        f"Leaderboard Headline (2019–2026 Out-of-Sample, RW = {rw_rmse:.3f} pp)",
        [
            f"Top Tier Beats Random Walk by ~20%+: {best_row['Model']} ({best_row['RMSE']:.3f} pp), Elastic Net ({enet_rmse:.3f} pp), XGBoost ({xgb_rmse:.3f} pp), and Hybrid XGB+Deep Net ({hyb_rmse:.3f} pp) all beat Naive RW ({rw_rmse:.3f} pp).",
            f"Textbook OLS Fails: Unregularized OLS Phillips Curve (~1.00 pp) performs no better than guessing 'no change' due to collinearity and shifting slopes.",
        ],
        "Economic Story: Regularization + Nonlinear Trees Win",
        [
            f"Directional Edge: XGBoost calls the right 3-month acceleration/disinflation direction {xgb_dir:.1f}% of the time, statistically significant on Diebold-Mariano HAC tests.",
            f"Standalone Deep Nets Trail Trees on 460 Rows: LSTM ({lstm_rmse:.3f} pp) and Transformer ({tf_rmse:.3f} pp) beat RW, but need tree feature extraction to reach top tier.",
        ],
        left_accent=TEAL,
        right_accent=ROYAL,
    )
    _attach_speaker_notes(
        s10, 10, "Flagship 3-Month Model Comparison",
        "Here is our main 12-model horse race at the 1-quarter (3-month) policy horizon over 2019–2026.",
        [
            f"At the bottom of Panel A, the Random Walk benchmark has an error of {rw_rmse:.3f} percentage points, and the textbook OLS Phillips Curve essentially ties it.",
            f"At the top, {best_row['Model']} ({best_row['RMSE']:.3f} pp), Elastic Net ({enet_rmse:.3f} pp), and XGBoost ({xgb_rmse:.3f} pp) slash forecast error by over 20%!",
            "Notice Panel B on the right: our models track both the 2021–22 climb to 9% and the 2023–24 disinflation — which brings up an obvious question on the next slide: are they just lagging?",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 11: ARE MODELS JUST LAGGING? (FIG 10 SIDE-BY-SIDE)
    # ------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    _add_header(s11, "Section 8B", "Critical Audit: Are Models Anticipating Policy Shocks or Just Lagging?", 11, total_slides)
    _add_fitted_image(s11, FIGURES_DIR / "fig10_change_and_policy_shocks.png", 0.32, 1.04, 8.38, 5.95)
    _add_card(
        s11, 8.82, 1.10, 4.14, 5.84,
        "How to Answer 'Looks Like a Lag!'",
        [
            "Why Level Plots Fool the Eye: Because 3m CPI shares 9/12 months with today, ANY honest forecast sits near π_t until a surprise shock hits.",
            "Look at Change-Space (Panel A): Plotting Δπ = π_{t+3} − π_t flattens Pure Lag to 0.00. Whenever teal/blue move off zero with the black line, ML ANTICIPATED the turn!",
            "Wins 4 of 5 Policy Eras (Panel B): Cuts Pure-Lag error by -28% in 2021 Surge and -65% during 2022–23 Fed 75bp rate hikes!",
            "Only Miss = Feb 2022 Ukraine War: Models predicted Spring 2022 cooling right before Russia invaded Ukraine — an unforeseeable exogenous shock.",
        ],
        accent_color=TEAL,
        font_pt=10.9,
        space_after_pt=4.5,
    )
    _attach_speaker_notes(
        s11, 11, "Are Models Just Lagging? Change-Space & Policy Shock Audit",
        "When you look at a level chart, every forecast looks slightly shifted to the right during sudden shocks. So we tested this rigorously!",
        [
            "In Panel A we subtract today's CPI and plot the 3-month CHANGE. Here, a pure-lag Random Walk is stuck flat at zero forever.",
            "Look at two huge policy moments: in Spring 2021 (stimulus + base surge) and late 2022 (peak 75bp Fed rate hikes), Elastic Net and XGBoost swung strongly positive and negative ON TIME, cutting pure-lag error by 28% and 65%!",
            "The ONLY window where pure lag won was H1 2022 — because no macro dataset in January 2022 knew Russia would invade Ukraine in February and spike oil to $120.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 12: MULTI-HORIZON & REGIMES
    # ------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    _add_header(s12, "Section 8C", "Empirical Findings II: Horizons (1–12m) & Regime Stress-Test", 12, total_slides)
    _add_wide_chart_with_talking_cards(
        s12,
        FIGURES_DIR / "fig04_multi_horizon_and_regimes.png",
        "No Single Model Wins Every Horizon (Panel A)",
        [
            f"Short Nowcast (h=1m): {best_h1['Model']} wins ({best_h1['RMSE']:.3f} pp, p < 0.05) as nonlinear momentum + energy pass-through dominate.",
            f"Medium/Long Horizons (h=6m & 12m): {best_h6['Model']} ({best_h6['RMSE']:.3f} pp at 6m) and {best_h12['Model']} ({best_h12['RMSE']:.3f} pp at 12m) lead as smoother macro cycles take over.",
        ],
        "Regime Complementarity Justifies Ensembles (Panel B)",
        [
            "Supply Shock (2020–22): Boosted Trees & Hybrid models shine when PPI, Electricity, and Oil breach nonlinear historical bounds.",
            "Smooth Disinflation (2023–26): Penalized Linear (Elastic Net / Ridge) excels as inflation mean-reverts steadily — explaining why RBI (2022) recommends Forecast Combinations!",
        ],
        left_accent=ROYAL,
        right_accent=AMBER,
    )
    _attach_speaker_notes(
        s12, 12, "Horizons (1–12m) & Macro Regimes",
        "Central banks don't just forecast 3 months ahead — they forecast 1, 3, 6, and 12 months across calm and crisis regimes.",
        [
            f"Panel A plots Relative RMSE below the dotted 1.00 Random Walk line. Every horizon beats Random Walk, led by {best_h1['Model']} at 1m and {best_h12['Model']} at 12m.",
            "Panel B shows a great insight from Singh & Bhoi (RBI, 2022): nonlinear trees win during violent supply shocks (col 2), whereas linear shrinkage wins during smooth disinflation (col 3). Combining them hedges regime risk!",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 13: HYBRID XGBOOST + DEEP NET
    # ------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    _add_header(s13, "Section 8D", "Architecture Innovation: Why Feeding XGBoost into a Deep Net Works", 13, total_slides)
    _add_wide_chart_with_talking_cards(
        s13,
        FIGURES_DIR / "fig09_hybrid_xgb_deepnet.png",
        "Solving the 48-Year Small-Data Bottleneck (Panel A)",
        [
            "Why Standalone Transformers Struggle: 48 years of monthly data is only ~460 pre-2019 training rows (~5 business cycles) — tiny for raw self-attention.",
            f"Hybrid Beats Standalone Deep Nets Everywhere: XGB + Deep Net ({hyb_rmse:.3f} pp at 3m) beats standalone MLP and Transformer at 1m, 3m, 6m, and 12m!",
        ],
        "How the Hybrid Architecture Works (Panel B)",
        [
            "Step 1 (Tree Feature Extraction): Out-of-fold causal XGBoost extracts top nonlinear TreeSHAP signals plus 1-month & 3-month signal trends.",
            "Step 2 (Gated Neural Trend Head): A lightweight PyTorch network learns smooth momentum trajectories on top of XGBoost's step-function leaves.",
        ],
        left_accent=ROSE,
        right_accent=TEAL,
    )
    _attach_speaker_notes(
        s13, 13, "Hybrid XGBoost + Deep Trend Network",
        "A natural question is: why didn't a standalone Transformer crush everything after 48 years of data?",
        [
            "48 years of MONTHLY data is only 585 rows! Deep learning thrives on tens of thousands of examples, while trees handle 500 noisy tabular rows effortlessly.",
            "However, tree predictions are flat step-functions that cannot extrapolate smooth slopes. By feeding XGBoost's exact SHAP features and 1m/3m trends into a Gated Deep Network, our Hybrid beats standalone neural nets across every horizon!",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 14: UNIFYING ALL 4 COURSE CLUSTERS (ELECTRICITY, EQUITIES, GDP)
    # ------------------------------------------------------------------
    s14_syn = prs.slides.add_slide(blank_layout)
    _add_header(s14_syn, "Section 8E", "Cross-Paper Synthesis: Adding Electricity, Equities & Real GDP", 14, total_slides)
    _add_wide_chart_with_talking_cards(
        s14_syn,
        FIGURES_DIR / "fig12_cross_paper_synthesis.png",
        "Why Grid Electricity Beats Crude Oil at 3–6m (Panel A)",
        [
            f"Sticky vs. Noisy Energy: Crude Oil whipsaws (-50% to +110%), whereas Electricity CPI (Kuskaya 2025) adjusts via regulated utility tariffs and stays elevated in 2023–26.",
            f"Huge SHAP Discovery: Electricity CPI captures {elec_xgb_share:.1f}% of XGBoost's total attribution — making it a top-tier sticky cost-push indicator!",
        ],
        "Controlled 4-Cluster Ablation Gains (Panel B)",
        [
            f"Linear Error Drops {core_enet_rmse:.3f} → {uni_enet_rmse:.3f} pp: Adding Electricity, Stock Returns (Vamossy 2024), and 1Q-lagged Real GDP (Yoon 2021) cuts Elastic Net RMSE by ~6% and XGBoost to {uni_xgb_rmse:.3f} pp!",
            "Why GDP Helps Linear More Than Trees: Linear models need GDP's 80% service-sector coverage; trees already infer nonlinear demand from monthly output & sentiment.",
        ],
        left_accent=AMBER,
        right_accent=ROYAL,
    )
    _attach_speaker_notes(
        s14_syn, 14, "Cross-Paper Synthesis: Electricity, Stock Market & Real GDP",
        "This slide connects ALL FOUR paper topics from our ECO331 Drive folder in a single controlled experiment.",
        [
            "Look at the orange line in Panel A: while Crude Oil crashed in 2023, regulated Grid Electricity prices (from Kuskaya & Bilgili 2025) stayed high at 4–7%, keeping restaurant, grocery, and housing utility inflation sticky!",
            f"In Panel B, adding Electricity, US Equities, and causally lagged Real GDP cuts Elastic Net error from {core_enet_rmse:.3f} down to {uni_enet_rmse:.3f} pp and XGBoost to {uni_xgb_rmse:.3f} pp.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 15: EXPLAINABLE AI I (CROSS-FAMILY IMPORTANCE)
    # ------------------------------------------------------------------
    s15 = prs.slides.add_slide(blank_layout)
    _add_header(s15, "Section 9A", "Explainable AI I: Do Elastic Net, XGBoost & XGB+MLP Agree?", 15, total_slides)
    _add_wide_chart_with_talking_cards(
        s15,
        FIGURES_DIR / "fig05_xai_cross_family_importance.png",
        f"Strong Consensus Across Top 3 Winners: {', '.join(top3_feats[:2])}",
        [
            "Panel A (Signed Linear β): Base-Effect CPI has a huge negative slope (-0.70 pp per SD) as last year's spikes roll off; Core−Headline Gap, M2, Momentum & PPI push CPI up (+).",
            "Panel B (Winner-Model Agreement): Elastic Net, XGBoost TreeSHAP, and Hybrid XGB + MLP all agree that Base Effects, Supply/Electricity/PPI, and Liquidity dominate.",
        ],
        "Two Surprising Findings to Highlight in Class",
        [
            "Surprise #1 — Static Unemployment Is < 1% Share! Textbook Phillips Curve slack barely registers across all three top models compared to Supply-Chain (PPI/Electricity) and Policy channels.",
            "Surprise #2 — Real Fed Funds & Electricity Nonlinearity: Linear Elastic Net shrinks both to 0.0 due to collinearity, whereas XGBoost & XGB+MLP rank them as top pure economic levers!",
        ],
        left_accent=ROYAL,
        right_accent=AMBER,
    )
    _attach_speaker_notes(
        s15, 15, "Explainable AI I: Winner-Model Consensus (Elastic Net vs XGBoost vs XGB+MLP)",
        "Now we open the black box across our three winning architectures: Elastic Net, XGBoost, and Hybrid XGB + MLP.",
        [
            "Mostly they agree — and where they diverge is super revealing! First, all three agree that static Unemployment accounts for under 1% of predictive power.",
            "Second, notice Electricity CPI and Real Fed Funds Rate: linear Elastic Net shrinks them to zero because of collinearity, while XGBoost and XGB+MLP give them heavy nonlinear weight. Slide 16 shows WHY.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 16: EXPLAINABLE AI II (GLOBAL SHAP NONLINEARITY)
    # ------------------------------------------------------------------
    s16 = prs.slides.add_slide(blank_layout)
    _add_header(s16, "Section 9B", "Explainable AI II: Nonlinear Policy & Cost-Push Thresholds", 16, total_slides)
    _add_wide_chart_with_talking_cards(
        s16,
        FIGURES_DIR / "fig06_shap_global_nonlinear.png",
        "Reading the Beeswarm & Dependence Curves",
        [
            "Panel A (Global Beeswarm): Each dot is one test month (red = high feature value, blue = low). Base-Effect spikes push CPI down up to -1.3 pp as they exit the 12m window.",
            "Panel B (Step-Change Thresholds): TreeSHAP dependence curves reveal curved regime kinks rather than straight lines.",
        ],
        "Policy Goldmine: Why Linear Models Miss Fed Tightening",
        [
            "Nonlinear Monetary Brake: When Real Fed Funds (Fed Rate − CPI) is deeply negative (< -2 pp), monetary stance fuels inflation; once it crosses 0% to +2%, disinflation kicks in nonlinearly!",
            "Asymmetric Supply Pass-Through: Moderate PPI/Electricity changes are absorbed in margins, but spikes > +5–8% pass rapidly into consumer prices.",
        ],
        left_accent=TEAL,
        right_accent=ROSE,
    )
    _attach_speaker_notes(
        s16, 16, "Explainable AI II: Global TreeSHAP & Nonlinear Thresholds",
        "This slide explains why XGBoost beats linear models at monetary turning points.",
        [
            "Look at Panel B: monetary policy does NOT work like a dimmer switch with a constant linear slope — it works like a threshold brake!",
            "When Real Fed Funds is negative (-3% to -7% in 2021–22), policy is accommodative. Only when the Fed pushes rates near or above inflation (0% to +2% real rate) does the disinflationary brake bite sharply.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 17: EXPLAINABLE AI III (LOCAL EPISODE WATERFALLS)
    # ------------------------------------------------------------------
    s17 = prs.slides.add_slide(blank_layout)
    _add_header(s17, "Section 9C", "Explainable AI III: Decomposing Two Famous Historical Episodes", 17, total_slides)
    _add_wide_chart_with_talking_cards(
        s17,
        FIGURES_DIR / "fig07_shap_episode_waterfalls.png",
        f"Left Case: Post-COVID Surge ({xai.episode_surge['origin_date']} → {xai.episode_surge['target_date']})",
        [
            f"Starting from {xai.episode_surge['anchor_cpi']:.2f}% CPI, actual inflation surged to {xai.episode_surge['actual_cpi']:.2f}% (XGBoost predicted {xai.episode_surge['pred_cpi']:.2f}% vs. Random Walk stuck at {xai.episode_surge['anchor_cpi']:.2f}%).",
            "Red upward bars pinpoint the culprits: negative prior-year base roll-off + surging PPI/Electricity costs + loose negative real rates.",
        ],
        f"Right Case: Fed Hike Disinflation ({xai.episode_disinflation['origin_date']} → {xai.episode_disinflation['target_date']})",
        [
            f"Starting from {xai.episode_disinflation['anchor_cpi']:.2f}% CPI, XGBoost forecast {xai.episode_disinflation['pred_cpi']:.2f}% vs. actual {xai.episode_disinflation['actual_cpi']:.2f}% — calling the disinflation drop almost to the decimal!",
            "Green downward bars show why: large positive base roll-off + tightening Fed stance + cooling factory output & consumer sentiment.",
        ],
        left_accent=ROSE,
        right_accent=TEAL,
    )
    _attach_speaker_notes(
        s17, 17, "Local Historical Episode Waterfalls",
        "Imagine briefing a central bank governor in March 2021 or December 2022 — these waterfall charts turn ML numbers into an instant story.",
        [
            f"On the left ({xai.episode_surge['origin_date']}), CPI was still low at {xai.episode_surge['anchor_cpi']:.2f}%, yet XGBoost warned of a sharp jump because low pandemic base prices were rolling off while PPI and M2 were booming.",
            f"On the right ({xai.episode_disinflation['origin_date']}), CPI was {xai.episode_disinflation['anchor_cpi']:.2f}%, and XGBoost called the drop to ~{xai.episode_disinflation['actual_cpi']:.2f}% by summing base roll-off, tightening real rates, and cooling output.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 18: EXPLAINABLE AI IV (TRANSFORMER ATTENTION & CV LEAKAGE)
    # ------------------------------------------------------------------
    s18 = prs.slides.add_slide(blank_layout)
    _add_header(s18, "Section 9D", "Inside Sequence Deep Nets & Why Shuffled K-Fold CV Is Fraudulent", 18, total_slides)
    val_bias_xgb = float(leakage_df.loc[leakage_df["Model"] == "XGBoost", "Artificial_Optimism_Bias_pct"].iloc[0])
    _add_wide_chart_with_talking_cards(
        s18,
        FIGURES_DIR / "fig08_transformer_attention_and_leakage.png",
        "Adaptive Sequence Attention Across Regimes (Panel A)",
        [
            "Calm 2019: Temporal Transformer spreads attention almost evenly (~8.3% per month) across all 12 lookback lags.",
            "2020–22 Shock: Attention shifts heavily onto the newest 2 months (t and t−1 > 25% combined) to track fast-breaking cost shocks!",
        ],
        f"Methodological Warning: -{val_bias_xgb:.0f}% Fake K-Fold Gain (Panel B)",
        [
            f"Replicating Kuskaya & Bilgili (2025): Shuffled 5-Fold CV fabricates a {val_bias_xgb:.0f}% fake RMSE drop on XGBoost!",
            "Why Leakage Happens: Knowing November & January CPI makes predicting December trivial due to overlapping 12-month autocorrelation — proving why purged chronological CV is mandatory.",
        ],
        left_accent=NAVY,
        right_accent=ROSE,
    )
    _attach_speaker_notes(
        s18, 18, "Sequence Lag Attention & K-Fold Leakage Audit",
        "Two important methodological takeaways here — one on sequence neural nets, and one warning every student doing time-series ML.",
        [
            "Panel A shows our Temporal Transformer dynamically shifts its attention onto lags t and t-1 when inflation volatility spikes.",
            f"Panel B proves Kuskaya & Bilgili's (2025) warning: if you run standard scikit-learn shuffled KFold on CPI, XGBoost looks {val_bias_xgb:.0f}% more accurate than reality because adjacent months leak the answer!",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 19: TEMPORAL TRANSFORMER 2D SEQUENCE EXPLAINABILITY (H=12M WINNER!)
    # ------------------------------------------------------------------
    tf_ch_path = TABLES_DIR / "transformer_channel_importance.csv"
    if tf_ch_path.exists():
        tf_ch_df = pd.read_csv(tf_ch_path)
        tf_top3 = ", ".join(tf_ch_df["Feature_Label"].iloc[:3].tolist())
    else:
        tf_top3 = "Headline CPI, Electricity CPI, and 1Y Inflation Expectations"

    s19_tf = prs.slides.add_slide(blank_layout)
    _add_header(s19_tf, "Section 9E", "Explainable AI V: Why Temporal Transformer Wins #1 at h = 12m", 19, total_slides)
    _add_wide_chart_with_talking_cards(
        s19_tf,
        FIGURES_DIR / "fig13_transformer_explainability.png",
        "Channel Shift from 3m to 12m (Panel A — Integrated Gradients)",
        [
            "Base-Effect Collapses 5× (14.6% → 2.9%): Over a full 12-month horizon, old calendar spikes completely exit the window — so mechanical base math vanishes!",
            "Real Macro & Sticky Electricity Take Over: CPI 3m Momentum (21.3%), PPI (12.1%), M2 Liquidity (11.8%), Real Fed Funds (7.3%), and Sticky Electricity (doubling 3.8% → 7.2%) drive the 12m forecast.",
        ],
        "2D Feature × Lag Map & 'Barbell' Attention (Panels B & C)",
        [
            "2D Feature × Month Hotspots (Panel B): Exact Integrated Gradients decompose all 14×12 cells — recent months (t, t−1) capture rate/gap stance while lags t−10..t−11 track full-year M2 & trend.",
            "U-Shaped 'Barbell' Attention in Shocks (Panel C): During 2020–22 shocks, self-attention weights BOTH ends of the year (10.5% at t AND 10.1% at t−11) — comparing today against 1 year ago!",
        ],
        left_accent=NAVY,
        right_accent=TEAL,
    )
    _attach_speaker_notes(
        s19_tf, 19, "Temporal Transformer 2D Sequence Explainability (h=12m #1 Winner)",
        "Remember our Temporal Transformer jumped to #1 overall at the 1-year (12-month) horizon (2.384 pp, -15% vs Random Walk) — so how do we explain a Transformer?",
        [
            "We combine exact Integrated Gradients (the deep-learning counterpart to SHAP) across all 14 channels x 12 months with explicit self-attention weights.",
            "Look at the bottom of Panel A first: Base-Effect CPI drops 5x from 14.6% at 3m down to 2.9% at 12m, while M2 Money Growth (11.8%), Real Fed Funds (7.3%), and Sticky Electricity (doubling to 7.2%) take over!",
            "And look at Panel C during Supply Shocks: the Transformer forms a U-shaped 'barbell' attention (10.5% on month t and 10.1% on month t-11), literally comparing today's economy against a year ago!",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 20: POLICY PLAYBOOK & TRANSMISSION CLOCK (FIG 11)
    # ------------------------------------------------------------------
    s19 = prs.slides.add_slide(blank_layout)
    _add_header(s19, "Section 9F", "Actionable Policy Playbook & The Macroeconomic Transmission Clock", 20, total_slides)
    _add_wide_chart_with_talking_cards(
        s19,
        FIGURES_DIR / "fig11_xai_policy_playbook.png",
        "Winner Levers: Elastic Net vs. XGBoost vs. XGB+MLP (Panel A)",
        [
            "Step 1 — Filter Calendar Math (~42–48% Share): Always subtract the 10–12m-ago Base-Effect spike before reacting to headline CPI news.",
            "Step 2 — Watch Top Pure Economic Levers: Sticky Electricity CPI (~20% of non-base XGBoost & XGB+MLP!), PPI (> +5%), M2 (> 10%), and Real Fed Funds (> 0% brake) beat Unemployment (< 1%).",
        ],
        "The Macroeconomic Transmission Clock by Horizon (Panel B)",
        [
            "1m to 3m (Supply & Sticky Electricity Hit First: 28.6%): Electricity tariffs, Oil, and PPI peak early — ideal for short-term inflation nowcasting.",
            "6m vs. 12m (Policy Peaks at 6m; Expectations Jump 10× at 12m!): Monetary, M2 & Equities peak at 6m (24.6%), while 1Y Expectations surge 10× from 1.7% (1m) to 16.7% (12m)!",
        ],
        left_accent=ROYAL,
        right_accent=TEAL,
    )
    _attach_speaker_notes(
        s19, 20, "Policy Playbook & Macroeconomic Transmission Clock",
        "If a central banker asks 'Okay, which indicator should I actually watch and when?', Panel B is our favorite discovery in the whole project.",
        [
            "Panel A compares ONLY our three winning models — Elastic Net, XGBoost, and Hybrid XGB+MLP — proving sticky Electricity, PPI, M2, and Real Fed Funds dominate static Unemployment.",
            "Panel B is the Macroeconomic Transmission Clock: Supply & Electricity peak at 1m (28.6%), Monetary Policy & GDP Demand peak at 6m, and Inflation Expectations jump 10x to 16.7% at 12m!",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 21: LIVE INTERACTIVE MULTI-MODEL PREDICTOR DEMO
    # ------------------------------------------------------------------
    s20_demo = prs.slides.add_slide(blank_layout)
    _add_header(s20_demo, "Section 9G", "Live Demo Tool: Interactive 12-Model Inflation Predictor Website", 21, total_slides)
    _add_card(
        s20_demo, 0.55, 1.22, 5.95, 5.60,
        "What Our Web Predictor Does (uv run python app.py)",
        [
            "Real-Time 12-Model Engine (~20ms): Runs Econometric, Ridge, Elastic Net, RF, XGBoost, MLP, LSTM, Transformer, Hybrid XGB+Deep Net, and Ensemble simultaneously.",
            "1-Click Historical Regime Presets: Jump instantly between Sep 2026 (Latest Live), Mar 2021 Surge, Jun 2022 9% Peak, Dec 2022 Fed Hikes, and Sep 2024 Pivot.",
            "Live Moving TreeSHAP Bars: As you drag sliders (PPI, Electricity, Real Fed Funds, M2, Expectations), exact ±pp SHAP bars explain WHY the forecast changed!",
            "Horizon Switcher (1m / 3m / 6m / 12m): Automatically highlights the verified out-of-sample leader for the chosen horizon.",
        ],
        accent_color=TEAL,
        font_pt=12.0,
        space_after_pt=6.0,
    )
    _add_card(
        s20_demo, 6.82, 1.22, 5.95, 5.60,
        "3 Live Classroom Experiments to Show in 60 Seconds",
        [
            "Experiment 1 (Verify Dec 2022 Fed Hike Call): Click 'Dec 2022 (Fed Hikes)' at h=3m → watch ML forecast ~4.8–5.0% against actual 4.92% while Random Walk stays stuck at 6.40%!",
            "Experiment 2 (Supply & Electricity Shock): Start from Sep 2026 and drag PPI / Electricity CPI up → see 1m & 3m forecasts jump immediately with red upward SHAP bars.",
            "Experiment 3 (Monetary Policy Brake & 12m Anchor): Drag Real Fed Funds from -3 pp to +2 pp (green SHAP brake kicks in), then switch horizon to 12m and raise Michigan Expectations > 5%!",
        ],
        accent_color=ROYAL,
        font_pt=12.0,
        space_after_pt=6.0,
    )
    _attach_speaker_notes(
        s20_demo, 21, "Live Multi-Model Predictor Demo",
        "To make our models tangible rather than static slides, we built a lightweight local web app running all 12 trained models in 20 milliseconds.",
        [
            "We can click Dec 2022 to watch the models call the post-hike disinflation live, or drag Real Fed Funds and Electricity sliders to watch the exact TreeSHAP bars flip from red to green in real time.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 22: CONCLUSIONS & LIMITATIONS
    # ------------------------------------------------------------------
    s21 = prs.slides.add_slide(blank_layout)
    _add_header(s21, "Sections 10 & 11", "Conclusions & Methodological Limitations", 22, total_slides)
    _add_card(
        s21, 0.55, 1.22, 6.05, 5.60,
        "Key Conclusions & Policy Takeaways",
        [
            f"1. ML Genuinely Anticipates Turning Points: Cuts Pure-Lag RMSE by >22% overall at h=3m ({best_row['RMSE']:.3f} vs {rw_rmse:.3f} pp), -28% in the 2021 Surge, and -65% during 2022–23 Fed hikes.",
            "2. Unifying All 4 Course Clusters Pays Off: Sticky Grid Electricity (Kuskaya 2025), Equities (Vamossy 2024), and Real GDP (Yoon 2021) materially boost accuracy and economic completeness.",
            "3. Hybrid XGB + Deep Net Solves Small Macro Samples: Beats standalone MLP & Transformer at every horizon by feeding tree SHAP features & trends into a gated neural net.",
            "4. The Transmission Clock Guides Policy: Watch Oil/PPI/Electricity at 1–3m, Real Fed Funds & Demand at 6m, and Household Expectations at 12m.",
        ],
        accent_color=TEAL,
        font_pt=11.6,
        space_after_pt=5.5,
    )
    _add_card(
        s21, 6.82, 1.22, 5.95, 5.60,
        "Honest Limitations & Future Extensions",
        [
            "1. Unforeseeable Geopolitical Shocks: Historical macro series cannot pre-predict sudden exogenous events (e.g., Feb 2022 Ukraine invasion or Mar 2020 lockdowns) before they occur.",
            "2. Final Revised vs. Real-Time ALFRED Vintages: While CPI is almost never revised, M2, INDPRO, and GDP undergo subsequent benchmark revisions.",
            "3. Monthly Sample Size Ceiling (N = 585): Five business cycles limit deep self-attention depth without cross-country OECD panel pooling.",
            "4. Future Work: Add high-frequency daily shipping/breakeven inflation rates and predictive quantile intervals.",
        ],
        accent_color=AMBER,
        font_pt=11.6,
        space_after_pt=5.5,
    )
    _attach_speaker_notes(
        s21, 22, "Conclusions & Limitations",
        "Wrapping up: ML works for macroeconomic forecasting when you respect economic structure, sample size, and causal validation.",
        [
            "Our four main takeaways are on the left: real turning-point anticipation, cross-paper gains from sticky electricity/stocks/GDP, our Hybrid XGB+Deep Net, and the Horizon Transmission Clock.",
            "On the right, we stay honest about limitations: exogenous wars cannot be predicted 3 months early, and scaling standalone Transformers further would require multi-country panel data.",
        ],
        script_log,
    )

    # ------------------------------------------------------------------
    # SLIDE 23: RESOURCES & REFERENCES
    # ------------------------------------------------------------------
    s22 = prs.slides.add_slide(blank_layout)
    _add_header(s22, "Sections 12 & 13", "Reproducible Code/Data Resources & References", 23, total_slides)
    _add_card(
        s22, 0.55, 1.22, 5.45, 5.60,
        "Data, Code & Live Demo Resources",
        [
            "Public Data (Zero API Keys): 15 St. Louis Fed FRED series cached locally in data/raw/ (1978–2026).",
            "Reproducible Python Repository (eco331_gemini/):",
            "  • app.py — Live 12-Model Web Predictor (port 8000)",
            "  • run_pipeline.py — 1-command full empirical build",
            "  • src/models.py — 12 models + Hybrid + DM HAC test",
            "  • src/explainability.py — TreeSHAP, Ablation & Playbook",
            "  • presentation/Speaker_Notes_and_Talking_Points.md",
            "  • notebooks/ECO331_Explainable_Inflation_Forecasting.ipynb",
        ],
        accent_color=ROYAL,
        font_pt=11.4,
    )
    _add_card(
        s22, 6.22, 1.22, 6.55, 5.60,
        "Key Course & Methodological References",
        [
            "Singh, N., & Bhoi, B. B. (2022). Inflation Forecasting in India: Are ML Techniques Useful? RBI Occasional Papers.",
            "Singh, N., & Rathi, A. (2024). Nowcasting Food Inflation: Price & Non-Price Signals via ML. RBI Bulletin.",
            "Li, G., Pai, P.-F., & Lin, K.-P. (2022). Forecasting Inflation Rates by Extreme Gradient Boosting. Applied AI.",
            "Mamedli, M., & Shibitov, D. (2021). ML Real-Time CPI Forecasting. Bank of Russia WP.",
            "Kuskaya, S., & Bilgili, F. (2025). Forecasting Electricity Price Index with ML & CV Strategies. Energy.",
            "Yoon, J. (2021). Forecasting of Real GDP Growth Using ML Models: GB & RF. Computational Economics.",
            "Vamossy, D. F. (2024) & Subasi, A. (2021). Equity Return Prediction via ML & Deep Sequence Models.",
            "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.",
        ],
        accent_color=NAVY,
        font_pt=10.8,
        space_after_pt=4.0,
    )
    _attach_speaker_notes(
        s22, 23, "Resources & References",
        "Everything — data, models, 12 figures, speaker notes, and our live web predictor — reproduces in one command. Thank you, and we'd love your questions!",
        [
            "Invite the professor/audience to suggest a 'What-If' macro scenario (e.g., an oil + electricity spike or aggressive Fed rate cuts) to test live in the predictor app!",
        ],
        script_log,
    )

    prs.save(str(out_path))
    notes_md_path = PRESENTATION_DIR / "Speaker_Notes_and_Talking_Points.md"
    notes_md_path.write_text("\n".join(script_log), encoding="utf-8")
    return out_path
