(() => {
  "use strict";

  const FAMILY_COLORS = {
    "1. Econometric Benchmark": "#64748b",
    "2. Regularized Linear": "#2563eb",
    "3. Tree Ensemble": "#0d9488",
    "4. Deep Learning": "#7c3aed",
    "5. Hybrid Tree + Deep Net": "#e11d48",
    "6. Cross-Family Ensemble": "#d97706",
  };

  let appMeta = null;
  let activePresetId = "latest_2026_09";
  let activeHorizon = 3;
  let debounceTimer = null;

  const formEl = document.getElementById("predictor-form");
  const presetBtnsContainer = document.getElementById("preset-buttons");
  const presetDescEl = document.getElementById("active-preset-desc");
  const horizonContainer = document.getElementById("horizon-options");
  const mainSlidersContainer = document.getElementById("main-slider-groups");
  const advSlidersContainer = document.getElementById("advanced-sliders");
  const resetBtn = document.getElementById("reset-btn");

  function formatSigned(val, suffix = " pp") {
    const sign = val > 0 ? "+" : "";
    return `${sign}${val.toFixed(2)}${suffix}`;
  }

  function syncSliderVisuals(inputEl, unit) {
    const val = Number(inputEl.value);
    inputEl.style.setProperty("--attr-min", inputEl.min || "0");
    inputEl.style.setProperty("--attr-max", inputEl.max || "100");
    inputEl.style.setProperty("--control-value", String(val));

    const formatted = `${val > 0 && unit.includes("pp") ? "+" : ""}${val.toFixed(
      Number(inputEl.step) >= 1 ? 0 : 1
    )}${unit}`;
    inputEl.setAttribute("aria-valuetext", formatted);

    const outEl = document.getElementById(`out-${inputEl.id}`);
    if (outEl) {
      outEl.textContent = formatted;
    }
  }

  function createSliderControl(spec) {
    const wrapper = document.createElement("div");
    wrapper.className = "slider-item";

    const labelRow = document.createElement("div");
    labelRow.className = "slider-label-row";

    const label = document.createElement("label");
    label.setAttribute("for", `slider-${spec.key}`);
    label.textContent = spec.label;

    const output = document.createElement("output");
    output.className = "slider-output";
    output.id = `out-slider-${spec.key}`;
    output.setAttribute("for", `slider-${spec.key}`);

    labelRow.append(label, output);

    const hint = document.createElement("span");
    hint.className = "slider-hint";
    hint.id = `hint-${spec.key}`;
    hint.textContent = spec.hint;

    const input = document.createElement("input");
    input.type = "range";
    input.id = `slider-${spec.key}`;
    input.name = spec.key;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = "0";
    input.dataset.unit = spec.unit;
    input.setAttribute("aria-describedby", `hint-${spec.key}`);

    input.addEventListener("input", () => {
      syncSliderVisuals(input, spec.unit);
      schedulePrediction();
    });

    wrapper.append(labelRow, hint, input);
    return wrapper;
  }

  function buildControls(meta) {
    // 1. Preset Pills
    presetBtnsContainer.replaceChildren();
    meta.presets.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.dataset.presetId = p.id;
      btn.setAttribute("aria-pressed", String(p.id === activePresetId));
      btn.textContent = p.title;
      btn.addEventListener("click", () => applyPreset(p.id));
      presetBtnsContainer.appendChild(btn);
    });

    // 2. Horizon Radio Buttons (< 6 exclusive options)
    horizonContainer.replaceChildren();
    meta.horizons.forEach((h) => {
      const wrap = document.createElement("div");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.id = `horizon-${h.value}`;
      radio.name = "horizon";
      radio.value = String(h.value);
      radio.className = "visually-hidden";
      radio.checked = h.value === activeHorizon;

      const lbl = document.createElement("label");
      lbl.setAttribute("for", `horizon-${h.value}`);
      lbl.className = "horizon-option-label";

      const st = document.createElement("strong");
      st.textContent = h.label;
      const sp = document.createElement("span");
      sp.textContent = h.tag;
      lbl.append(st, sp);

      radio.addEventListener("change", () => {
        if (radio.checked) {
          activeHorizon = h.value;
          runPrediction();
        }
      });

      wrap.append(radio, lbl);
      horizonContainer.appendChild(wrap);
    });

    // 3. Grouped Primary Sliders
    mainSlidersContainer.replaceChildren();
    meta.slider_groups.forEach((grp) => {
      const fs = document.createElement("fieldset");
      fs.className = "slider-fieldset";
      const lg = document.createElement("legend");
      lg.textContent = grp.title;
      fs.appendChild(lg);

      grp.sliders.forEach((s) => {
        fs.appendChild(createSliderControl(s));
      });
      mainSlidersContainer.appendChild(fs);
    });

    // 4. Secondary Advanced Sliders
    advSlidersContainer.replaceChildren();
    meta.advanced_sliders.forEach((s) => {
      advSlidersContainer.appendChild(createSliderControl(s));
    });
  }

  function applyPreset(presetId) {
    const preset = appMeta.presets.find((p) => p.id === presetId);
    if (!preset) return;
    activePresetId = preset.id;

    presetBtnsContainer.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(btn.dataset.presetId === activePresetId));
    });
    presetDescEl.textContent = `${preset.title}: ${preset.description}`;

    Object.entries(preset.values).forEach(([key, val]) => {
      const input = document.getElementById(`slider-${key}`);
      if (input) {
        input.value = String(val);
        syncSliderVisuals(input, input.dataset.unit || "%");
      }
    });

    runPrediction();
  }

  function collectOverrides() {
    const overrides = {};
    formEl.querySelectorAll('input[type="range"]').forEach((inp) => {
      overrides[inp.name] = Number(inp.value);
    });
    return overrides;
  }

  function schedulePrediction() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPrediction, 45);
  }

  function setDeltaChip(el, changePp) {
    el.textContent = formatSigned(changePp, " pp");
    el.className = "kpi-delta";
    if (changePp > 0.05) {
      el.classList.add("delta-up");
    } else if (changePp < -0.05) {
      el.classList.add("delta-down");
    } else {
      el.classList.add("delta-flat");
    }
  }

  function renderResults(res) {
    // 1. Hero KPIs
    document.getElementById("best-model-badge").textContent = `Recommended (${res.horizon}m Leader)`;
    document.getElementById("kpi-best-cpi").textContent = `${res.recommended_cpi.toFixed(2)}%`;
    setDeltaChip(document.getElementById("kpi-best-delta"), res.recommended_change_pp);
    document.getElementById("kpi-best-name").textContent = `${res.recommended_model} (lowest ${res.horizon}m test error)`;

    document.getElementById("kpi-consensus-cpi").textContent = `${res.consensus_cpi.toFixed(2)}%`;
    setDeltaChip(document.getElementById("kpi-consensus-delta"), res.consensus_change_pp);
    document.getElementById("kpi-consensus-range").textContent = `Model spread: ${res.consensus_min_cpi.toFixed(2)}% to ${res.consensus_max_cpi.toFixed(2)}%`;

    document.getElementById("kpi-anchor-cpi").textContent = `${res.current_cpi.toFixed(2)}%`;
    const actualBadge = document.getElementById("kpi-actual-badge");
    const refSub = document.getElementById("kpi-ref-sub");
    if (res.actual_realized_cpi !== null && res.actual_realized_cpi !== undefined) {
      actualBadge.hidden = false;
      actualBadge.textContent = `Actual ${res.horizon}m Later: ${res.actual_realized_cpi.toFixed(2)}%`;
      refSub.textContent = `Historical realized CPI ${res.horizon}m after ${res.preset_title.split(" (")[0]}`;
    } else {
      actualBadge.hidden = true;
      refSub.textContent = res.is_modified
        ? "Custom What-If scenario (Random Walk stays flat)"
        : "Forward live forecast (future CPI not yet realized)";
    }

    // 2. Visual 12-Model Horizontal Bar Comparison
    const barsChart = document.getElementById("model-bars-chart");
    barsChart.replaceChildren();

    const allVals = res.models.map((m) => m.predicted_cpi).concat([res.current_cpi]);
    const minScale = Math.min(0, Math.floor(Math.min(...allVals) - 0.8));
    const maxScale = Math.max(4.5, Math.ceil(Math.max(...allVals) + 0.8));
    const span = Math.max(1, maxScale - minScale);
    const anchorPct = ((res.current_cpi - minScale) / span) * 100;

    res.models.forEach((m) => {
      const row = document.createElement("div");
      row.className = m.is_recommended ? "bar-row is-top" : "bar-row";

      const nameEl = document.createElement("div");
      nameEl.className = "bar-model-label";
      nameEl.textContent = m.is_recommended ? `★ ${m.model}` : m.model;

      const track = document.createElement("div");
      track.className = "bar-track";

      const fill = document.createElement("div");
      fill.className = "bar-fill";
      const fillPct = Math.max(2, Math.min(100, ((m.predicted_cpi - minScale) / span) * 100));
      fill.style.inlineSize = `${fillPct}%`;
      fill.style.background = FAMILY_COLORS[m.family] || "#0d9488";

      const anchorLine = document.createElement("div");
      anchorLine.className = "bar-anchor-line";
      anchorLine.style.insetInlineStart = `${Math.max(0, Math.min(99.5, anchorPct))}%`;
      anchorLine.title = `Today's CPI: ${res.current_cpi.toFixed(2)}%`;

      track.append(fill, anchorLine);

      const valEl = document.createElement("div");
      valEl.className = "bar-value-text";
      valEl.textContent = `${m.predicted_cpi.toFixed(2)}% (${formatSigned(m.predicted_change_pp, "")})`;

      row.append(nameEl, track, valEl);
      barsChart.appendChild(row);
    });

    // 3. Live Exact XGBoost TreeSHAP Breakdown
    const shapSub = document.getElementById("shap-equation-sub");
    shapSub.textContent = `Today's CPI (${res.current_cpi.toFixed(2)}%) + XGBoost TreeSHAP Contributions = ${res.xgb_predicted_cpi.toFixed(2)}% (${res.horizon}m forecast)`;

    const shapContainer = document.getElementById("shap-bars-chart");
    shapContainer.replaceChildren();
    const maxAbsShap = Math.max(0.25, ...res.shap_drivers.map((d) => Math.abs(d.shap_pp)));

    res.shap_drivers.forEach((d) => {
      const row = document.createElement("div");
      row.className = "shap-row";

      const lbl = document.createElement("div");
      lbl.className = "shap-feat-name";
      lbl.textContent = `${d.label} `;
      const valSpan = document.createElement("span");
      valSpan.className = "shap-feat-val";
      valSpan.textContent = `[${d.feature_value}]`;
      lbl.appendChild(valSpan);

      const track = document.createElement("div");
      track.className = "shap-track";
      const zeroLine = document.createElement("div");
      zeroLine.className = "shap-zero-line";

      const bar = document.createElement("div");
      bar.className = "shap-bar";
      const halfWidthPct = Math.min(48, (Math.abs(d.shap_pp) / maxAbsShap) * 48);
      if (d.shap_pp >= 0) {
        bar.style.insetInlineStart = "50%";
        bar.style.inlineSize = `${halfWidthPct}%`;
        bar.style.background = "#ef4444";
      } else {
        bar.style.insetInlineStart = `${50 - halfWidthPct}%`;
        bar.style.inlineSize = `${halfWidthPct}%`;
        bar.style.background = "#0d9488";
      }
      track.append(zeroLine, bar);

      const badge = document.createElement("div");
      badge.className = "shap-val-badge";
      badge.style.color = d.shap_pp >= 0 ? "#b91c1c" : "#047857";
      badge.textContent = formatSigned(d.shap_pp, " pp");

      row.append(lbl, track, badge);
      shapContainer.appendChild(row);
    });

    // 4. Responsive Semantic Table
    const tbody = document.getElementById("model-table-body");
    tbody.replaceChildren();
    res.models.forEach((m) => {
      const tr = document.createElement("tr");
      if (m.is_recommended) tr.className = "top-row";

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = m.is_recommended ? `★ ${m.model}` : m.model;

      const tdFam = document.createElement("td");
      tdFam.textContent = m.family.replace(/^\d+\.\s*/, "");

      const tdPred = document.createElement("td");
      tdPred.textContent = `${m.predicted_cpi.toFixed(2)}%`;

      const tdDelta = document.createElement("td");
      tdDelta.textContent = formatSigned(m.predicted_change_pp, " pp");
      tdDelta.style.color = m.predicted_change_pp > 0.05 ? "#b91c1c" : m.predicted_change_pp < -0.05 ? "#047857" : "#334155";
      tdDelta.style.fontWeight = "600";

      const tdRmse = document.createElement("td");
      tdRmse.textContent = `${m.oos_rmse_pp.toFixed(3)} pp`;

      tr.append(th, tdFam, tdPred, tdDelta, tdRmse);
      tbody.appendChild(tr);
    });
  }

  async function runPrediction() {
    try {
      const payload = {
        horizon: activeHorizon,
        preset_id: activePresetId,
        overrides: collectOverrides(),
      };
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      renderResults(data);
    } catch (err) {
      console.error("Prediction request failed:", err);
    }
  }

  formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });

  resetBtn.addEventListener("click", () => {
    applyPreset(activePresetId);
  });

  async function init() {
    const resp = await fetch("/api/meta");
    appMeta = await resp.json();
    buildControls(appMeta);
    applyPreset(activePresetId);
  }

  init();
})();
