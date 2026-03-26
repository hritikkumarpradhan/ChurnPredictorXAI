"""
ChurnPredictorXAI — Web Dashboard

A Flask-powered localhost dashboard that trains the churn model,
generates all explainability artefacts, and serves them in a
premium dark-mode UI.

Usage:
    python app.py
    # → Open http://127.0.0.1:5000 in your browser
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template_string, request
from pygam import LogisticGAM, s
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  Core ML class (same logic as churn_predictor_xai.py, adapted
#  to return base-64 encoded PNGs for the web UI)
# ═══════════════════════════════════════════════════════════════════

class ChurnEngine:
    CONTINUOUS  = ["tenure", "monthly_charges", "total_charges"]
    CONTINUOUS_MODEL = ["tenure", "monthly_charges"]  # exclude total_charges from model
    CATEGORICAL = ["contract_type", "internet_service", "tech_support"]

    def __init__(self) -> None:
        self.pipeline = None
        self.preprocessor = None
        self.gam = None
        self.metrics: Dict[str, Any] = {}
        self.X_train = self.y_train = self.X_test = self.y_test = None
        self.df = None

    # ── data ───────────────────────────────────────────────────────
    @staticmethod
    def generate_data(n: int = 8000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        tenure = rng.integers(0, 73, size=n).astype(float)
        monthly = rng.uniform(20, 120, size=n)
        # total_charges: weakly correlated with tenure/monthly
        # but dominated by independent randomness so RF can't
        # use it as a proxy for tenure
        total = np.clip(
            rng.uniform(100, 5000, n)          # large random base
            + 0.3 * tenure * monthly           # weak signal
            + rng.normal(0, 500, n),           # heavy noise
            0, None,
        )
        contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.50, .25, .25])
        internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.40, .45, .15])
        tech     = rng.choice(["Yes", "No", "No internet"], n, p=[.30, .55, .15])
        # Very strong coefficients for crisp decision boundaries
        logit = (
            -0.5
            - 0.10 * tenure          # dominant tenure effect
            + 0.04 * monthly         # strong charges effect
            + 2.5 * (contract == "Month-to-month").astype(float)
            - 2.0 * (contract == "Two year").astype(float)
            + 1.0 * (internet == "Fiber optic").astype(float)
            - 0.8 * (tech == "Yes").astype(float)
        )
        churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        return pd.DataFrame(dict(tenure=tenure, monthly_charges=monthly,
                                  total_charges=total, contract_type=contract,
                                  internet_service=internet, tech_support=tech,
                                  churn=churn))

    # ── pipeline ───────────────────────────────────────────────────
    def _build(self) -> Pipeline:
        num = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler())])
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))])
        pre = ColumnTransformer([("num", num, self.CONTINUOUS_MODEL),
                                 ("cat", cat, self.CATEGORICAL)])
        return Pipeline([("preprocessor", pre),
                         ("classifier", RandomForestClassifier(
                             n_estimators=200, random_state=42,
                             max_depth=15,
                             class_weight="balanced"))])

    def train(self) -> None:
        self.df = self.generate_data()
        feature_cols = self.CONTINUOUS_MODEL + self.CATEGORICAL
        X, y = self.df[feature_cols], self.df["churn"]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
        self.pipeline = self._build()
        self.pipeline.fit(self.X_train, self.y_train)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        y_pred = self.pipeline.predict(self.X_test)
        y_prob = self.pipeline.predict_proba(self.X_test)[:, 1]
        self.metrics = dict(
            accuracy  = round(accuracy_score(self.y_test, y_pred), 4),
            precision = round(precision_score(self.y_test, y_pred), 4),
            recall    = round(recall_score(self.y_test, y_pred), 4),
            f1        = round(f1_score(self.y_test, y_pred), 4),
            roc_auc   = round(roc_auc_score(self.y_test, y_prob), 4),
        )
        self.metrics["confusion"] = confusion_matrix(self.y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        self.metrics["roc_fpr"] = fpr.tolist()
        self.metrics["roc_tpr"] = tpr.tolist()
        self.metrics["churn_rate"] = round(self.df["churn"].mean(), 4)
        self.metrics["n_samples"] = len(self.df)

    # ── feature names ──────────────────────────────────────────────
    def _feat_names(self):
        ohe = self.preprocessor.named_transformers_["cat"].named_steps["ohe"]
        return list(self.CONTINUOUS_MODEL) + list(ohe.get_feature_names_out(self.CATEGORICAL))

    def _transform(self, X):
        return self.preprocessor.transform(X)

    # ── GAM ────────────────────────────────────────────────────────
    def train_gam(self) -> None:
        Xt = self._transform(self.X_train)[:, :len(self.CONTINUOUS_MODEL)]
        terms = s(0)
        for i in range(1, len(self.CONTINUOUS_MODEL)):
            terms += s(i)
        self.gam = LogisticGAM(terms).gridsearch(Xt, self.y_train.values, progress=False)

    def plot_gam_b64(self) -> str:
        n = len(self.CONTINUOUS_MODEL)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1: axes = [axes]
        colors = ["#818cf8", "#34d399", "#fbbf24"]
        for i, (ax, fn) in enumerate(zip(axes, self.CONTINUOUS_MODEL)):
            XX = self.gam.generate_X_grid(term=i)
            pdep, ci = self.gam.partial_dependence(term=i, X=XX, width=.95)
            ax.plot(XX[:, i], pdep, color=colors[i], lw=2.5)
            ax.fill_between(XX[:, i], ci[:, 0], ci[:, 1], alpha=.18, color=colors[i])
            ax.set_title(fn.replace("_", " ").title(), fontsize=14, fontweight="bold", color="white")
            ax.set_xlabel(f"{fn} (scaled)", color="#94a3b8")
            ax.set_ylabel("log-odds", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.set_facecolor("#1e293b")
            for sp in ax.spines.values(): sp.set_color("#334155")
            ax.grid(True, alpha=.15, color="#475569")
        fig.patch.set_facecolor("#0f172a")
        fig.suptitle("pyGAM — Partial Dependence (95% CI)", fontsize=16,
                     fontweight="bold", color="white", y=1.02)
        plt.tight_layout()
        return self._fig_to_b64(fig)

    # ── SHAP ───────────────────────────────────────────────────────
    def shap_summary_b64(self) -> str:
        rf = self.pipeline.named_steps["classifier"]
        Xt = self._transform(self.X_test)
        explainer = shap.TreeExplainer(rf)
        sv = explainer(Xt)
        if sv.values.ndim == 3:
            vals = sv.values[:, :, 1]
        else:
            vals = sv.values
        fig = plt.figure(figsize=(10, 7))
        fig.patch.set_facecolor("#0f172a")
        shap.summary_plot(vals, Xt, feature_names=self._feat_names(), show=False)
        ax = plt.gca()
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.set_title("SHAP Global Feature Importance", color="white",
                      fontsize=14, fontweight="bold")
        ax.set_xlabel("SHAP value", color="#94a3b8")
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def shap_waterfall_b64(self) -> tuple[str, int]:
        rf = self.pipeline.named_steps["classifier"]
        Xt = self._transform(self.X_test)
        proba = rf.predict_proba(Xt)[:, 1]
        idx = int(np.argmax(proba))
        explainer = shap.TreeExplainer(rf)
        sv = explainer(Xt)
        fnames = self._feat_names()
        if sv.values.ndim == 3:
            expl = shap.Explanation(values=sv.values[idx, :, 1],
                                     base_values=sv.base_values[idx, 1] if sv.base_values.ndim==2 else sv.base_values[idx],
                                     data=sv.data[idx],
                                     feature_names=fnames)
        else:
            expl = sv[idx]
            expl.feature_names = fnames
        fig = plt.figure(figsize=(10, 7))
        fig.patch.set_facecolor("#0f172a")
        shap.waterfall_plot(expl, show=False)
        ax = plt.gca()
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        plt.tight_layout()
        return self._fig_to_b64(fig), idx

    # ── predict single customer ────────────────────────────────────
    def predict_single(self, data: dict) -> dict:
        # Only keep model features (drop total_charges if present)
        model_data = {k: data[k] for k in self.CONTINUOUS_MODEL + self.CATEGORICAL if k in data}
        row = pd.DataFrame([model_data])
        prob = self.pipeline.predict_proba(row)[0, 1]
        pred = int(prob >= 0.5)
        return {"churn_probability": round(float(prob), 4),
                "prediction": "Churn" if pred else "Retained"}

    # ── utility ────────────────────────────────────────────────────
    @staticmethod
    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════════════════
#  Flask App
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
engine = ChurnEngine()

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ChurnPredictorXAI Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── reset & base ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #0f172a;
  --surface:   #1e293b;
  --border:    #334155;
  --muted:     #64748b;
  --text:      #e2e8f0;
  --heading:   #f8fafc;
  --accent:    #818cf8;
  --green:     #34d399;
  --amber:     #fbbf24;
  --red:       #f87171;
  --cyan:      #22d3ee;
  --radius:    14px;
}
html { scroll-behavior: smooth; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

/* ── header ────────────────────────────────────────────────── */
.hero {
  position: relative;
  overflow: hidden;
  padding: 56px 0 40px;
  text-align: center;
  background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #064e3b 100%);
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 30% 20%, rgba(129,140,248,.12) 0%, transparent 60%),
              radial-gradient(circle at 70% 80%, rgba(52,211,153,.08) 0%, transparent 60%);
}
.hero h1 {
  position: relative;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -.02em;
  background: linear-gradient(135deg, #818cf8, #34d399);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero p { position: relative; color: var(--muted); margin-top: 8px; font-size: 1.05rem; }
.badge {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 999px;
  font-size: .75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .05em;
  margin-top: 12px;
  position: relative;
}
.badge--ok  { background: rgba(52,211,153,.15); color: var(--green); }
.badge--run { background: rgba(251,191,36,.15); color: var(--amber); }

/* ── layout ────────────────────────────────────────────────── */
.container { max-width: 1260px; margin: 0 auto; padding: 0 24px; }
.grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 20px; margin-top: 32px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 32px; }
@media(max-width:800px){ .grid-2{ grid-template-columns:1fr; } }

/* ── cards ─────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  transition: transform .2s, box-shadow .2s;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,.35); }
.card__label { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.card__value { font-size: 2rem; font-weight: 800; margin-top: 6px; }
.card__value--accent { color: var(--accent); }
.card__value--green  { color: var(--green);  }
.card__value--amber  { color: var(--amber);  }
.card__value--red    { color: var(--red);    }
.card__value--cyan   { color: var(--cyan);   }

/* ── section titles ────────────────────────────────────────── */
.section { margin-top: 56px; }
.section__title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--heading);
  display: flex;
  align-items: center;
  gap: 10px;
}
.section__title .dot {
  width: 10px; height: 10px; border-radius: 50%;
  display: inline-block;
}
.dot--purple { background: var(--accent); }
.dot--green  { background: var(--green);  }
.dot--amber  { background: var(--amber);  }
.section__sub { color: var(--muted); margin-top: 4px; font-size: .92rem; }

/* ── plot panels ───────────────────────────────────────────── */
.plot-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-top: 24px;
}
.plot-panel img { width: 100%; display: block; }
.plot-panel__caption {
  padding: 14px 20px;
  font-size: .85rem;
  color: var(--muted);
  border-top: 1px solid var(--border);
}

/* ── confusion matrix ──────────────────────────────────────── */
.cm-table { width: 100%; border-collapse: collapse; margin-top: 16px; }
.cm-table th, .cm-table td {
  padding: 14px;
  text-align: center;
  border: 1px solid var(--border);
  font-size: .95rem;
}
.cm-table th {
  background: rgba(129,140,248,.08);
  color: var(--accent);
  font-weight: 600;
  text-transform: uppercase;
  font-size: .75rem;
  letter-spacing: .05em;
}
.cm-table td { font-weight: 700; font-size: 1.15rem; }
.cm-tp { background: rgba(52,211,153,.12); color: var(--green); }
.cm-tn { background: rgba(129,140,248,.12); color: var(--accent); }
.cm-fp { background: rgba(251,191,36,.10); color: var(--amber); }
.cm-fn { background: rgba(248,113,113,.10); color: var(--red); }

/* ── predict form ──────────────────────────────────────────── */
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }
@media(max-width:600px){ .form-grid{ grid-template-columns:1fr; } }
.form-group label {
  display: block;
  font-size: .78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: var(--muted);
  margin-bottom: 6px;
}
.form-group input, .form-group select {
  width: 100%;
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  font-family: inherit;
  font-size: .95rem;
  transition: border-color .2s;
}
.form-group input:focus, .form-group select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(129,140,248,.15);
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 20px;
  padding: 12px 32px;
  border: none;
  border-radius: 10px;
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform .15s, box-shadow .15s;
}
.btn--primary {
  background: linear-gradient(135deg, #818cf8, #6366f1);
  color: #fff;
}
.btn--primary:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(99,102,241,.35); }
.result-box {
  margin-top: 20px;
  padding: 20px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--bg);
  display: none;
}
.result-box.show { display: block; animation: fadeIn .4s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ── nav ───────────────────────────────────────────────────── */
.nav {
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(15,23,42,.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: center;
  gap: 28px;
  padding: 12px 0;
}
.nav a {
  color: var(--muted);
  text-decoration: none;
  font-size: .85rem;
  font-weight: 600;
  letter-spacing: .02em;
  transition: color .2s;
}
.nav a:hover { color: var(--accent); }

/* ── loading ───────────────────────────────────────────────── */
.loader-wrap {
  position: fixed; inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--bg);
  z-index: 9999;
  transition: opacity .5s;
}
.loader-wrap.hide { opacity: 0; pointer-events: none; }
.spinner {
  width: 48px; height: 48px;
  border: 4px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-wrap p { margin-top: 16px; color: var(--muted); font-size: .95rem; }

/* ── footer ────────────────────────────────────────────────── */
footer {
  margin-top: 80px;
  padding: 28px 0;
  text-align: center;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: .82rem;
}
</style>
</head>
<body>

<!-- loading screen -->
<div class="loader-wrap" id="loader">
  <div class="spinner"></div>
  <p>Training model & computing explanations …</p>
</div>

<div id="app" style="display:none;">

<!-- nav -->
<nav class="nav">
  <a href="#metrics">Metrics</a>
  <a href="#gam">GAM</a>
  <a href="#shap">SHAP</a>
  <a href="#predict">Predict</a>
</nav>

<!-- hero -->
<header class="hero">
  <h1>ChurnPredictor<span style="-webkit-text-fill-color:#34d399">XAI</span></h1>
  <p>Random Forest · SHAP · pyGAM — Explainable Churn Intelligence</p>
  <span class="badge badge--ok" id="statusBadge">● Model Ready</span>
</header>

<main class="container">

  <!-- ── metrics cards ──────────────────────────────────────── -->
  <section class="section" id="metrics">
    <div class="section__title"><span class="dot dot--purple"></span> Model Performance</div>
    <p class="section__sub">Random Forest (balanced) evaluated on 25 % hold-out test set</p>
    <div class="grid-4" id="metricCards"></div>

    <div class="grid-2" style="margin-top:28px;">
      <!-- confusion matrix -->
      <div class="card" id="cmCard"></div>
      <!-- ROC curve (canvas) -->
      <div class="card">
        <div class="card__label">ROC Curve</div>
        <canvas id="rocCanvas" style="width:100%;margin-top:12px;border-radius:10px;"></canvas>
      </div>
    </div>
  </section>

  <!-- ── GAM ────────────────────────────────────────────────── -->
  <section class="section" id="gam">
    <div class="section__title"><span class="dot dot--green"></span> pyGAM — Non-linear Trends</div>
    <p class="section__sub">LogisticGAM partial-dependence term plots with 95 % confidence intervals</p>
    <div class="plot-panel" id="gamPanel"></div>
  </section>

  <!-- ── SHAP ───────────────────────────────────────────────── -->
  <section class="section" id="shap">
    <div class="section__title"><span class="dot dot--amber"></span> SHAP — Feature Explanations</div>
    <p class="section__sub">TreeExplainer on the Random Forest (churn class)</p>
    <div class="grid-2">
      <div>
        <div class="plot-panel" id="shapSummaryPanel"></div>
      </div>
      <div>
        <div class="plot-panel" id="shapWaterfallPanel"></div>
      </div>
    </div>
  </section>

  <!-- ── predict ────────────────────────────────────────────── -->
  <section class="section" id="predict">
    <div class="section__title"><span class="dot dot--purple"></span> Predict a Customer</div>
    <p class="section__sub">Enter customer details and get an instant churn probability</p>
    <div class="card" style="margin-top:24px;">
      <form id="predictForm">
        <div class="form-grid">
          <div class="form-group">
            <label>Tenure (months)</label>
            <input type="number" name="tenure" value="6" min="0" max="72" required>
          </div>
          <div class="form-group">
            <label>Monthly Charges ($)</label>
            <input type="number" step="0.01" name="monthly_charges" value="85.0" min="20" max="120" required>
          </div>
          <div class="form-group">
            <label>Total Charges ($)</label>
            <input type="number" step="0.01" name="total_charges" value="510.0" min="0" required>
          </div>
          <div class="form-group">
            <label>Contract Type</label>
            <select name="contract_type">
              <option value="Month-to-month" selected>Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>
          <div class="form-group">
            <label>Internet Service</label>
            <select name="internet_service">
              <option value="DSL">DSL</option>
              <option value="Fiber optic" selected>Fiber optic</option>
              <option value="No">No</option>
            </select>
          </div>
          <div class="form-group">
            <label>Tech Support</label>
            <select name="tech_support">
              <option value="Yes">Yes</option>
              <option value="No" selected>No</option>
              <option value="No internet">No internet</option>
            </select>
          </div>
        </div>
        <button type="submit" class="btn btn--primary">⚡ Predict Churn</button>
      </form>

      <div class="result-box" id="resultBox">
        <div style="display:flex;align-items:center;gap:16px;">
          <div id="resultIcon" style="font-size:2.6rem;"></div>
          <div>
            <div id="resultLabel" style="font-size:1.3rem;font-weight:700;"></div>
            <div id="resultProb"  style="color:var(--muted);font-size:.95rem;margin-top:2px;"></div>
          </div>
        </div>
        <div id="resultBar" style="margin-top:16px;height:8px;border-radius:99px;background:var(--border);overflow:hidden;">
          <div id="resultFill" style="height:100%;border-radius:99px;transition:width .6s ease;"></div>
        </div>
      </div>
    </div>
  </section>
</main>

<footer>ChurnPredictorXAI · Built with scikit-learn · SHAP · pyGAM · Flask</footer>
</div>

<script>
// ── boot ───────────────────────────────────────────────────────
(async () => {
  const res = await fetch("/api/dashboard");
  const d   = await res.json();
  renderMetrics(d.metrics);
  renderCM(d.metrics.confusion);
  renderROC(d.metrics.roc_fpr, d.metrics.roc_tpr, d.metrics.roc_auc);
  renderPlot("gamPanel", d.gam_plot, "Partial dependence of each continuous feature on churn log-odds");
  renderPlot("shapSummaryPanel", d.shap_summary, "Global SHAP feature importance (bee-swarm)");
  renderPlot("shapWaterfallPanel", d.shap_waterfall, `Local explanation for highest-risk customer (#${d.shap_waterfall_idx})`);
  document.getElementById("loader").classList.add("hide");
  document.getElementById("app").style.display = "";
})();

// ── metrics cards ──────────────────────────────────────────────
function renderMetrics(m) {
  const defs = [
    ["Accuracy",  m.accuracy,  "--accent"],
    ["Precision", m.precision, "--green"],
    ["Recall",    m.recall,    "--amber"],
    ["F1-Score",  m.f1,        "--cyan"],
    ["ROC-AUC",   m.roc_auc,   "--accent"],
    ["Churn Rate", (m.churn_rate*100).toFixed(1)+"%", "--red"],
    ["Train Size", Math.round(m.n_samples*.75), "--green"],
    ["Test Size",  Math.round(m.n_samples*.25), "--amber"],
  ];
  const wrap = document.getElementById("metricCards");
  defs.forEach(([lbl, val, clr]) => {
    const c = document.createElement("div");
    c.className = "card";
    c.innerHTML = `<div class="card__label">${lbl}</div>
                   <div class="card__value" style="color:var(${clr})">${val}</div>`;
    wrap.appendChild(c);
  });
}

// ── confusion matrix ───────────────────────────────────────────
function renderCM(cm) {
  document.getElementById("cmCard").innerHTML = `
    <div class="card__label">Confusion Matrix</div>
    <table class="cm-table">
      <tr><th></th><th>Pred Retained</th><th>Pred Churned</th></tr>
      <tr><th>Actual Retained</th><td class="cm-tn">${cm[0][0]}</td><td class="cm-fp">${cm[0][1]}</td></tr>
      <tr><th>Actual Churned</th><td class="cm-fn">${cm[1][0]}</td><td class="cm-tp">${cm[1][1]}</td></tr>
    </table>`;
}

// ── ROC curve (canvas) ─────────────────────────────────────────
function renderROC(fpr, tpr, auc) {
  const canvas = document.getElementById("rocCanvas");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = w * 0.75;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const pad = 44;
  const pw = w - 2*pad, ph = h - 2*pad;

  ctx.fillStyle = "#1e293b"; ctx.fillRect(0,0,w,h);

  // grid
  ctx.strokeStyle = "#334155"; ctx.lineWidth = .5;
  for(let i=0;i<=4;i++){
    const x=pad+pw*i/4, y=pad+ph*i/4;
    ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,pad+ph);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+pw,y);ctx.stroke();
  }

  // diagonal
  ctx.setLineDash([6,4]); ctx.strokeStyle="#475569"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad+ph); ctx.lineTo(pad+pw,pad); ctx.stroke();
  ctx.setLineDash([]);

  // curve
  ctx.strokeStyle = "#818cf8"; ctx.lineWidth = 2.5; ctx.beginPath();
  fpr.forEach((f,i) => {
    const x = pad + f*pw, y = pad + ph - tpr[i]*ph;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();

  // fill
  ctx.globalAlpha = .1; ctx.fillStyle = "#818cf8"; ctx.beginPath();
  fpr.forEach((f,i)=>{const x=pad+f*pw,y=pad+ph-tpr[i]*ph;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.lineTo(pad+pw,pad+ph);ctx.lineTo(pad,pad+ph);ctx.closePath();ctx.fill();ctx.globalAlpha=1;

  // labels
  ctx.fillStyle="#94a3b8";ctx.font="600 11px Inter,sans-serif";ctx.textAlign="center";
  ctx.fillText("False Positive Rate",pad+pw/2,h-6);
  ctx.save();ctx.translate(12,pad+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText("True Positive Rate",0,0);ctx.restore();

  // AUC badge
  ctx.fillStyle="rgba(129,140,248,.15)";
  const tx=pad+pw-90, ty=pad+ph-30;
  ctx.beginPath();ctx.roundRect(tx,ty,86,24,6);ctx.fill();
  ctx.fillStyle="#818cf8";ctx.font="700 12px Inter,sans-serif";ctx.textAlign="center";
  ctx.fillText(`AUC = ${auc}`,tx+43,ty+16);
}

// ── plots ──────────────────────────────────────────────────────
function renderPlot(id, b64, caption) {
  document.getElementById(id).innerHTML =
    `<img src="data:image/png;base64,${b64}" alt="${caption}">
     <div class="plot-panel__caption">${caption}</div>`;
}

// ── predict form ───────────────────────────────────────────────
document.getElementById("predictForm").addEventListener("submit", async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const data = Object.fromEntries(fd.entries());
  ["tenure","monthly_charges","total_charges"].forEach(k => data[k] = parseFloat(data[k]));
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(data),
  });
  const r = await res.json();
  const prob = r.churn_probability;
  const isChurn = r.prediction === "Churn";
  const box = document.getElementById("resultBox");
  box.classList.add("show");
  document.getElementById("resultIcon").textContent = isChurn ? "🔴" : "🟢";
  document.getElementById("resultLabel").textContent = r.prediction;
  document.getElementById("resultLabel").style.color = isChurn ? "var(--red)" : "var(--green)";
  document.getElementById("resultProb").textContent = `Churn probability: ${(prob*100).toFixed(1)}%`;
  const fill = document.getElementById("resultFill");
  fill.style.width = (prob*100)+"%";
  fill.style.background = isChurn
    ? "linear-gradient(90deg,#f87171,#ef4444)"
    : "linear-gradient(90deg,#34d399,#10b981)";
});
</script>
</body>
</html>
"""

# ── routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/dashboard")
def dashboard():
    """Return all pre-computed data as JSON (metrics + base-64 plots)."""
    return jsonify(
        metrics=engine.metrics,
        gam_plot=engine.plot_gam_b64(),
        shap_summary=engine.shap_summary_b64(),
        shap_waterfall=engine.shap_waterfall_b64()[0],
        shap_waterfall_idx=engine.shap_waterfall_b64()[1],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    return jsonify(engine.predict_single(data))


# ── boot ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("━" * 56)
    print("  ChurnPredictorXAI — Web Dashboard")
    print("━" * 56)
    print("\n⏳ Training model & computing explanations …\n")

    engine.train()
    print("  ✅ Random Forest trained")

    engine.train_gam()
    print("  ✅ LogisticGAM fitted")

    print("  ✅ All ready\n")
    print("━" * 56)
    print("  🌐  Open → http://127.0.0.1:5000")
    print("━" * 56)

    app.run(host="127.0.0.1", port=5000, debug=False)
