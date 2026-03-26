import json
from pathlib import Path

# Import the engine and HTML string
from app import engine, HTML_TEMPLATE

print("⏳ Training model & computing explanations for static snapshot …\n")
engine.train()
engine.train_gam()

data = {
    "metrics": engine.metrics,
    "gam_plot": engine.plot_gam_b64(),
    "shap_summary": engine.shap_summary_b64(),
    "shap_waterfall": engine.shap_waterfall_b64()[0],
    "shap_waterfall_idx": engine.shap_waterfall_b64()[1],
}

Path("data.json").write_text(json.dumps(data))

# Transform the HTML to work statically
static_html = HTML_TEMPLATE.replace('fetch("/api/dashboard")', 'fetch("data.json")')

# Disable the predict form functionality for static hosting
static_html = static_html.replace(
    'const res = await fetch("/api/predict", {',
    '''
  const box = document.getElementById("resultBox");
  box.classList.add("show");
  document.getElementById("resultIcon").textContent = "ℹ️";
  document.getElementById("resultLabel").textContent = "Static Demo";
  document.getElementById("resultLabel").style.color = "var(--ambed)";
  document.getElementById("resultProb").textContent = "Live prediction is disabled in this GitHub Pages static demo. Run locally or on Render to use the inference engine.";
  const fill = document.getElementById("resultFill");
  fill.style.width = "0%";
  return;
  // Ignore the rest
  const res = await fetch("/api/predict", {
'''
)

Path("index.html").write_text(static_html, encoding="utf-8")
print("✅ Static site built (index.html + data.json)")
