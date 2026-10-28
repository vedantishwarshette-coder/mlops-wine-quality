import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from src.serving.extra_models import router as extra_router

MODEL_NAME = "wine-quality-classifier"
EXPERIMENT = "wine-quality"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
client = MlflowClient()

COLUMNS = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
           "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
           "pH", "sulphates", "alcohol"]


class WineInput(BaseModel):
    fixed_acidity: float = Field(ge=3, le=20)
    volatile_acidity: float = Field(ge=0, le=2)
    citric_acid: float = Field(ge=0, le=2)
    residual_sugar: float = Field(ge=0, le=20)
    chlorides: float = Field(ge=0, le=1)
    free_sulfur_dioxide: float = Field(ge=0, le=100)
    total_sulfur_dioxide: float = Field(ge=0, le=350)
    density: float = Field(ge=0.98, le=1.1)
    pH: float = Field(ge=2, le=5)
    sulphates: float = Field(ge=0, le=3)
    alcohol: float = Field(ge=5, le=16)


app = FastAPI(title="Wine Quality Service")
app.include_router(extra_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: WineInput):
    df = pd.DataFrame([list(item.model_dump().values())], columns=COLUMNS)
    pred = int(model.predict(df)[0])
    result = {"good_quality": bool(pred)}
    try:
        result["confidence"] = float(max(model.predict_proba(df)[0]))
    except Exception:
        pass
    return result


@app.get("/model-info")
def model_info():
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)
    return {
        "name": MODEL_NAME,
        "version": mv.version,
        "alias": "champion",
        "run_id": mv.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": {k: v for k, v in run.data.tags.items()
                 if k in ("git_commit", "model_type", "author")},
    }


def _value(row, col):
    v = row.get(col)
    return None if pd.isna(v) else v


@app.get("/runs")
def runs():
    df = mlflow.search_runs(experiment_names=[EXPERIMENT],
                            order_by=["metrics.f1 DESC"], max_results=15)
    out = []
    for _, r in df.iterrows():
        out.append({
            "run_id": r["run_id"][:8],
            "model_type": _value(r, "params.model_type"),
            "f1": _value(r, "metrics.f1"),
            "accuracy": _value(r, "metrics.accuracy"),
            "C": _value(r, "params.C"),
            "n_estimators": _value(r, "params.n_estimators"),
            "max_depth": _value(r, "params.max_depth"),
        })
    return out


DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wine Quality Dashboard</title>
<style>
  :root { --bg:#f6f5f2; --card:#ffffff; --text:#1d1d1f; --muted:#6b6b70; --line:#e3e1dc;
          --accent:#7b1e3a; --good:#1e7b4a; --bad:#b3261e; color-scheme: light; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161416; --card:#201d20; --text:#f2f0f0; --muted:#a09aa0; --line:#34303a;
            --accent:#d46a8a; --good:#5fd394; --bad:#ff8a80; color-scheme: dark; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; }
  h1 { font-size:20px; margin:0; }
  .status { font-size:13px; color:var(--muted); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--bad); margin-right:6px; }
  .dot.ok { background:var(--good); }
  main { max-width:1100px; margin:0 auto; padding:24px; display:grid; gap:20px;
         grid-template-columns: 1fr 1fr; }
  @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }
  .card.wide { grid-column: 1 / -1; }
  h2 { font-size:15px; margin:0 0 14px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
  label { font-size:12px; color:var(--muted); display:block; margin-bottom:4px; }
  input { width:100%; padding:8px 10px; border-radius:8px; border:1px solid var(--line);
          background:var(--bg); color:var(--text); font-size:14px; }
  button { margin-top:16px; padding:10px 18px; border:0; border-radius:8px; background:var(--accent);
           color:#fff; font-size:14px; cursor:pointer; }
  button:hover { opacity:.9; }
  #result { margin-top:16px; font-size:18px; font-weight:600; min-height:28px; }
  .good { color:var(--good); } .bad { color:var(--bad); }
  .kpis { display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:14px; }
  .kpi { border:1px solid var(--line); border-radius:10px; padding:12px; }
  .kpi b { display:block; font-size:20px; margin-top:2px; }
  .kpi span { font-size:12px; color:var(--muted); }
  .meta { font-size:13px; color:var(--muted); line-height:1.7; }
  .tablewrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:500; }
  tr.top td { font-weight:600; }
</style>
</head>
<body>
<header>
  <h1>Wine Quality Dashboard</h1>
  <div class="status"><span id="dot" class="dot"></span><span id="statusText">checking...</span>
    &nbsp;|&nbsp; <a href="/docs" style="color:var(--accent)">API docs</a></div>
</header>
<main>
  <section class="card">
    <h2>Try a prediction</h2>
    <div class="grid" id="form"></div>
    <button id="go">Predict quality</button>
    <div id="result"></div>
  </section>

  <section class="card">
    <h2>Champion model</h2>
    <div class="kpis">
      <div class="kpi"><span>Version</span><b id="kVer">-</b></div>
      <div class="kpi"><span>F1</span><b id="kF1">-</b></div>
      <div class="kpi"><span>Accuracy</span><b id="kAcc">-</b></div>
    </div>
    <div class="meta" id="meta">Loading...</div>
  </section>

  <section class="card wide">
    <h2>Experiment runs (best F1 first)</h2>
    <div class="tablewrap"><table id="runs"></table></div>
  </section>
</main>

<script>
const FIELDS = [
  ["fixed_acidity","Fixed acidity",7.4], ["volatile_acidity","Volatile acidity",0.7],
  ["citric_acid","Citric acid",0], ["residual_sugar","Residual sugar",1.9],
  ["chlorides","Chlorides",0.076], ["free_sulfur_dioxide","Free sulfur dioxide",11],
  ["total_sulfur_dioxide","Total sulfur dioxide",34], ["density","Density",0.9978],
  ["pH","pH",3.51], ["sulphates","Sulphates",0.56], ["alcohol","Alcohol",9.4]
];

const form = document.getElementById("form");
FIELDS.forEach(([key, label, val]) => {
  const d = document.createElement("div");
  d.innerHTML = '<label for="' + key + '">' + label + '</label>' +
                '<input id="' + key + '" type="number" step="any" value="' + val + '">';
  form.appendChild(d);
});

const fmt = (v) => (v === null || v === undefined) ? "-" : (typeof v === "number" ? v.toFixed(3) : v);

document.getElementById("go").onclick = async () => {
  const body = {};
  FIELDS.forEach(([key]) => body[key] = Number(document.getElementById(key).value));
  const out = document.getElementById("result");
  out.className = ""; out.textContent = "Predicting...";
  try {
    const r = await fetch("/predict", { method:"POST", headers:{"Content-Type":"application/json"},
                                        body: JSON.stringify(body) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    const conf = d.confidence ? " (" + (d.confidence * 100).toFixed(1) + "% confidence)" : "";
    out.className = d.good_quality ? "good" : "bad";
    out.textContent = (d.good_quality ? "Good quality wine" : "Not good quality") + conf;
  } catch (e) {
    out.className = "bad"; out.textContent = "Error: " + e.message;
  }
};

async function loadHealth() {
  try {
    const r = await fetch("/health"); const d = await r.json();
    document.getElementById("dot").className = "dot ok";
    document.getElementById("statusText").textContent = "API " + d.status;
  } catch (e) { document.getElementById("statusText").textContent = "API offline"; }
}

async function loadInfo() {
  try {
    const d = await (await fetch("/model-info")).json();
    document.getElementById("kVer").textContent = "v" + d.version;
    document.getElementById("kF1").textContent = fmt(d.metrics.f1);
    document.getElementById("kAcc").textContent = fmt(d.metrics.accuracy);
    const t = d.tags || {};
    document.getElementById("meta").innerHTML =
      "Name: " + d.name + "<br>Alias: " + d.alias + "<br>Type: " + (t.model_type || "-") +
      "<br>Git commit: " + (t.git_commit || "-") + "<br>Run: " + d.run_id.slice(0, 8);
  } catch (e) { document.getElementById("meta").textContent = "Could not load model info."; }
}

async function loadRuns() {
  const t = document.getElementById("runs");
  try {
    const rows = await (await fetch("/runs")).json();
    let html = "<tr><th>Run</th><th>Model</th><th>F1</th><th>Accuracy</th><th>C</th><th>Trees</th><th>Max depth</th></tr>";
    rows.forEach((r, i) => {
      html += "<tr" + (i === 0 ? ' class="top"' : "") + "><td>" + r.run_id + "</td><td>" + (r.model_type || "-") +
        "</td><td>" + fmt(r.f1) + "</td><td>" + fmt(r.accuracy) + "</td><td>" + (r.C || "-") +
        "</td><td>" + (r.n_estimators || "-") + "</td><td>" + (r.max_depth || "-") + "</td></tr>";
    });
    t.innerHTML = html;
  } catch (e) { t.innerHTML = "<tr><td>Could not load runs.</td></tr>"; }
}

loadHealth(); loadInfo(); loadRuns();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD