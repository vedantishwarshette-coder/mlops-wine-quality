import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import f1_score, r2_score

parser = argparse.ArgumentParser()
parser.add_argument("since", nargs="?", type=int, default=0)  # start time (ms)
parser.add_argument("--config", default="configs/wine.yaml")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
NAME = cfg["model_name"]
METRIC = cfg["metric"]
MARGIN = cfg.get("margin", 0.005)
score_fn = {"f1": f1_score, "r2": r2_score}[METRIC]

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()

test = pd.read_csv(cfg["data"]["test"])
text_col = cfg["data"].get("text_column")
X_test = test[text_col] if text_col else test.drop(columns=[cfg["data"]["target"]])
y_test = test[cfg["data"]["target"]]

# 1. sabse achha run (parent runs me ye metric nahi hota, isliye wo chhoot jate hain)
exp = mlflow.get_experiment_by_name(cfg["experiment"])
runs = mlflow.search_runs(
    [exp.experiment_id],
    filter_string=f"attributes.start_time >= {args.since} and metrics.{METRIC} >= -1000",
    order_by=[f"metrics.{METRIC} DESC"],
    max_results=1,
)
if runs.empty:
    print("Koi naya run nahi mila")
    sys.exit(2)
best = runs.iloc[0]

# 2. register karo
mv = mlflow.register_model(f"runs:/{best.run_id}/model", NAME)
new_version = mv.version
cand = score_fn(y_test, mlflow.sklearn.load_model(f"models:/{NAME}/{new_version}").predict(X_test))

# 3. purane champion se compare karo
try:
    client.get_model_version_by_alias(NAME, "champion")
    champ = score_fn(y_test, mlflow.sklearn.load_model(f"models:/{NAME}@champion").predict(X_test))
    has_champ = True
except Exception:
    champ, has_champ = 0.0, False

if (not has_champ) or cand >= champ + MARGIN:
    client.set_registered_model_alias(NAME, "champion", new_version)
    decision = "promoted"
else:
    client.set_registered_model_alias(NAME, "challenger", new_version)
    decision = "rejected"

client.set_model_version_tag(NAME, new_version, "decision", decision)
client.set_model_version_tag(NAME, new_version, f"{METRIC}_diff", str(round(cand - champ, 4)))
print(f"candidate {METRIC}={cand:.3f} champion {METRIC}={champ:.3f} -> {decision}")