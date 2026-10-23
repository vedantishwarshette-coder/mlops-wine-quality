import os
import mlflow, mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
NAME = "wine-quality-classifier"
MARGIN = 0.005   # naya model itna behtar hona chahiye

client = MlflowClient()
test = pd.read_csv("data/processed/test.csv")
X_test, y_test = test.drop(columns=["good"]), test["good"]

# 1. sabse achha run dhundo
exp = mlflow.get_experiment_by_name("wine-quality")
best = mlflow.search_runs([exp.experiment_id], order_by=["metrics.f1 DESC"], max_results=1).iloc[0]

# 2. use register karo
mv = mlflow.register_model(f"runs:/{best.run_id}/model", NAME)
new_version = mv.version
cand_f1 = f1_score(y_test, mlflow.sklearn.load_model(f"models:/{NAME}/{new_version}").predict(X_test))

# 3. purane champion se compare karo
try:
    champ = client.get_model_version_by_alias(NAME, "champion")
    champ_f1 = f1_score(y_test, mlflow.sklearn.load_model(f"models:/{NAME}@champion").predict(X_test))
except Exception:
    champ, champ_f1 = None, 0.0

if champ is None or cand_f1 >= champ_f1 + MARGIN:
    client.set_registered_model_alias(NAME, "champion", new_version)
    decision = "promoted"
else:
    client.set_registered_model_alias(NAME, "challenger", new_version)
    decision = "rejected"

client.set_model_version_tag(NAME, new_version, "decision", decision)
client.set_model_version_tag(NAME, new_version, "f1_diff", str(round(cand_f1 - champ_f1, 4)))
print(f"candidate f1={cand_f1:.3f} champion f1={champ_f1:.3f} -> {decision}")