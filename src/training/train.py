import os, subprocess, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "wine-quality-classifier"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("wine-quality")

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train.drop(columns=["good"]), train["good"]
X_test, y_test = test.drop(columns=["good"]), test["good"]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


experiments = []
for C in [0.01, 0.1, 1, 10]:
    experiments.append(("logreg", {"C": C}))
for n, d in itertools.product([50, 100, 200], [5, 10, None]):
    experiments.append(("random_forest", {"n_estimators": n, "max_depth": d}))

best = {"f1": -1, "run_id": None}

for model_type, params in experiments:
    with mlflow.start_run() as run:
        if model_type == "logreg":
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
        else:
            model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = f1_score(y_test, pred)

        mlflow.log_params({"model_type": model_type, **{k: str(v) for k, v in params.items()}})
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.set_tags({"git_commit": git_commit(), "model_type": model_type, "author": "vedant"})

        ConfusionMatrixDisplay.from_predictions(y_test, pred)
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        sig = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, "model",
            signature=sig,
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        print(model_type, params, "F1 =", round(f1, 3))

        if f1 > best["f1"]:
            best = {"f1": f1, "run_id": run.info.run_id}

# Register the best run and mark it as champion
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Registered {MODEL_NAME} v{mv.version} as champion (F1={best['f1']:.3f})")