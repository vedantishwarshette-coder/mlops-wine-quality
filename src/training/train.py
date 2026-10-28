import argparse
import itertools
import os
import subprocess
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/wine.yaml")
parser.add_argument("--extra", default=None, help="retraining ke liye naya labeled csv")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
task = cfg["task"]
target = cfg["data"]["target"]
text_col = cfg["data"].get("text_column")
nested = cfg.get("nested", False)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(cfg["experiment"])

train = pd.read_csv(cfg["data"]["train"])
test = pd.read_csv(cfg["data"]["test"])
data_tag = "original"
if args.extra:
    train = pd.concat([train, pd.read_csv(args.extra)], ignore_index=True)
    data_tag = os.path.basename(args.extra)


def split(df):
    X = df[text_col] if text_col else df.drop(columns=[target])
    return X, df[target]


X_train, y_train = split(train)
X_test, y_test = split(test)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def build_model(model_type, params):
    if model_type == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=42, **params)
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42, **params)
    if model_type == "tfidf_nb":
        return make_pipeline(TfidfVectorizer(), MultinomialNB(**params))
    if model_type == "tfidf_logreg":
        return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000, **params))
    raise ValueError(f"unknown model type: {model_type}")


def metrics_for(y_true, pred):
    if task == "classification":
        return {"f1": f1_score(y_true, pred), "accuracy": accuracy_score(y_true, pred)}
    return {
        "r2": r2_score(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": mean_absolute_error(y_true, pred),
    }


def save_plot(y_true, pred):
    if task == "classification":
        ConfusionMatrixDisplay.from_predictions(y_true, pred)
        path = "confusion_matrix.png"
    else:
        plt.scatter(pred, y_true - pred, s=6)
        plt.axhline(0, color="red")
        plt.xlabel("predicted")
        plt.ylabel("residual")
        path = "residuals.png"
    plt.savefig(path)
    plt.close()
    return path


for cand in cfg["candidates"]:
    model_type = cand["type"]
    grid = cand.get("grid", {})
    trials = ([dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
              if grid else [{}])

    parent = mlflow.start_run(run_name=f"search-{model_type}") if nested else nullcontext()
    with parent:
        for params in trials:
            with mlflow.start_run(nested=nested):
                model = build_model(model_type, params)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores = metrics_for(y_test, pred)

                mlflow.log_params({"model_type": model_type, **{k: str(v) for k, v in params.items()}})
                mlflow.log_metrics(scores)
                mlflow.set_tags({
                    "git_commit": git_commit(),
                    "model_type": model_type,
                    "author": "vedant",
                    "data_version": data_tag,
                    "config": cfg["name"],
                })
                mlflow.log_artifact(save_plot(y_test, pred))

                sig = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model, "model",
                    signature=sig,
                    input_example=X_train.head(3),
                    serialization_format="cloudpickle",
                )
                print(model_type, params, cfg["metric"], "=", round(scores[cfg["metric"]], 3))

# Registration aur champion ka faisla sirf promote.py karta hai