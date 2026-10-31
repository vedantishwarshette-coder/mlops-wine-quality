import argparse
import itertools
import json
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

DVC_FILES = {
    "wine": "data/raw/winequality-red.csv.dvc",
    "housing": "data/raw/housing.csv.dvc",
    "spam": "data/raw/SMSSpamCollection.dvc",
}

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(cfg["experiment"])


def data_hash():
    path = DVC_FILES.get(cfg["name"])
    if path and os.path.exists(path):
        return yaml.safe_load(open(path))["outs"][0]["md5"][:8]
    return "unknown"


train = pd.read_csv(cfg["data"]["train"])
test = pd.read_csv(cfg["data"]["test"])
data_tag = data_hash()
if args.extra:
    train = pd.concat([train, pd.read_csv(args.extra)], ignore_index=True)
    data_tag = data_tag + "+" + os.path.basename(args.extra)


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


best = {"value": -1e9}

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

                if text_col:
                    sig, example = None, X_train.head(3).tolist()
                else:
                    sig = infer_signature(X_train, model.predict(X_train))
                    example = X_train.head(3)
                mlflow.sklearn.log_model(
                    model, "model",
                    signature=sig,
                    input_example=example,
                    serialization_format="cloudpickle",
                )

                value = scores[cfg["metric"]]
                print(model_type, params, cfg["metric"], "=", round(value, 3))
                if value > best["value"]:
                    best = {
                        "metric": cfg["metric"],
                        "value": round(value, 6),
                        "model_type": model_type,
                        "params": {k: str(v) for k, v in params.items()},
                        "data_version": data_tag,
                    }

# DVC metrics file (retraining ke time nahi badalte, taaki dvc status saaf rahe)
if not args.extra:
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{cfg['name']}.json", "w") as f:
        json.dump(best, f, indent=2)
    print("best:", best)

# Registration aur champion ka faisla sirf promote.py karta hai