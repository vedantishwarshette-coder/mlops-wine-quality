"""G3 demo: register a deliberately worse model and show promote.py rejects it."""
import subprocess
import sys
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

CONFIG = "configs/wine.yaml"
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
target = cfg["data"]["target"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(cfg["experiment"])

train = pd.read_csv(cfg["data"]["train"])
test = pd.read_csv(cfg["data"]["test"])
X_train, y_train = train.drop(columns=[target]), train[target]
X_test, y_test = test.drop(columns=[target]), test[target]

start_ms = int(time.time() * 1000)

# useless model: always answers "bad quality" (0)
model = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)
pred = model.predict(X_test)

with mlflow.start_run():
    mlflow.log_params({"model_type": "dummy_always_bad"})
    mlflow.log_metric("f1", f1_score(y_test, pred, zero_division=0))
    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.set_tags({"purpose": "deliberately_worse", "config": "wine", "author": "vedant"})
    mlflow.sklearn.log_model(
        model, "model",
        signature=infer_signature(X_train, model.predict(X_train)),
        input_example=X_train.head(3),
        serialization_format="cloudpickle",
    )

# promote.py only looks at runs started after start_ms, i.e. just this one
subprocess.run([sys.executable, "src/registry/promote.py", str(start_ms), "--config", CONFIG], check=False)