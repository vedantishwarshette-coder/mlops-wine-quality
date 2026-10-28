import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "wine-quality-classifier"
COLUMNS = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
           "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
           "pH", "sulphates", "alcohol"]

SAMPLE = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}


@pytest.fixture(scope="session")
def api_client(tmp_path_factory):
    """Builds a throw-away MLflow store, registers a tiny champion model,
    then imports the real FastAPI app against it. No MLflow server or DVC data needed."""
    root = tmp_path_factory.mktemp("mlflow_test")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("wine-quality", artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment("wine-quality")

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((200, len(COLUMNS))), columns=COLUMNS)
    y = (X["alcohol"] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    pred = model.predict(X)

    with mlflow.start_run():
        mlflow.log_params({"model_type": "random_forest", "n_estimators": "10"})
        mlflow.log_metric("f1", f1_score(y, pred))
        mlflow.log_metric("accuracy", accuracy_score(y, pred))
        mlflow.set_tags({"git_commit": "test", "model_type": "random_forest", "author": "ci"})
        mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME,
                                 serialization_format="cloudpickle")

    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", 1)

    from fastapi.testclient import TestClient

    from src.serving.app import app
    return TestClient(app)