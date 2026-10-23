import os
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("iris")

data = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

NAME = "iris-classifier"

with mlflow.start_run():
    params = {"n_estimators": 50, "max_depth": 3, "random_state": 42}
    model = RandomForestClassifier(**params).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    sig = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=sig,
                             input_example=X_train.head(3),
                             registered_model_name=NAME)
    print("accuracy =", acc)

# naya version ko champion banao
client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions(f"name='{NAME}'"))
client.set_registered_model_alias(NAME, "champion", latest)
print("champion = version", latest)