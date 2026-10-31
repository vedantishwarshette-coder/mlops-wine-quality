"""Adds a description and tags to the registered models and their champion versions."""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

INFO = {
    "wine-quality-classifier": "Wine quality (good = quality 6 or higher). Random Forest, metric F1. Data: UCI red wine.",
    "housing-regressor": "California house price regressor. Gradient Boosting, metric R2. Data: sklearn California Housing.",
    "spam-detector": "SMS spam detector. TF-IDF + classifier as one pipeline, metric F1. Data: UCI SMS Spam Collection.",
}

for name, text in INFO.items():
    champ = client.get_model_version_by_alias(name, "champion")
    client.update_registered_model(name, description=text)
    client.update_model_version(name, champ.version, description="Champion. " + text)
    client.set_model_version_tag(name, champ.version, "approved_by", "promote.py")
    client.set_model_version_tag(name, champ.version, "stage_note", "serving")
    print(name, "v" + champ.version, "described")