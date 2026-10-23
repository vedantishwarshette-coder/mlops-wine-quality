import os
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model("models:/wine-quality-classifier@champion")

COLUMNS = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
           "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
           "pH", "sulphates", "alcohol"]


class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: WineInput):
    df = pd.DataFrame([list(item.model_dump().values())], columns=COLUMNS)
    pred = int(model.predict(df)[0])
    return {"good_quality": bool(pred)}