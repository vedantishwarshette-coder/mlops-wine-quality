from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Iris MLOps API", description="API for Iris Flower Classification")

# Model load karne ke liye (hum sklearn ya mlflow se load kar sakte hain)
# Simple tarike ke liye hum ek sample model load karte hain ya train kar lete hain

import os
import mlflow.sklearn
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model("models:/iris-classifier@champion")

class IrisInput(BaseModel):
    sepal_length: float     
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Welcome to MLOps Iris Prediction API!"}

@app.post("/predict")
def predict(input_data: IrisInput):
    features = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    prediction = model.predict(features)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    result = species_map[int(prediction[0])]
    
    return {"predicted_species": result}