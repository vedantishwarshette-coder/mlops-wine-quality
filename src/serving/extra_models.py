import os
import threading

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

router = APIRouter()
_cache = {}
_lock = threading.Lock()


def get_model(name):
    # pehli request pe champion load hota hai, import pe nahi (tests/CI na tootein)
    with _lock:
        if name not in _cache:
            try:
                _cache[name] = mlflow.sklearn.load_model(f"models:/{name}@champion")
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"model '{name}' load nahi hua: {exc}")
        return _cache[name]


HOUSING_COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                   "Population", "AveOccup", "Latitude", "Longitude"]


class HousingInput(BaseModel):
    MedInc: float = Field(ge=0, le=20)
    HouseAge: float = Field(ge=0, le=100)
    AveRooms: float = Field(ge=0, le=150)
    AveBedrms: float = Field(ge=0, le=50)
    Population: float = Field(ge=0, le=50000)
    AveOccup: float = Field(ge=0, le=1500)
    Latitude: float = Field(ge=32, le=43)
    Longitude: float = Field(ge=-125, le=-114)


class SpamInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@router.post("/predict/housing")
def predict_housing(item: HousingInput):
    model = get_model("housing-regressor")
    df = pd.DataFrame([[getattr(item, c) for c in HOUSING_COLUMNS]], columns=HOUSING_COLUMNS)
    value = float(model.predict(df)[0])
    return {"median_house_value_100k_usd": round(value, 3), "approx_usd": round(value * 100000)}


@router.post("/predict/spam")
def predict_spam(item: SpamInput):
    model = get_model("spam-detector")
    result = {"is_spam": bool(int(model.predict([item.text])[0]))}
    try:
        result["spam_probability"] = round(float(model.predict_proba([item.text])[0][1]), 3)
    except Exception:
        pass
    return result