import json
import os
import threading
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from mlflow import MlflowClient
from pydantic import BaseModel, Field

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

router = APIRouter()
_cache = {}
_lock = threading.Lock()
LOG_PATH = os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
CHAMPIONS = ["wine-quality-classifier", "housing-regressor", "spam-detector"]


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


@router.get("/models")
def models_info():
    client = MlflowClient()
    out = {}
    for name in CHAMPIONS:
        try:
            mv = client.get_model_version_by_alias(name, "champion")
            out[name] = {"version": mv.version, "alias": "champion", "run_id": mv.run_id}
        except Exception:
            out[name] = None
    return out


@router.post("/reload")
def reload_models():
    with _lock:
        _cache.clear()  # Housing aur Spam ka naya champion agli request pe load hoga
    try:
        import src.serving.app as main  # Wine ka model app.py me module-level `model` hai
        name = getattr(main, "MODEL_NAME", "wine-quality-classifier")
        main.model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"wine reload fail: {exc}")
    return {"status": "reloaded", "models": CHAMPIONS}


def setup(app):
    app.include_router(router)

    @app.middleware("http")
    async def log_predictions(request: Request, call_next):
        if not (request.method == "POST" and request.url.path.startswith("/predict")):
            return await call_next(request)
        body = await request.body()
        response = await call_next(request)
        chunks = [chunk async for chunk in response.body_iterator]
        out = b"".join(chunks)
        try:
            record = {
                "ts": datetime.now(UTC).isoformat(),
                "path": request.url.path,
                "status": response.status_code,
                "input": json.loads(body) if body else None,
                "output": json.loads(out) if out else None,
            }
            os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # logging fail ho to prediction na rukay
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return Response(content=out, status_code=response.status_code,
                        headers=headers, media_type=response.media_type)