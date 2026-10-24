from pathlib import Path

import yaml

from conftest import SAMPLE, COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_bool(api_client):
    r = api_client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert isinstance(r.json()["good_quality"], bool)


def test_predict_returns_confidence(api_client):
    r = api_client.post("/predict", json=SAMPLE)
    conf = r.json()["confidence"]
    assert 0.0 <= conf <= 1.0


def test_predict_missing_field_rejected(api_client):
    bad = {k: v for k, v in SAMPLE.items() if k != "alcohol"}
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_wrong_type_rejected(api_client):
    bad = {**SAMPLE, "alcohol": "very high"}
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_model_info_fields(api_client):
    d = api_client.get("/model-info").json()
    assert d["name"] == "wine-quality-classifier"
    assert d["alias"] == "champion"
    assert "f1" in d["metrics"] and "accuracy" in d["metrics"]


def test_runs_returns_list(api_client):
    rows = api_client.get("/runs").json()
    assert isinstance(rows, list) and len(rows) >= 1
    assert rows[0]["model_type"] == "random_forest"


def test_dashboard_served(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    assert "Wine Quality Dashboard" in r.text


def test_input_schema_matches_columns(api_client):
    props = api_client.get("/openapi.json").json()["components"]["schemas"]["WineInput"]["properties"]
    assert len(props) == len(COLUMNS) == 11


def test_params_yaml_valid():
    p = yaml.safe_load(open(ROOT / "params.yaml"))["prepare"]
    assert 0 < p["test_size"] < 1
    assert isinstance(p["seed"], int)