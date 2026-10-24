import os
import sys

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Drift rule: 11 columns me se kitne drift hue. 0.15 matlab kam se kam 2 columns
# (2/11 = 0.18). Ek column to sirf ittefaq se bhi drift dikha sakta hai (1/11 = 0.09).
DRIFT_SHARE_THRESHOLD = 0.15

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


def get_drift_numbers(result):
    data = result.dict()
    for metric in data["metrics"]:
        if str(metric["metric_id"]).startswith("DriftedColumnsCount"):
            return metric["value"]["count"], metric["value"]["share"]
    print("Result ke metric ids:", [m["metric_id"] for m in data["metrics"]])
    raise RuntimeError("DriftedColumnsCount metric nahi mila")


def check_drift(batch_path):
    name = os.path.splitext(os.path.basename(batch_path))[0]
    reference = pd.read_csv("data/processed/train.csv").drop(columns=["good"])
    current = pd.read_csv(batch_path)

    definition = DataDefinition(numerical_columns=list(reference.columns))
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    os.makedirs("reports/drift", exist_ok=True)
    html_path = f"reports/drift/{name}.html"
    result.save_html(html_path)

    count, share = get_drift_numbers(result)
    detected = share >= DRIFT_SHARE_THRESHOLD
    print(f"{name}: {count} drifted columns, share={share:.2f}, drift_detected={detected}")
    print("report:", html_path)

    try:
        mlflow.set_experiment("wine-monitoring")
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("drifted_columns", count)
            mlflow.log_metric("drift_share", share)
            mlflow.log_metric("drift_detected", int(detected))
            mlflow.log_artifact(html_path)
    except Exception as exc:
        print("MLflow logging fail hui (drift check phir bhi hua):", exc)

    return detected


if __name__ == "__main__":
    drift = check_drift(sys.argv[1])
    sys.exit(1 if drift else 0)