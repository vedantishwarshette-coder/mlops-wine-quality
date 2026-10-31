import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime

BATCH = sys.argv[1]      # features wali batch (drift check ke liye)
LABELED = sys.argv[2]    # labels wali batch (retraining ke liye)
CONFIG = "configs/wine.yaml"
API = os.environ.get("API_URL", "http://127.0.0.1:8000")
LOG = "logs/retrain_log.jsonl"


def run(cmd, capture=False):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True)


def log_decision(record):
    os.makedirs("logs", exist_ok=True)
    record["ts"] = datetime.now(UTC).isoformat()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# 1. drift check (drift mile to exit code 1)
code = run([sys.executable, "src/monitoring/drift.py", BATCH]).returncode
if code == 0:
    log_decision({"batch": BATCH, "drift": False, "action": "none"})
    print("Drift nahi mila, retraining ki zaroorat nahi.")
    sys.exit(0)
if code != 1:
    print("Drift check me error aaya, ruk gaya.")
    sys.exit(code)

print("Drift mila, retraining shuru.")
start_ms = int(time.time() * 1000)

# 2. naye labeled data ke saath retrain
if run([sys.executable, "src/training/train.py", "--config", CONFIG, "--extra", LABELED]).returncode != 0:
    log_decision({"batch": BATCH, "drift": True, "action": "train_failed"})
    print("Training fail hui.")
    sys.exit(1)

# 3. naye runs me se chuno, champion se compare, behtar ho to promote
proc = run([sys.executable, "src/registry/promote.py", str(start_ms), "--config", CONFIG], capture=True)
print(proc.stdout)
last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
promoted = "promoted" in last
log_decision({"batch": BATCH, "drift": True, "action": "promoted" if promoted else "rejected", "detail": last})

# 4. API ko naya champion uthane ko bolo (image rebuild nahi chahiye)
if promoted:
    try:
        req = urllib.request.Request(API + "/reload", data=b"", method="POST")
        print("API reload:", urllib.request.urlopen(req, timeout=60).read().decode())
    except Exception as exc:
        print("API reload nahi hua (API chalu hai?):", exc)

sys.exit(proc.returncode)