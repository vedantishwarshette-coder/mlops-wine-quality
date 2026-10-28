import subprocess
import sys
import time

BATCH = sys.argv[1]            # features wali batch (drift check ke liye)
LABELED = sys.argv[2]          # labels wali batch (retraining ke liye)


def run(cmd):
    print(">", " ".join(cmd))
    return subprocess.run(cmd).returncode


# 1. drift check (drift mile to exit code 1)
code = run([sys.executable, "src/monitoring/drift.py", BATCH])
if code == 0:
    print("Drift nahi mila, retraining ki zaroorat nahi.")
    sys.exit(0)
if code != 1:
    print("Drift check me error aaya, ruk gaya.")
    sys.exit(code)

print("Drift mila, retraining shuru.")
start_ms = int(time.time() * 1000)

# 2. naye labeled data ke saath retrain (naye runs MLflow me jayenge)
if run([sys.executable, "src/training/train.py", "--config", "configs/wine.yaml", "--extra", LABELED]) != 0:
    print("Training fail hui.")
    sys.exit(1)

# 3. sirf naye runs me se chuno, champion se compare karo, behtar ho to promote
sys.exit(run([sys.executable, "src/registry/promote.py", str(start_ms)]))