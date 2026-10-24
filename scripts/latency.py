import time

import numpy as np
import requests

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}

for _ in range(10):  # warm-up
    requests.post(URL, json=PAYLOAD).raise_for_status()

times = []
for _ in range(200):
    start = time.perf_counter()
    requests.post(URL, json=PAYLOAD).raise_for_status()
    times.append((time.perf_counter() - start) * 1000)

print(f"requests: {len(times)}")
print(f"mean: {np.mean(times):.1f} ms")
print(f"p50 : {np.percentile(times, 50):.1f} ms")
print(f"p95 : {np.percentile(times, 95):.1f} ms")
print(f"max : {np.max(times):.1f} ms")