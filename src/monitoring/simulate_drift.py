import os

import pandas as pd

os.makedirs("data/monitoring", exist_ok=True)

test = pd.read_csv("data/processed/test.csv").drop(columns=["good"])

# normal batch: koi badlav nahi
normal = test.sample(n=200, random_state=1)
normal.to_csv("data/monitoring/normal_batch.csv", index=False)

# shifted batch: alcohol aur sulphates badha do
shifted = test.sample(n=200, random_state=2).copy()
shifted["alcohol"] = shifted["alcohol"] + 1.5
shifted["sulphates"] = shifted["sulphates"] + 0.3
shifted.to_csv("data/monitoring/shifted_batch.csv", index=False)

print("normal:", normal.shape, "shifted:", shifted.shape)