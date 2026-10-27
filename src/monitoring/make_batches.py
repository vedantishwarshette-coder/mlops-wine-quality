import os

import pandas as pd

SEED = 42
N = 300

os.makedirs("data/monitoring", exist_ok=True)

train = pd.read_csv("data/processed/train.csv").drop(columns=["good"])
test = pd.read_csv("data/processed/test.csv").drop(columns=["good"])
std = train.std()

# normal batch: test data jaisa ka waisa
normal = test.sample(n=N, random_state=SEED)
normal.to_csv("data/monitoring/normal_batch.csv", index=False)

# shifted batch: 5 columns ko training std ke hisaab se khisko do
shifted = test.sample(n=N, random_state=SEED + 1).copy()
shifted["alcohol"] += 1.5 * std["alcohol"]
shifted["sulphates"] += 1.5 * std["sulphates"]
shifted["density"] += 1.5 * std["density"]
shifted["volatile acidity"] += 1.0 * std["volatile acidity"]
shifted["pH"] -= 1.0 * std["pH"]
shifted.to_csv("data/monitoring/shifted_batch.csv", index=False)

print("normal_batch.csv :", normal.shape)
print("shifted_batch.csv:", shifted.shape)