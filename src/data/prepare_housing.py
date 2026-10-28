import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

df = fetch_california_housing(as_frame=True).frame  # target column: MedHouseVal
df.to_csv("data/raw/housing.csv", index=False)

train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv("data/processed/housing_train.csv", index=False)
test.to_csv("data/processed/housing_test.csv", index=False)
print("train:", train.shape, "test:", test.shape)