import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/raw/winequality-red.csv", sep=";")
df["good"] = (df["quality"] >= 6).astype(int)
df = df.drop(columns=["quality"])

train, test = train_test_split(
    df, test_size=params["test_size"], random_state=params["seed"], stratify=df["good"])
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("train:", train.shape, "test:", test.shape)