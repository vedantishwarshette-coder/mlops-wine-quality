import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/winequality-red.csv", sep=";")
df["good"] = (df["quality"] >= 6).astype(int)
df = df.drop(columns=["quality"])

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["good"])
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("train:", train.shape, "test:", test.shape)