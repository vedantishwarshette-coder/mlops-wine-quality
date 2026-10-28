import io
import os
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/raw/SMSSpamCollection"
URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

if not os.path.exists(RAW):
    for url in URLS:
        try:
            blob = urllib.request.urlopen(url, timeout=60).read()
            zipfile.ZipFile(io.BytesIO(blob)).extract("SMSSpamCollection", "data/raw")
            print("downloaded from", url)
            break
        except Exception as exc:
            print("fail:", url, exc)
    else:
        raise SystemExit("Download fail. UCI site se 'SMS Spam Collection' zip download karke "
                         "usme se SMSSpamCollection file data\\raw\\ me rakh do.")

df = pd.read_csv(RAW, sep="\t", header=None, names=["label", "text"])
df["label"] = (df["label"] == "spam").astype(int)  # spam = 1

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train.to_csv("data/processed/spam_train.csv", index=False)
test.to_csv("data/processed/spam_test.csv", index=False)
print("train:", train.shape, "test:", test.shape, "spam share:", round(df["label"].mean(), 3))