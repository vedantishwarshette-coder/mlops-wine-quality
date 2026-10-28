"""Data validation with Pandera.

    python src/data/validate.py

Checks the processed wine train/test files: column names, numeric ranges
(same limits as the API input schema), no missing values, and a 0/1 target.
Exits with code 1 if anything is wrong, so a pipeline or CI step can stop early.
"""
import sys

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # older pandera versions
    import pandera as pa

TARGET = "good"

# column -> (min, max); these match the Field(ge=..., le=...) limits in the API.
FEATURE_RANGES = {
    "fixed acidity": (3, 20),
    "volatile acidity": (0, 2),
    "citric acid": (0, 2),
    "residual sugar": (0, 20),
    "chlorides": (0, 1),
    "free sulfur dioxide": (0, 100),
    "total sulfur dioxide": (0, 350),
    "density": (0.98, 1.1),
    "pH": (2, 5),
    "sulphates": (0, 3),
    "alcohol": (5, 16),
}


def build_schema():
    columns = {
        name: pa.Column(float, pa.Check.in_range(lo, hi), nullable=False, coerce=True)
        for name, (lo, hi) in FEATURE_RANGES.items()
    }
    columns[TARGET] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True)
    return pa.DataFrameSchema(columns, strict=True)


def validate_frame(df):
    """Returns the validated frame; raises pandera SchemaErrors listing every problem."""
    return build_schema().validate(df, lazy=True)


def main():
    ok = True
    for path in ("data/processed/train.csv", "data/processed/test.csv"):
        df = pd.read_csv(path)
        try:
            validate_frame(df)
            print(f"{path}: OK ({len(df)} rows)")
        except pa.errors.SchemaErrors as err:
            ok = False
            print(f"{path}: FAILED, {len(err.failure_cases)} problem(s)")
            print(err.failure_cases.head(10).to_string())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()