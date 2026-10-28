import pandas as pd
import pytest

from src.data.validate import FEATURE_RANGES, TARGET, pa, validate_frame


def good_frame(rows=5):
    row = {name: (lo + hi) / 2 for name, (lo, hi) in FEATURE_RANGES.items()}
    row[TARGET] = 1
    return pd.DataFrame([row] * rows)


def test_valid_data_passes():
    assert len(validate_frame(good_frame())) == 5


def test_out_of_range_value_is_rejected():
    df = good_frame()
    df.loc[0, "alcohol"] = 50
    with pytest.raises(pa.errors.SchemaErrors):
        validate_frame(df)


def test_missing_target_is_rejected():
    df = good_frame()
    df.loc[1, TARGET] = None
    with pytest.raises(pa.errors.SchemaErrors):
        validate_frame(df)


def test_missing_column_is_rejected():
    df = good_frame().drop(columns=["pH"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_frame(df)


def test_target_must_be_binary():
    df = good_frame()
    df.loc[0, TARGET] = 3
    with pytest.raises(pa.errors.SchemaErrors):
        validate_frame(df)