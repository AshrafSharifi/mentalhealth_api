import pandas as pd
import pytest

from src.train import FEATURES, TARGET, load_data


def make_csv(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_data_returns_features_in_order(tmp_path):
    df = pd.DataFrame({col: [1, 2] for col in reversed(FEATURES + [TARGET, "id"])})
    X, y = load_data(make_csv(tmp_path, df))
    assert list(X.columns) == FEATURES
    assert y.name == TARGET


def test_load_data_rejects_missing_column(tmp_path):
    df = pd.DataFrame({col: [1, 2] for col in FEATURES if col != "cesd"} | {TARGET: [1, 2]})
    with pytest.raises(ValueError, match="cesd"):
        load_data(make_csv(tmp_path, df))


def test_load_data_rejects_missing_values(tmp_path):
    df = pd.DataFrame({col: [1, 2] for col in FEATURES + [TARGET]})
    df.loc[0, "age"] = None
    with pytest.raises(ValueError, match="missing values"):
        load_data(make_csv(tmp_path, df))