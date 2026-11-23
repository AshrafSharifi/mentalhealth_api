"""Shared test fixtures.

The tests never use the real data or the real model: they train a tiny model on
random numbers, so they run in seconds and work on any machine (including CI).
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from api.main import app
from src.train import FEATURES


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 1, size=(50, len(FEATURES))), columns=FEATURES)
    y = rng.uniform(5, 30, size=50)
    pipeline = Pipeline([("model", RandomForestRegressor(n_estimators=10, random_state=0))])
    pipeline.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, path)
    return path


@pytest.fixture(scope="session")
def client(model_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(model_path))
        # "with" runs the app's lifespan, i.e. loads the model, like a real server start.
        with TestClient(app) as c:
            yield c


@pytest.fixture
def valid_input():
    return {
        "age": 22, "year": 3, "sex": 2, "glang": 1, "part": 1, "job": 0, "stud_h": 25,
        "health": 4, "psyt": 0, "jspe": 107, "qcae_cog": 58, "qcae_aff": 35, "amsp": 23,
        "erec_mean": 0.72, "cesd": 16, "stai_t": 43,
    }