"""FastAPI service for the emotional-exhaustion model.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_PATH = "models/model.joblib"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE when the server starts, not on every request.
    # Read the path at startup (not import time), so tests can point to a test model.
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    bundle = joblib.load(model_path)
    state["pipeline"] = bundle["pipeline"]
    state["features"] = bundle["features"]
    # TreeExplainer needs the tree model itself, i.e. the last step of the pipeline.
    state["explainer"] = shap.TreeExplainer(state["pipeline"].steps[-1][1])
    yield
    state.clear()


app = FastAPI(
    title="Medical student burnout model (deployment exercise)",
    description="Predicts the MBI emotional-exhaustion score. Not a clinical or diagnostic tool.",
    version="0.1.0",
    lifespan=lifespan,
)

class StudentFeatures(BaseModel):
    """One student's survey answers. Ranges follow the questionnaires' possible scores."""

    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422 error

    age: int = Field(ge=16, le=100, description="Age in years")
    year: int = Field(ge=1, le=6, description="Year of study")
    sex: int = Field(ge=1, le=3, description="1 = man, 2 = woman, 3 = non-binary")
    glang: int = Field(ge=1, le=121, description="Mother tongue (code)")
    part: int = Field(ge=0, le=1, description="Has a partner (0/1)")
    job: int = Field(ge=0, le=1, description="Has a paid job (0/1)")
    stud_h: int = Field(ge=0, le=100, description="Study hours per week")
    health: int = Field(ge=1, le=5, description="Satisfaction with health, 1 (very low) to 5 (very high)")
    psyt: int = Field(ge=0, le=1, description="Consulted a psychotherapist in the last year (0/1)")
    jspe: int = Field(ge=20, le=140, description="Jefferson empathy score")
    qcae_cog: int = Field(ge=19, le=76, description="QCAE cognitive empathy")
    qcae_aff: int = Field(ge=12, le=48, description="QCAE affective empathy")
    amsp: int = Field(ge=0, le=35, description="Academic motivation")
    erec_mean: float = Field(ge=0.0, le=1.0, description="Emotion recognition (share correct)")
    cesd: int = Field(ge=0, le=60, description="CES-D depression score")
    stai_t: int = Field(ge=20, le=80, description="STAI trait anxiety score")

class Prediction(BaseModel):
    mbi_ex: float = Field(description="Predicted emotional-exhaustion score")


class Contribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class Explanation(BaseModel):
    mbi_ex: float
    base_value: float = Field(description="Average prediction over the training data")
    top_contributions: list[Contribution]

def to_frame(x: StudentFeatures) -> pd.DataFrame:
    # Build the row in exactly the column order the model was trained on.
    row = x.model_dump()
    return pd.DataFrame([[row[f] for f in state["features"]]], columns=state["features"])

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict", response_model=Prediction)
def predict(x: StudentFeatures) -> Prediction:
    y = state["pipeline"].predict(to_frame(x))[0]
    return Prediction(mbi_ex=round(float(y), 2))


@app.post("/explain", response_model=Explanation)
def explain(x: StudentFeatures, top_k: int = 5) -> Explanation:
    if not 1 <= top_k <= len(state["features"]):
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {len(state['features'])}")

    X = to_frame(x)
    shap_values = state["explainer"].shap_values(X)[0]
    contributions = sorted(
        (
            Contribution(feature=f, value=float(X.iloc[0][f]), shap_value=round(float(s), 3))
            for f, s in zip(state["features"], shap_values)
        ),
        key=lambda c: abs(c.shap_value),
        reverse=True,
    )
    return Explanation(
        mbi_ex=round(float(state["pipeline"].predict(X)[0]), 2),
        base_value=round(float(np.ravel(state["explainer"].expected_value)[0]), 2),
        top_contributions=contributions[:top_k],
    )