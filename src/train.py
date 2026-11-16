"""Train and evaluate models for emotional exhaustion (mbi_ex).

Usage:
    python -m src.train --data data/mentalhealth.csv --out models/
"""

import argparse
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split,GridSearchCV,KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/mentalhealth.csv")
#print(df.columns)
TARGET = ["mbi_ex"]
EXCLUDED = ["id", "mbi_cy", "mbi_ea"]
FEATURES = [
    "age", "year", "sex", "glang", "part", "job", "stud_h", "health", "psyt",
    "jspe", "qcae_cog", "qcae_aff", "amsp", "erec_mean", "cesd", "stai_t",
]
RANDOM_STATE = 42


def load_data(path: Path) -> tuple[pd.DataFrame,pd.Series]:
    df= pd.read_csv(path)
    missing = set(FEATURES+TARGET)-set(df.columns)
    if missing:
        raise ValueError(f"These columns are missing: {sorted(missing)}")
    elif df[FEATURES+TARGET].isnull().values.any():
        raise ValueError("Data contains null value")
    
    return df[FEATURES],df[TARGET]


def build_models() -> dict[str,object]:

    lin_model = Pipeline([("scaler",StandardScaler()),("model",LinearRegression())])
    #the double underscore means “the max_depth parameter of the pipeline step called model”.
    grid = {
        "model__n_estimators": [300],
        "model__max_depth": [6, 8, 10],
        "model__min_samples_leaf": [5, 10],
        "model__max_features": [3, 8, 16],
    }

    rf = Pipeline([("model",RandomForestRegressor(n_jobs=-1,random_state=RANDOM_STATE))])
    rf_tuned = GridSearchCV(rf,
                  grid,
                  cv=KFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE),
                  n_jobs=-1,
                   scoring="neg_root_mean_squared_error" ) #highest value is preferable , near to zero 

    return {"linear_model": lin_model, "rf_model":rf_tuned}

def evaluate(model, X_train, y_train, X_test, y_test) -> dict:


    cv = KFold(n_splits=6, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(model,X_train,y_train,cv=cv,scoring={"rmse":"neg_root_mean_squared_error", "r2":"r2"})
    cross_rmse = -scores["test_rmse"]
    cross_r2 = scores["test_r2"]

    model.fit(X_train,y_train)
    predict = model.predict(X_test)

    result = {
        "cross_rmse_mean": cross_rmse.mean(),
        "cross_rmse_std": cross_rmse.std(),
        "cross_r2_mean": cross_r2.mean(),
        "cross_r2_std": cross_r2.std(),
        "test_rmse": root_mean_squared_error(predict,y_test),
        "test_r2":r2_score(predict,y_test)


    }
    if isinstance(model, GridSearchCV):
        result["best_params"] = {k.replace("model__", ""): v for k, v in model.best_params_.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mentalhealth.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    X, y = load_data(args.data)
    # Hold out a test set FIRST; nothing below sees it until the final score.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    metrics = {"n_train": len(X_train), "n_test": len(X_test), "models": {}}
    fitted = {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        metrics["models"][name] = evaluate(model, X_train, y_train, X_test, y_test)
        fitted[name] = model
        m = metrics["models"][name]
        print(f"  CV RMSE {m['cross_rmse_mean']:.2f} ± {m['cross_rmse_std']:.2f} | "
              f"CV R2 {m['cross_r2_mean']:.2f} ± {m['cross_r2_std']:.2f} | "
              f"test RMSE {m['test_rmse']:.2f} | test R2 {m['test_r2']:.2f}")

    best_pipeline = fitted["rf_model"].best_estimator_

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "features": FEATURES}, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {args.out}/")


if __name__ == "__main__":
    main()




    
        
    

