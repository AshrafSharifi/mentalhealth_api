# Medical student burnout model: ML deployment exercise

[![tests](https://github.com/<your-username>/mental-health-api/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-username>/mental-health-api/actions/workflows/tests.yml)

An end-to-end ML deployment exercise on a public dataset: a reproducible training script,
a FastAPI service with input validation and SHAP explanations, automated tests, and a Docker image
that runs with one command.

> **Not a clinical or diagnostic tool.** This project practises the engineering workflow.
> The strongest predictors (`cesd`, `stai_t`) are depression and anxiety screening scores that are
> closely related to the target. The model shows associations in one survey sample, not causes,
> and must not be used to assess anyone's mental health.

## How it works

```
data/mentalhealth.csv ──► src/train.py ──► models/model.joblib + metrics.json
                                                   │
                                                   ▼
                          api/main.py (FastAPI) ──► Docker image ──► docker compose up
```

**Stack:** Python 3.12 · scikit-learn · FastAPI · Pydantic · SHAP · pytest · Docker · Docker Compose · GitHub Actions

## Quick start (Docker)

1. Download the data as described in [`data/README.md`](data/README.md).
2. Train the model (creates `models/model.joblib`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -m src.train
   ```
3. Start the API:
   ```bash
   docker compose up --build
   ```
4. Open the interactive docs at <http://127.0.0.1:8000/docs>.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status and whether the model is loaded |
| POST | `/predict` | Predicted emotional-exhaustion score (`mbi_ex`) |
| POST | `/explain?top_k=5` | Prediction plus the top SHAP feature contributions |

Inputs are validated against each questionnaire's possible score range. Invalid input
(wrong type, out of range, missing or unknown fields) returns **422**.

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 22, "year": 3, "sex": 2, "glang": 1, "part": 1, "job": 0, "stud_h": 25,
       "health": 4, "psyt": 0, "jspe": 107, "qcae_cog": 58, "qcae_aff": 35, "amsp": 23,
       "erec_mean": 0.72, "cesd": 16, "stai_t": 43}'
```

Response (the value depends on your trained model):

```json
{"mbi_ex": 15.8}
```

## Results

886 students, 16 features, target `mbi_ex` (emotional exhaustion, observed range 5–30).
708 rows for training and model selection; 178 rows (20%) held out as a test set and used once.
Cross-validation: 6 folds on the training set, mean ± standard deviation across folds.

| Model | CV RMSE | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear regression (baseline) | 4.07 ± 0.31 | 0.39 ± 0.06 | 4.21 | 0.38 |
| Random forest (tuned, nested CV) | 4.05 ± 0.36 | 0.40 ± 0.08 | 4.15 | 0.40 |

Tuned random forest: 300 trees, `max_depth=8`, `max_features=8`, `min_samples_leaf=10`.
Numbers come from `models/metrics.json`, produced by `python -m src.train`.

**What the results show.** Both models explain about 40% of the variance and are off by about
4 points on average. Test-set scores agree with cross-validation, so there is no sign of
overfitting to the training data. The tuned random forest is not meaningfully better than the linear baseline:
the difference (0.02 RMSE) is far smaller than the spread across folds (about 0.3). In practice the
linear model would be the better choice, being simpler and easier to explain. The API serves the
random forest in order to demonstrate tree-based SHAP explanations.

**Validation choices**
- Scaling and tuning happen inside scikit-learn `Pipeline`s, so no information from validation
  folds leaks into training.
- The random forest is tuned with `GridSearchCV` (5 inner folds) and scored with nested
  cross-validation (6 outer folds), so its score is not inflated by the tuning.
- The other two burnout subscales (`mbi_cy`, `mbi_ea`) are excluded from the features, since they
  come from the same questionnaire as the target.

## Development

```bash
pip install -r requirements-dev.txt
pytest -v
uvicorn api.main:app --reload      # run the API without Docker
```

The tests train a tiny model on random data, so they run in seconds and need neither the real
data nor the trained model. GitHub Actions runs them on every push.

## Project structure

```
├── api/main.py              FastAPI app: schemas, validation, /health, /predict, /explain
├── src/train.py             load, split, cross-validate, tune, save model + metrics
├── tests/                   pytest suite (API integration tests, data-loading unit tests)
├── data/README.md           how to download the dataset (data is not in the repo)
├── models/                  trained model output (not in the repo)
├── notebooks/               original exploratory analysis
├── Dockerfile               slim Python image, non-root user, health check
├── docker-compose.yml       one-command run; mounts the model read-only
├── requirements.txt         pinned runtime dependencies
└── requirements-dev.txt     test dependencies
```

## Limitations

- **Small, single-sample dataset** (886 students from one survey). Results may not generalise to
  other universities, countries or years, and the spread across folds is large.
- **Predictors close to the target.** Depression (`cesd`) and anxiety (`stai_t`) scores dominate
  the model; they are related constructs, not independent causes.
- **Categorical codes treated as numbers.** `sex` and `glang` are codes; one-hot encoding would be
  more correct, especially for the linear model.
- **Not production-ready.** No authentication, HTTPS, rate limiting or monitoring. Sensitive
  health-related inputs would need all of these, plus an ethical review.
- **The notebook** in `notebooks/` is the original exploration and loads files that are not part
  of this repo. The reproducible results are the ones from `src/train.py`.

## Data and licence

- Dataset: [Medical Student Mental Health](https://www.kaggle.com/datasets/thedevastator/medical-student-mental-health)
  on Kaggle. Its licence is set by the dataset's authors: see the Kaggle page. The data is not
  redistributed here.
- Code: MIT licence (see [`LICENSE`](LICENSE)).
