# Insurance Premium Prediction API

A FastAPI-based ML service that predicts an individual's insurance premium category (`Low` / `Medium` / `High`) from demographic, health, and socio-economic inputs.

A scikit-learn `Pipeline` (OneHotEncoder + RandomForestClassifier) is trained in `notebook/fastapi_ml_model.ipynb`, serialized to `model/model.pkl`, and served via FastAPI with Pydantic validation and derived feature engineering.

## Features

- `POST /predict` returning predicted category, confidence, and full class probabilities
- `GET /health` with model version and loaded status
- Pydantic input validation with auto-derived features:
  - `bmi` from `weight / height²`
  - `lifestyle_risk` (`low` / `medium` / `high`) from smoking + BMI
  - `age_group` (`young` / `adult` / `middle_aged` / `senior`)
  - `city_tier` (`1` / `2` / `3`) from curated Indian city lists
- Dockerized with `uv` for reproducible builds
- Training notebook included for transparency / retraining

## Project Structure

```text
.
├── app.py                    # FastAPI app: /, /health, /predict
├── config/
│   └── city_tier.py          # tier_1_cities, tier_2_cities lists
├── data/
│   └── insurance.csv         # Training data (age, weight, height, income_lpa, smoker, city, occupation, insurance_premium_category)
├── model/
│   ├── model.pkl             # Trained sklearn Pipeline
│   └── predict.py            # predict_output(), MODEL_VERSION = 1.0.0
├── notebook/
│   └── fastapi_ml_model.ipynb # Training + evaluation notebook
├── schema/
│   ├── user_input.py         # UserInput request model + computed fields
│   └── prediction_response.py # PredictionResponse response model
├── Dockerfile                # python:3.14-slim + uv + uvicorn
├── pyproject.toml            # fastapi, pandas, scikit-learn, uvicorn, pyd
└── uv.lock
```

## Dataset

`data/insurance.csv` — 100 rows, 8 columns:

| Column | Type | Example |
|---|---|---|
| `age` | int | 67 |
| `weight` | float (kg) | 119.8 |
| `height` | float (m) | 1.56 |
| `income_lpa` | float | 2.92 |
| `smoker` | bool | False |
| `city` | str | Jaipur |
| `occupation` | categorical (7 values) | retired |
| `insurance_premium_category` | target: `High`, `Low`, `Medium` | High |

Occupations: `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job`.

## Feature Engineering

Raw inputs are transformed both in training (notebook) and at inference (Pydantic computed fields in `schema/user_input.py`) so they stay in sync:

1. **BMI:** `weight / height²`
2. **Age group:** `<25 young`, `<45 adult`, `<60 middle_aged`, else `senior`
3. **Lifestyle risk:**
   - `high` if smoker and BMI > 30
   - `medium` if smoker or BMI > 27
   - else `low`
4. **City tier:** `1` if in `tier_1_cities` (Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune), `2` if in `tier_2_cities` (~48 cities, see `config/city_tier.py`), else `3`

Final model features (6): `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`.

## Model

- Pipeline: `ColumnTransformer OneHotEncoder(age_group, lifestyle_risk, occupation, city_tier) + passthrough(bmi, income_lpa)` → `RandomForestClassifier(random_state=42)`
- Train/test split: `test_size=0.2, random_state=1`
- Version: `1.0.0` (see `MODEL_VERSION` in `model/predict.py`)
- Artifact: `model/model.pkl` (pickle)
- Classes: `['High', 'Low', 'Medium']`

To retrain, run `notebook/fastapi_ml_model.ipynb` and overwrite `model/model.pkl`.

## API

### `GET /`

```json
{ "message": "insurance premium prediction api" }
```

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0", "model_loaded": true }
```

### `POST /predict`

Request body (`UserInput`):

```json
{
  "age": 35,
  "weight": 80,
  "height": 1.75,
  "income_lpa": 12.5,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

Validation rules: `0 < age < 120`, `weight > 0`, `0 < height < 2.5`, `income_lpa > 0`, `city` auto-normalized with `.strip().title()`.

Success response (`200`, `PredictionResponse`):

```json
{
  "response": {
    "predicted_category": "Low",
    "confidence": 0.84,
    "class_probabilities": { "High": 0.01, "Low": 0.84, "Medium": 0.15 }
  }
}
```

Note: `app.py` returns `{'response': prediction}` inside a `JSONResponse`; the Pydantic `PredictionResponse` shape describes the inner `prediction` dict.

## Getting Started

Requires Python 3.14.

### With uv (recommended)

```bash
uv sync
uv run uvicorn app:app --reload --port 8000
```

### With pip

```bash
pip install "fastapi[standard]>=0.143.0" pandas scikit-learn uvicorn
uvicorn app:app --reload --port 8000
```

Then open:

- API: `http://localhost:8000/`
- Docs (Swagger): `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Example cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"weight":80,"height":1.75,"income_lpa":12.5,"smoker":false,"city":"Mumbai","occupation":"private_job"}'
```

## Docker

```bash
docker build -t insurance-premium-prediction .
docker run -p 8000:8000 insurance-premium-prediction
```

The image uses `python:3.14-slim`, installs deps with `uv sync --frozen`, exposes `8000`, and runs `uvicorn app:app --host 0.0.0.0 --port 8000`. `notebook/` and `data/` are excluded via `.dockerignore`.

## Tech Stack

- FastAPI + Uvicorn
- Pydantic v2 (`field_validator`, `computed_field`)
- pandas, scikit-learn
- uv, Docker
