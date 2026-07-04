# Heart Disease Prediction

An upgraded machine learning project for predicting heart disease risk from clinical health indicators. The project started as a Google Colab notebook and has been organized into a cleaner repository with a Python FastAPI app, saved model artifacts, Docker support, and model explainability outputs.

## Highlights

- Google Colab notebook workflow
- Logistic Regression and Random Forest baseline models
- XGBoost and LightGBM advanced models
- Optuna hyperparameter tuning
- SHAP explainability plots
- MLflow experiment tracking in the notebook
- FastAPI prediction API
- Docker and Docker Compose support
- Saved model, scaler, and feature list for inference

## Repository Structure

```text
.
├── README.md
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   └── heart.csv
├── notebooks/
│   └── Heart_Disease_Prediction_UPGRADED.ipynb
├── model/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── features.json
└── images/
    ├── eda_plots.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── shap_summary.png
    ├── shap_importance.png
    ├── shap_dependence.png
    └── shap_waterfall.png
```

## Dataset

The dataset is stored at:

```text
data/heart.csv
```

Target column:

```text
HeartDisease
```

## Notebook Workflow

The full model development workflow is available in:

```text
notebooks/Heart_Disease_Prediction_UPGRADED.ipynb
```

The notebook covers:

- data loading
- exploratory data analysis
- feature engineering
- preprocessing and scaling
- SMOTE oversampling
- baseline model training
- XGBoost and LightGBM training
- Optuna tuning
- model comparison
- SHAP explainability
- MLflow experiment logging
- model export
- FastAPI and Docker file generation

## Model Artifacts

The deployed API uses:

```text
model/xgb_model.pkl
model/scaler.pkl
model/features.json
```

These files store the trained XGBoost model, fitted scaler, and expected feature order.

## Visual Output

Generated plots are stored in `images/`:

- `eda_plots.png`
- `model_comparison.png`
- `roc_curves.png`
- `shap_summary.png`
- `shap_importance.png`
- `shap_dependence.png`
- `shap_waterfall.png`

## Run Locally

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI app:

```bash
uvicorn main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Prediction Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "anaemia": 1,
    "creatinine_phosphokinase": 250,
    "diabetes": 0,
    "ejection_fraction": 25,
    "high_blood_pressure": 1,
    "platelets": 200000,
    "serum_creatinine": 2.0,
    "serum_sodium": 130,
    "sex": 1,
    "smoking": 0,
    "time": 30
  }'
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.8123,
  "risk_level": "HIGH",
  "interpretation": "Patient at risk of mortality"
}
```

## Run With Docker

Build the image:

```bash
docker build -t heart-disease-api .
```

Run the container:

```bash
docker run -p 8000:8000 heart-disease-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run With Docker Compose

```bash
docker-compose up --build
```

Stop:

```bash
docker-compose down
```

## GitHub Language Detection

This repository includes `.gitattributes` so GitHub prioritizes Python over the notebook in language statistics:

```text
*.ipynb linguist-vendored
*.py linguist-detectable=true
```




