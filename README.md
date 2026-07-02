# Heart Failure Mortality Prediction

An upgraded machine learning project built in Google Colab for predicting heart disease / heart failure mortality risk from clinical health indicators. The project moves beyond a basic classroom notebook by adding advanced models, hyperparameter tuning, model explainability, experiment tracking, API deployment, and Docker support.

## Project Overview

This project follows a complete ML pipeline:

1. Exploratory Data Analysis
2. Feature Engineering and Preprocessing
3. Baseline Model Training
4. Advanced Model Training
5. Hyperparameter Tuning with Optuna
6. Model Evaluation and Comparison
7. SHAP Explainability
8. MLflow Experiment Tracking
9. Model Saving
10. FastAPI Deployment
11. Docker Containerization
12. GitHub Upload Package Creation

## Key Features

- Logistic Regression baseline model
- Random Forest baseline model
- XGBoost advanced model
- LightGBM advanced model
- Optuna hyperparameter tuning
- SMOTE handling for class imbalance
- SHAP explainability plots
- MLflow experiment tracking
- FastAPI prediction endpoint
- Docker and Docker Compose deployment
- Saved trained model, scaler, and feature list
- Ready-to-upload `github_upload` folder

## Tech Stack

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- MLflow
- FastAPI
- Uvicorn
- Docker

## Dataset

The project uses `heart.csv`.

Target column:

```text
HeartDisease
```

The notebook also creates engineered features from the available clinical columns, including:

- `creatinine_phosphokinase_log`
- `serum_creatinine_log`
- `age_ejection_interaction`
- `high_risk`

## Repository Structure

```text
.
├── Heart_Disease_Prediction_UPGRADED.ipynb
├── heart.csv
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── features.json
├── eda_plots.png
├── model_comparison.png
├── roc_curves.png
├── shap_summary.png
├── shap_importance.png
├── shap_dependence.png
└── shap_waterfall.png
```

## Google Colab Workflow

Open the notebook:

```text
Heart_Disease_Prediction_UPGRADED.ipynb
```

Upload `heart.csv` to Colab, then run the notebook cells in order.

The first cell installs the required libraries:

```python
!pip install xgboost lightgbm optuna shap mlflow fastapi uvicorn scikit-learn pandas numpy matplotlib seaborn imbalanced-learn -q
```

The notebook then performs:

- data loading
- EDA visualization
- feature engineering
- one-hot encoding
- train-test split
- scaling
- SMOTE oversampling
- baseline model training
- advanced model training
- Optuna tuning
- SHAP explanation generation
- MLflow logging
- model export
- FastAPI app creation
- Docker file creation
- GitHub upload ZIP creation

## Models Used

### Baseline Models

- Logistic Regression
- Random Forest

### Advanced Models

- XGBoost
- LightGBM

### Tuned Model

The final deployed model is an Optuna-tuned XGBoost classifier saved as:

```text
model/xgb_model.pkl
```

The scaler is saved as:

```text
model/scaler.pkl
```

The feature order is saved as:

```text
model/features.json
```

## Generated Visualizations

The Colab notebook generates the following visual outputs:

```text
eda_plots.png
model_comparison.png
roc_curves.png
shap_summary.png
shap_importance.png
shap_dependence.png
shap_waterfall.png
```

## Model Explainability

SHAP is used to explain model predictions.

Included explanation files:

- `shap_summary.png`: overall feature impact
- `shap_importance.png`: mean absolute SHAP feature importance
- `shap_dependence.png`: dependence plot for the most important feature
- `shap_waterfall.png`: explanation for one sample patient prediction

## MLflow Tracking

The notebook logs model experiments to MLflow using:

- accuracy
- F1 score
- ROC-AUC
- model parameters
- trained model artifacts

MLflow is used to compare:

- Logistic Regression
- Random Forest
- XGBoost
- XGBoost Tuned
- LightGBM

## Run API Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model": "XGBoost (Optuna-tuned)"
}
```

## Prediction API

Endpoint:

```text
POST /predict
```

Sample request:

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

Sample response:

```json
{
  "prediction": 1,
  "probability": 0.8123,
  "risk_level": "HIGH",
  "interpretation": "Patient at risk of mortality"
}
```

## Run With Docker

Build the Docker image:

```bash
docker build -t heart-failure-api .
```

Run the container:

```bash
docker run -p 8000:8000 heart-failure-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run With Docker Compose

```bash
docker-compose up --build
```

Stop the container:

```bash
docker-compose down
```

## GitHub Upload Instructions

If you are uploading the generated `github_upload` folder to GitHub:

```bash
cd github_upload
git init
git add .
git commit -m "Add upgraded heart failure prediction project"
git branch -M main
git remote add origin https://github.com/alwinpaul111/Heart-Disease-Prediction.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/alwinpaul111/Heart-Disease-Prediction.git
git push -u origin main
```

## Final Output Files

After running the Colab notebook, the final upload package contains:

```text
main.py
Dockerfile
docker-compose.yml
requirements.txt
heart.csv
model/xgb_model.pkl
model/scaler.pkl
model/features.json
eda_plots.png
model_comparison.png
roc_curves.png
shap_summary.png
shap_importance.png
shap_waterfall.png
shap_dependence.png
```

## Important Note

This project is for educational and portfolio use only. It is not intended for real medical diagnosis or clinical decision-making. A production medical ML system would require clinical validation, fairness checks, monitoring, security controls, and regulatory review.

