# Heart Failure Mortality Prediction

An end-to-end machine learning project for predicting heart failure mortality risk using clinical patient data. The project includes model artifacts, visual analysis, SHAP explainability, a FastAPI prediction API, and Docker deployment support.

## Project Highlights

- XGBoost model for mortality risk prediction
- Feature engineering for stronger predictive signals
- SHAP explainability visualizations
- FastAPI backend for real-time predictions
- Docker and Docker Compose deployment
- Saved model artifacts for reproducible inference
- Ready-to-upload GitHub project structure

## Project Structure

```text
.
├── main.py                    # FastAPI prediction API
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose service
├── heart.csv                  # Dataset
├── model/
│   ├── xgb_model.pkl          # Trained XGBoost model
│   ├── scaler.pkl             # Saved scaler
│   └── features.json          # Feature order used by the model
├── eda_plots.png              # Exploratory data analysis plots
├── model_comparison.png       # Model comparison chart
├── roc_curves.png             # ROC curve visualization
├── shap_summary.png           # SHAP summary plot
├── shap_importance.png        # SHAP feature importance plot
├── shap_dependence.png        # SHAP dependence plot
└── shap_waterfall.png         # SHAP waterfall explanation
```

## Tech Stack

- Python
- XGBoost
- Scikit-learn
- FastAPI
- Uvicorn
- Pydantic
- SHAP
- Docker

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/alwinpaul111/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The API Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open the interactive API documentation:

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

## Test Prediction Endpoint

Use this sample request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "anaemia": 1,
    "creatinine_phosphokinase": 582,
    "diabetes": 0,
    "ejection_fraction": 38,
    "high_blood_pressure": 1,
    "platelets": 265000,
    "serum_creatinine": 1.9,
    "serum_sodium": 130,
    "sex": 1,
    "smoking": 0,
    "time": 4
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

Stop the service:

```bash
docker-compose down
```

## Model Explainability

The project includes SHAP-based explainability outputs:

- `shap_summary.png`: Global feature impact summary
- `shap_importance.png`: Feature importance ranking
- `shap_dependence.png`: Relationship between key feature values and predictions
- `shap_waterfall.png`: Individual prediction explanation

These plots help make the model more transparent and easier to explain in interviews, portfolio reviews, and project demonstrations.

## GitHub Upload Steps

From the project folder:

```bash
git init
git add .
git commit -m "Upgrade heart failure prediction project"
git branch -M main
git remote add origin https://github.com/alwinpaul111/Heart-Disease-Prediction.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/alwinpaul111/Heart-Disease-Prediction.git
git push -u origin main
```

## Important Note

This project is for educational and portfolio purposes only. It should not be used for real medical decisions without clinical validation, bias testing, regulatory review, and continuous monitoring.

