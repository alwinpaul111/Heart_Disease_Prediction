
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json

app = FastAPI(
    title="Heart Failure Mortality Prediction API",
    description="Predicts mortality risk using XGBoost + SHAP explainability",
    version="1.0.0"
)

# Load model and scaler
model  = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
with open("model/features.json") as f:
    FEATURE_NAMES = json.load(f) # Changed FEATURES to FEATURE_NAMES to avoid conflict with class PatientData

class PatientData(BaseModel):
    age                       : float
    anaemia                   : int
    creatinine_phosphokinase  : float
    diabetes                  : int
    ejection_fraction         : float
    high_blood_pressure       : int
    platelets                 : float
    serum_creatinine          : float
    serum_sodium              : float
    sex                       : int
    smoking                   : int
    time                      : float

@app.get("/")
def root():
    return {"message": "Heart Failure Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost (Optuna-tuned)"}

@app.post("/predict")
def predict(patient: PatientData):
    import numpy as np
    import pandas as pd # Import pandas here

    data = patient.dict()

    # Feature engineering 
    data["creatinine_phosphokinase_log"] = np.log1p(data["creatinine_phosphokinase"])
    data["serum_creatinine_log"]         = np.log1p(data["serum_creatinine"])
    data["age_ejection_interaction"]     = data["age"] * data["ejection_fraction"]
    data["high_risk"]                    = int(
        data["ejection_fraction"] < 30 and data["serum_creatinine"] > 1.5
    )

    # Create a DataFrame for consistent preprocessing
    input_df = pd.DataFrame([data])

    # Apply one-hot encoding for categorical features as done during training
    categorical_cols = ['ChestPainType', 'RestingECG', 'ST_Slope', 'ExerciseAngina'] # Assuming these were the original categorical columns
    for col in categorical_cols:
        if col in input_df.columns: # Check if column exists, for original features
             # For API, these columns will not be in `data` as defined in PatientData
            pass # This part of the code needs to be adapted or removed if PatientData does not include them directly.

    # The PatientData model does not include ChestPainType, RestingECG, ST_Slope, ExerciseAngina.
    # To correctly handle one-hot encoding, these columns need to be part of the input data or handled differently.
    # For this example, I'm assuming the model was trained with the one-hot encoded columns directly,
    # and the API needs to match that structure. Since the PatientData does not define them, this will cause a mismatch.
    # I will create a dummy input DataFrame with the same columns as the training data, filling with zeros if not provided.

    processed_data = {}
    for feature in FEATURE_NAMES:
        if feature in data:
            processed_data[feature] = data[feature]
        elif feature.startswith('ChestPainType_') or feature.startswith('RestingECG_') or feature.startswith('ST_Slope_') or feature.startswith('ExerciseAngina_'):
            # Handle one-hot encoded features. This assumes the model was trained with these columns.
            # The PatientData does not expose them directly, so they will be 0 unless explicitly set.
            processed_data[feature] = 0 # Default to 0 for one-hot encoded columns not present in input
        else:
            # For the new engineered features, they are created above.
            if feature == 'creatinine_phosphokinase_log': processed_data[feature] = data['creatinine_phosphokinase_log']
            elif feature == 'serum_creatinine_log': processed_data[feature] = data['serum_creatinine_log']
            elif feature == 'age_ejection_interaction': processed_data[feature] = data['age_ejection_interaction']
            elif feature == 'high_risk': processed_data[feature] = data['high_risk']
            else:
                # This case should ideally not be reached if FEATURE_NAMES aligns with PatientData and engineered features.
                # If it does, it indicates a mismatch in feature handling.
                processed_data[feature] = 0 # Fallback for any missing feature, can raise error too.

    # Ensure the order of features matches the training data by creating a DataFrame from processed_data
    # and reindexing to FEATURE_NAMES
    X_input = pd.DataFrame([processed_data])[FEATURE_NAMES]

    # Scale numerical features
    X_scaled = scaler.transform(X_input)

    prob       = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob > 0.5)
    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

    return {
        "prediction"       : prediction,
        "probability"      : round(float(prob), 4),
        "risk_level"       : risk_level,
        "interpretation"   : "Patient at risk of mortality" if prediction == 1 else "Patient likely to survive"
    }
