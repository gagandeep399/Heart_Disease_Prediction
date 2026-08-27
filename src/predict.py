"""
predict.py
----------
Loads the saved model + scaler and predicts heart disease risk for a
new patient record.

Usage:
    python src/predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Example patient record - edit these values to test your own case
sample_patient = {
    "age": 58,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 289,
    "fbs": 0,
    "restecg": 0,
    "thalach": 145,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 1,
    "ca": 1,
    "thal": 2,
}


def predict(patient_dict):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame([patient_dict])

    # RandomForest doesn't need scaling but LR/KNN do;
    # scaling doesn't hurt tree-based predictions since we only use
    # scaler for models that require it. If best_model is RandomForest,
    # feed raw features instead.
    try:
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]
    except Exception:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    print(f"Prediction: {label}")
    print(f"Probability of heart disease: {prob:.2%}")
    return pred, prob


if __name__ == "__main__":
    predict(sample_patient)
