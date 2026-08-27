"""
generate_data.py
-----------------
Generates a synthetic Heart Disease dataset that mimics the structure
of the well-known UCI Heart Disease dataset.

If you have the real dataset (e.g. from Kaggle / UCI), simply replace
`data/heart.csv` with it — the rest of the pipeline (preprocessing,
training, evaluation) will work without any changes, as long as the
column names match.

Columns:
    age       - age in years
    sex       - 1 = male, 0 = female
    cp        - chest pain type (0-3)
    trestbps  - resting blood pressure (mm Hg)
    chol      - serum cholesterol (mg/dl)
    fbs       - fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg   - resting ECG results (0-2)
    thalach   - maximum heart rate achieved
    exang     - exercise induced angina (1 = yes, 0 = no)
    oldpeak   - ST depression induced by exercise
    slope     - slope of peak exercise ST segment (0-2)
    ca        - number of major vessels colored by fluoroscopy (0-4)
    thal      - thalassemia (0-3)
    target    - 1 = heart disease present, 0 = no heart disease
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500  # number of synthetic patients


def generate_heart_data(n=N):
    target = np.random.binomial(1, 0.45, n)

    age = np.random.normal(54, 9, n).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n)
    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])

    # People with heart disease tend to have higher bp/chol/oldpeak,
    # lower max heart rate - we bake in a mild signal so models can learn.
    trestbps = (np.random.normal(131, 17, n) + target * 5).clip(94, 200).astype(int)
    chol = (np.random.normal(246, 51, n) + target * 10).clip(126, 564).astype(int)
    fbs = np.random.binomial(1, 0.15, n)
    restecg = np.random.choice([0, 1, 2], n, p=[0.5, 0.48, 0.02])
    thalach = (np.random.normal(150, 23, n) - target * 12).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.2 + target * 0.25, n).clip(0, 1)
    oldpeak = (np.random.exponential(1.0, n) + target * 0.5).clip(0, 6.2).round(1)
    slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3, 4], n, p=[0.58, 0.21, 0.12, 0.06, 0.03])
    thal = np.random.choice([0, 1, 2, 3], n, p=[0.02, 0.06, 0.55, 0.37])

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target,
    })
    return df


if __name__ == "__main__":
    df = generate_heart_data()
    df.to_csv("data/heart.csv", index=False)
    print(f"Synthetic heart.csv generated with {len(df)} rows -> data/heart.csv")
