"""
train_model.py
---------------
Trains and evaluates multiple classification models on the Heart Disease
dataset, then saves the best-performing model to disk using joblib.

Models compared:
    - Logistic Regression
    - Random Forest Classifier
    - K-Nearest Neighbors

Usage:
    python src/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/heart.csv"
MODEL_DIR = "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    return {"name": name, "model": model, "accuracy": acc, "f1": f1}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test_scaled, y_test))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)
    results.append(evaluate("K-Nearest Neighbors", knn, X_test_scaled, y_test))

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model: {best['name']} (F1 = {best['f1']:.3f})")

    joblib.dump(best["model"], f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print(f"Saved best model -> {MODEL_DIR}/best_model.pkl")
    print(f"Saved scaler     -> {MODEL_DIR}/scaler.pkl")


if __name__ == "__main__":
    main()
