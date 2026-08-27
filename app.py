"""
app.py
------
Interactive Streamlit dashboard for the Heart Disease Prediction project.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = "data/heart.csv"
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

st.set_page_config(page_title="Heart Disease Prediction Dashboard", page_icon="❤️", layout="wide")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        from src.generate_data import generate_heart_data
        os.makedirs("data", exist_ok=True)
        df = generate_heart_data()
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_models(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    trained = {}
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-score": f1_score(y_test, preds),
        })
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(trained[best_name], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return results_df, trained, scaler, X_test, y_test


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title("❤️ Heart Disease Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview & EDA", "🧠 Model Comparison", "🩺 Live Prediction"])

df = load_data()

# ---------------------------------------------------------
# Page 1: Overview & EDA
# ---------------------------------------------------------
if page == "📊 Overview & EDA":
    st.title("📊 Dataset Overview & EDA")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("With Heart Disease", int(df["target"].sum()))
    col3.metric("Without Heart Disease", int((df["target"] == 0).sum()))
    col4.metric("Average Age", f"{df['age'].mean():.1f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="target", data=df, ax=ax)
        ax.set_xticklabels(["No Disease", "Disease"])
        st.pyplot(fig)

    with c2:
        st.subheader("Age Distribution by Status")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="age", hue="target", kde=True, bins=20, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# Page 2: Model Comparison
# ---------------------------------------------------------
elif page == "🧠 Model Comparison":
    st.title("🧠 Model Training & Comparison")

    with st.spinner("Training models..."):
        results_df, trained, scaler, X_test, y_test = train_models(df)

    st.success(f"Best model: **{results_df.iloc[0]['Model']}** (F1 = {results_df.iloc[0]['F1-score']:.3f})")

    st.subheader("Performance Metrics")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"
    }), use_container_width=True)

    fig, ax = plt.subplots()
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
    plt.xticks(rotation=15)
    plt.title("Model Comparison")
    st.pyplot(fig)

    st.subheader("Confusion Matrix — Best Model")
    best_model = trained[results_df.iloc[0]["Model"]]
    if results_df.iloc[0]["Model"] == "Random Forest":
        preds = best_model.predict(X_test)
    else:
        preds = best_model.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------------------------------------------------
# Page 3: Live Prediction
# ---------------------------------------------------------
elif page == "🩺 Live Prediction":
    st.title("🩺 Predict Heart Disease Risk")
    st.write("Enter patient details below to get a live prediction.")

    with st.spinner("Preparing model..."):
        results_df, trained, scaler, X_test, y_test = train_models(df)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained[best_name]
    st.caption(f"Using best model: **{best_name}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 29, 77, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 94, 200, 130)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with c2:
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)

    with c3:
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

    patient = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    if st.button("🔍 Predict", type="primary"):
        patient_df = pd.DataFrame([patient])
        if best_name == "Random Forest":
            pred = best_model.predict(patient_df)[0]
            prob = best_model.predict_proba(patient_df)[0][1]
        else:
            scaled = scaler.transform(patient_df)
            pred = best_model.predict(scaled)[0]
            prob = best_model.predict_proba(scaled)[0][1]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **Heart Disease Detected** — Risk Probability: {prob:.1%}")
        else:
            st.success(f"✅ **No Heart Disease Detected** — Risk Probability: {prob:.1%}")

        st.progress(min(int(prob * 100), 100))
