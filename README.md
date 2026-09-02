# ❤️ Heart Disease Prediction — Data Science Project

A complete end-to-end machine learning project that predicts whether a
patient has heart disease based on clinical parameters.

## Project image
![image_alt](https://github.com/gagandeep399/Heart_Disease_Prediction/blob/35ec6eb969cfa744530d4f9a048248c8ebeb8139/Heat_disease(1).png)

## 📁 Project Structure

```
heart_disease_prediction/
├── data/
│   └── heart.csv              # dataset (synthetic sample included)
├── src/
│   ├── generate_data.py       # generates synthetic dataset
│   ├── eda.py                 # exploratory data analysis + plots
│   ├── train_model.py         # trains & compares ML models
│   └── predict.py             # predicts on a new patient record
├── models/                    # saved trained model + scaler (generated)
├── outputs/                   # EDA plots (generated)
├── notebooks/                 # optional Jupyter notebooks
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The dataset mimics the well-known **UCI Heart Disease** dataset with
these features:

| Column   | Description                                   |
|----------|------------------------------------------------|
| age      | Age in years                                   |
| sex      | 1 = male, 0 = female                           |
| cp       | Chest pain type (0–3)                          |
| trestbps | Resting blood pressure (mm Hg)                 |
| chol     | Serum cholesterol (mg/dl)                      |
| fbs      | Fasting blood sugar > 120 mg/dl (1/0)          |
| restecg  | Resting ECG results (0–2)                      |
| thalach  | Max heart rate achieved                        |
| exang    | Exercise induced angina (1/0)                  |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of peak exercise ST segment (0–2)        |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia type (0–3)                         |
| target   | 1 = heart disease present, 0 = no disease      |

> A ready-to-use **synthetic** `data/heart.csv` is included so the
> project runs immediately. To use the **real** dataset instead,
> download it from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
> or the [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
> and replace `data/heart.csv` (same column names).

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 How to Run

```bash
# 1. Generate the dataset (skip if you already have data/heart.csv)
python src/generate_data.py

# 2. Explore the data
python src/eda.py

# 3. Train and evaluate models
python src/train_model.py

# 4. Predict on a new patient
python src/predict.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

## 📊 Interactive Dashboard

The project includes a **Streamlit dashboard** (`app.py`) with three pages:

1. **Overview & EDA** — key metrics, target distribution, age distribution, correlation heatmap, raw data table
2. **Model Comparison** — trains all 3 models live and shows accuracy/precision/recall/F1 comparison + confusion matrix
3. **Live Prediction** — interactive sliders/dropdowns to enter patient details and get an instant risk prediction

Run it with:
```bash
streamlit run app.py
```
It opens automatically in your browser at `http://localhost:8501`.

## 🧠 Models Used

- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors

The script automatically picks the best model (by F1-score) and saves
it to `models/best_model.pkl`.

## 📈 Results

Running `train_model.py` prints accuracy, precision, recall, F1-score,
and a confusion matrix for each model, and reports the winner.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib

## 📌 Author

GAGANDEEP
