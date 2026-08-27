"""
eda.py
------
Basic Exploratory Data Analysis for the Heart Disease dataset.
Saves plots into the `outputs/` folder and prints summary stats.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/heart.csv"
OUT_DIR = "outputs"


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df["target"].value_counts())
    print("\nSummary statistics:\n", df.describe())

    # Target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="target", data=df)
    plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
    plt.savefig(f"{OUT_DIR}/target_distribution.png", bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # Age distribution by target
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue="target", kde=True, bins=20)
    plt.title("Age Distribution by Heart Disease Status")
    plt.savefig(f"{OUT_DIR}/age_distribution.png", bbox_inches="tight")
    plt.close()

    print(f"\nEDA plots saved to '{OUT_DIR}/' folder.")


if __name__ == "__main__":
    run_eda()
