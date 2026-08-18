"""
Train and save all classification models for ML Assignment 2.
Run from the project root:
    python model/train_models.py
"""
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.to_csv(os.path.join(DATA_DIR, "breast_cancer_wisconsin_diagnostic.csv"), index=False)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, filename + ".joblib"))

print("All models trained and saved.")
