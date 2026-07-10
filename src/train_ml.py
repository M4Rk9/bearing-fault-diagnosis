"""Train baseline ML models for bearing fault diagnosis.

Before running:
1. Download CWRU .mat files into data/raw/.
2. Fill FILE_LABEL_MAP in src/load_data.py.
3. Run: python src/train_ml.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate import plot_confusion_matrix, print_metrics
from feature_extraction import build_feature_table
from load_data import load_dataset


RANDOM_STATE = 42
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and segmenting dataset...")
    windows, labels = load_dataset()

    print("Extracting engineered features...")
    features_df = build_feature_table(windows, labels)
    features_path = PROCESSED_DIR / "features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")

    X = features_df.drop(columns=["label"])
    y = features_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("\nRandom Forest Results")
    print_metrics(y_test, rf_pred)
    plot_confusion_matrix(y_test, rf_pred, save_path=str(FIGURES_DIR / "rf_confusion_matrix.png"))
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")

    print("\nTraining SVM...")
    svm_model = train_svm(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    print("\nSVM Results")
    print_metrics(y_test, svm_pred)
    plot_confusion_matrix(y_test, svm_pred, save_path=str(FIGURES_DIR / "svm_confusion_matrix.png"))
    joblib.dump(svm_model, MODELS_DIR / "svm.pkl")

    print("\nTraining complete. Models saved in models/.")


if __name__ == "__main__":
    main()
