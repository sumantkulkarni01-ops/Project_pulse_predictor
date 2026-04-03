from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.processing.feature_engineering import build_feature_dataset

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "risk_model.joblib"


def train_and_save_model():
    df = build_feature_dataset().copy()

    feature_cols = [
        "cost_variance_pct",
        "effort_variance_pct",
        "schedule_variance_pct",
        "defect_count",
        "change_requests",
        "resource_attrition",
        "customer_escalation",
        "resource_utilization_pct",
        "practice",
        "risk_category"
    ]

    target_col = "risk_flag"

    X = df[feature_cols]
    y = df[target_col]

    numeric_features = [
        "cost_variance_pct",
        "effort_variance_pct",
        "schedule_variance_pct",
        "defect_count",
        "change_requests",
        "resource_attrition",
        "customer_escalation",
        "resource_utilization_pct"
    ]

    categorical_features = ["practice", "risk_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved at: {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save_model()