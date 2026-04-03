from pathlib import Path
import joblib
import pandas as pd
from src.processing.feature_engineering import build_feature_dataset

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_FILE = BASE_DIR / "models" / "risk_model.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Please run train_model.py first.")
    return joblib.load(MODEL_FILE)


def predict_risk(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()

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

    pred_probs = model.predict_proba(df[feature_cols])[:, 1]
    pred_labels = model.predict(df[feature_cols])

    result_df = df.copy()
    result_df["predicted_risk_probability"] = pred_probs
    result_df["predicted_risk_flag"] = pred_labels
    result_df["predicted_risk_level"] = result_df["predicted_risk_probability"].apply(map_risk_level)

    return result_df


def map_risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.65:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


if __name__ == "__main__":
    df = build_feature_dataset()
    predictions = predict_risk(df)
    print(predictions[["project_id", "week_id", "predicted_risk_probability", "predicted_risk_level"]])