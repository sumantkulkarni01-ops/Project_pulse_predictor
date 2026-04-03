from src.processing.feature_engineering import build_feature_dataset
from src.models.predict import predict_risk


from typing import Optional


def generate_alerts(df: Optional[object] = None):
    if df is None:
        df = build_feature_dataset()
    prediction_df = predict_risk(df)

    alerts = []

    for _, row in prediction_df.iterrows():
        reasons = []
        severity = row["predicted_risk_level"]

        if row["cost_variance_pct"] > 10:
            reasons.append(f"Cost variance is {row['cost_variance_pct']:.2f}%")

        if row["effort_variance_pct"] > 10:
            reasons.append(f"Effort variance is {row['effort_variance_pct']:.2f}%")

        if row["schedule_variance_pct"] > 5:
            reasons.append(f"Schedule slippage is {row['schedule_variance_pct']:.2f}%")

        if row["customer_escalation"] == 1:
            reasons.append("Customer escalation exists")

        if row["resource_utilization_pct"] > 110:
            reasons.append(f"Resource utilization is {row['resource_utilization_pct']:.2f}%")

        if reasons:
            alerts.append({
                "project_id": row["project_id"],
                "project_name": row["project_name"],
                "project_manager": row["project_manager"],
                "practice": row["practice"],
                "week_id": row["week_id"],
                "severity": severity,
                "predicted_risk_probability": round(float(row["predicted_risk_probability"]), 2),
                "reasons": reasons,
                "recommended_action": recommend_action(severity)
            })

    return alerts


def recommend_action(severity: str) -> str:
    if severity == "Critical":
        return "Immediate governance review within 24 hours and freeze non-essential changes."
    if severity == "High":
        return "Review project baseline, staffing, and cost leakages within 48 hours."
    if severity == "Medium":
        return "Monitor weekly and validate effort/cost trends with project manager."
    return "Continue normal monitoring."


if __name__ == "__main__":
    all_alerts = generate_alerts()
    for alert in all_alerts:
        print(alert)