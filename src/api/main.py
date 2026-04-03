from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.processing.feature_engineering import build_feature_dataset
from src.models.predict import predict_risk
from src.alerts.alert_engine import generate_alerts

app = FastAPI(title="Project Risk Alert API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Project Risk Alert API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/projects")
def get_projects():
    df = build_feature_dataset()
    return df[[
        "project_id",
        "project_name",
        "project_manager",
        "practice",
        "week_id",
        "cost_variance_pct",
        "effort_variance_pct",
        "schedule_variance_pct"
    ]].to_dict(orient="records")


@app.get("/predictions")
def get_predictions():
    df = build_feature_dataset()
    result = predict_risk(df)
    return result[[
        "project_id",
        "project_name",
        "week_id",
        "predicted_risk_probability",
        "predicted_risk_level"
    ]].to_dict(orient="records")


@app.get("/alerts")
def get_alerts():
    return generate_alerts()