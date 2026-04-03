import streamlit as st
import pandas as pd
import plotly.express as px

from src.processing.feature_engineering import build_feature_dataset
from src.models.predict import predict_risk
from src.alerts.alert_engine import generate_alerts

st.set_page_config(page_title="Project Risk Dashboard", layout="wide")
st.title("Project Risk Early Warning Dashboard")

base_df = build_feature_dataset()
pred_df = predict_risk(base_df)
alerts = generate_alerts()
alert_df = pd.DataFrame(alerts) if alerts else pd.DataFrame()

def show_alert_popups(alert_df: pd.DataFrame):
    if alert_df.empty:
        return

    for _, row in alert_df.iterrows():
        message = f"{row['project_name']} ({row['severity']}): {'; '.join(row['reasons'])}"
        details = f"Recommended: {row['recommended_action']}"

        # Prefer st.toast when available, fallback to message-level UI
        try:
            st.toast(message + ' -- ' + details)
        except Exception:
            if row['severity'] == 'Critical':
                st.error(message)
            elif row['severity'] == 'High':
                st.warning(message)
            else:
                st.info(message)
            st.caption(details)


show_alert_popups(alert_df)
st.sidebar.header("Add or Validate Project Record")
with st.sidebar.form("add_record_form"):
    project_id = st.text_input("Project ID")
    project_name = st.text_input("Project Name")
    project_manager = st.text_input("Project Manager")
    practice = st.selectbox("Practice", options=sorted(base_df['practice'].dropna().unique().tolist()) if 'practice' in base_df else ['Practice A', 'Practice B'])
    week_id = st.text_input("Week ID", value="2026-W01")

    cost_variance_pct = st.number_input("Cost Variance %", value=0.0)
    effort_variance_pct = st.number_input("Effort Variance %", value=0.0)
    schedule_variance_pct = st.number_input("Schedule Variance %", value=0.0)
    defect_count = st.number_input("Defect Count", value=0)
    change_requests = st.number_input("Change Requests", value=0)
    resource_attrition = st.number_input("Resource Attrition", value=0.0)
    customer_escalation = st.selectbox("Customer Escalation", options=[0, 1], index=0)
    resource_utilization_pct = st.number_input("Resource Utilization %", value=100.0)
    risk_category = st.selectbox("Risk Category", options=sorted(base_df['risk_category'].dropna().unique().tolist()) if 'risk_category' in base_df else ['Low', 'Medium', 'High'])

    submitted = st.form_submit_button("Add/Validate Record")

if submitted:
    validation_errors = []
    if not project_id:
        validation_errors.append("Project ID is required.")
    if not project_name:
        validation_errors.append("Project Name is required.")
    if not project_manager:
        validation_errors.append("Project Manager is required.")
    if cost_variance_pct < -100 or cost_variance_pct > 1000:
        validation_errors.append("Cost variance percent seems unrealistic (-100 to 1000).")
    if effort_variance_pct < -100 or effort_variance_pct > 1000:
        validation_errors.append("Effort variance percent seems unrealistic (-100 to 1000).")
    if resource_utilization_pct < 0 or resource_utilization_pct > 1000:
        validation_errors.append("Resource utilization percent seems unrealistic (0 to 1000).")

    if validation_errors:
        for err in validation_errors:
            st.sidebar.error(err)
    else:
        new_row = {
            "project_id": project_id,
            "project_name": project_name,
            "project_manager": project_manager,
            "practice": practice,
            "week_id": week_id,
            "cost_variance_pct": cost_variance_pct,
            "effort_variance_pct": effort_variance_pct,
            "schedule_variance_pct": schedule_variance_pct,
            "defect_count": defect_count,
            "change_requests": change_requests,
            "resource_attrition": resource_attrition,
            "customer_escalation": customer_escalation,
            "resource_utilization_pct": resource_utilization_pct,
            "risk_category": risk_category
        }

        combined_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)
        combined_pred_df = predict_risk(combined_df)
        combined_alerts = generate_alerts(combined_df)

        # Find alert(s) for the added record
        added_alerts = [a for a in combined_alerts if a['project_id'] == project_id]

        if added_alerts:
            st.sidebar.success(f"Record \"{project_name}\" added and alert(s) generated.")
            added_alert_df = pd.DataFrame(added_alerts)
            show_alert_popups(added_alert_df)
            st.sidebar.dataframe(added_alert_df)
        else:
            st.sidebar.success(f"Record \"{project_name}\" added and no alerts generated.")

        # Update the main tables to include the new row
        pred_df = combined_pred_df
        alert_df = pd.DataFrame(combined_alerts) if combined_alerts else pd.DataFrame()
col1, col2, col3 = st.columns(3)
col1.metric("Total Projects Records", len(pred_df))
col2.metric("High/Critical Alerts", len(pred_df[pred_df["predicted_risk_level"].isin(["High", "Critical"])]))
col3.metric("Customer Escalations", int(pred_df["customer_escalation"].sum()))

st.subheader("Predicted Risk by Project")
fig = px.bar(
    pred_df,
    x="project_name",
    y="predicted_risk_probability",
    color="predicted_risk_level",
    hover_data=["project_manager", "practice", "week_id"]
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cost Variance % vs Effort Variance %")
fig2 = px.scatter(
    pred_df,
    x="cost_variance_pct",
    y="effort_variance_pct",
    size="defect_count",
    color="predicted_risk_level",
    hover_data=["project_name", "project_manager", "practice"]
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Detailed Prediction Table")
st.dataframe(pred_df[[
    "project_id",
    "project_name",
    "project_manager",
    "practice",
    "week_id",
    "cost_variance_pct",
    "effort_variance_pct",
    "schedule_variance_pct",
    "predicted_risk_probability",
    "predicted_risk_level"
]])

st.subheader("Generated Alerts")
if not alert_df.empty:
    st.dataframe(alert_df)
else:
    st.info("No alerts generated.")