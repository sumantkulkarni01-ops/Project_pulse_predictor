import pandas as pd
from src.ingestion.load_data import load_all_data


def build_feature_dataset() -> pd.DataFrame:
    data = load_all_data()
    projects = data["projects"]
    weekly = data["weekly_metrics"]
    resource = data["resource_allocation"]
    outcomes = data["historical_outcomes"]

    resource_summary = (
        resource.groupby(["project_id", "week_start"], as_index=False)
        .agg(
            allocated_hours_total=("allocated_hours", "sum"),
            actual_hours_total=("actual_hours", "sum")
        )
    )

    df = weekly.merge(projects, on="project_id", how="left")
    df = df.merge(resource_summary, on=["project_id", "week_start"], how="left")
    df = df.merge(outcomes, on="project_id", how="left")

    # Avoid divide-by-zero
    df["planned_cost_till_date"] = df["planned_cost_till_date"].replace(0, 1)
    df["planned_effort_till_date"] = df["planned_effort_till_date"].replace(0, 1)
    df["allocated_hours_total"] = df["allocated_hours_total"].fillna(1)

    df["cost_variance"] = df["actual_cost_till_date"] - df["planned_cost_till_date"]
    df["cost_variance_pct"] = (df["cost_variance"] / df["planned_cost_till_date"]) * 100

    df["effort_variance"] = df["actual_effort_till_date"] - df["planned_effort_till_date"]
    df["effort_variance_pct"] = (df["effort_variance"] / df["planned_effort_till_date"]) * 100

    df["schedule_variance_pct"] = df["planned_progress_pct"] - df["actual_progress_pct"]

    df["resource_utilization_pct"] = (
        (df["actual_hours_total"] / df["allocated_hours_total"]) * 100
    )

    df["defect_density"] = df["defect_count"] / (df["actual_effort_till_date"] + 1)

    # Binary risk label for model training
    df["risk_flag"] = ((df["cost_overrun_flag"] == 1) | (df["effort_overrun_flag"] == 1)).astype(int)

    return df


if __name__ == "__main__":
    feature_df = build_feature_dataset()
    print(feature_df[[
        "project_id",
        "week_id",
        "cost_variance_pct",
        "effort_variance_pct",
        "schedule_variance_pct",
        "resource_utilization_pct",
        "risk_flag"
    ]].head())