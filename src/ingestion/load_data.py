import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"


def load_csv(file_name: str) -> pd.DataFrame:
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def load_all_data():
    projects = load_csv("projects.csv")
    weekly_metrics = load_csv("weekly_project_metrics.csv")
    resource_allocation = load_csv("resource_allocation.csv")
    historical_outcomes = load_csv("historical_outcomes.csv")

    return {
        "projects": projects,
        "weekly_metrics": weekly_metrics,
        "resource_allocation": resource_allocation,
        "historical_outcomes": historical_outcomes,
    }


if __name__ == "__main__":
    data = load_all_data()
    for name, df in data.items():
        print(f"\n{name.upper()}\n", df.head())