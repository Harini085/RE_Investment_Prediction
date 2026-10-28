"""Shared paths, dataset loading, feature engineering, and MLflow setup."""
from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("REA_DATA_PATH", ROOT / "data" / "raw" / "india_housing_prices.csv"))
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_NAME = "Real Estate Investment Advisor"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place india_housing_prices.csv in data/raw/.")
    return pd.read_csv(path, low_memory=False)


def clean_data(frame):
    """Basic deterministic cleanup; fitted imputing/encoding stays in sklearn pipelines."""
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data.drop_duplicates()
    for col in data.select_dtypes(include="object"):
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
    for col in data.columns:
        if data[col].dtype == "object" or str(data[col].dtype).startswith("string"):
            continue
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    return data


def amenity_count(values):
    if pd.isna(values) or not str(values).strip():
        return np.nan
    return len({part.strip().lower() for part in str(values).split(",") if part.strip()})


def model_features(frame):
    """Prediction-time candidate features, excluding IDs and current-price leakage."""
    data = clean_data(frame)
    if "Amenities" in data:
        data["Amenity_Count"] = data["Amenities"].map(amenity_count)
        data = data.drop(columns="Amenities")
    if "Year_Built" in data and "Age_of_Property" in data:
        data = data.drop(columns="Age_of_Property")
    elif "Year_Built" in data:
        data["Age_of_Property"] = pd.Timestamp.now().year - pd.to_numeric(data["Year_Built"], errors="coerce")
    drop = [c for c in ["ID", "Price_in_Lakhs", "Price_per_SqFt"] if c in data]
    return data.drop(columns=drop)


def get_mlflow():
    """Use a project-local SQLite tracking DB unless explicitly overridden."""
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        database = (ROOT / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
