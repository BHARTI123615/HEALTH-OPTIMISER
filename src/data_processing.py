"""Validation and feature preparation for synthetic demo patient profiles."""
from __future__ import annotations

import math
import pandas as pd

REQUIRED_FEATURES = ["age", "bmi", "heart_rate", "systolic_bp", "symptom_score"]

# Software-demo limits only. These are not clinical reference ranges.
FEATURE_BOUNDS = {
    "age": (18, 100),
    "bmi": (10, 80),
    "heart_rate": (30, 220),
    "systolic_bp": (50, 250),
    "symptom_score": (0, 10),
}


def validate_patient(patient: dict) -> pd.DataFrame:
    """Validate required synthetic inputs and return a one-row feature DataFrame."""
    if not isinstance(patient, dict):
        raise ValueError("Patient profile must be a dictionary of named features.")

    missing = [feature for feature in REQUIRED_FEATURES if feature not in patient]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    frame = pd.DataFrame([{feature: patient[feature] for feature in REQUIRED_FEATURES}])
    if frame[REQUIRED_FEATURES].isnull().any().any():
        raise ValueError("Patient inputs cannot be empty.")
    if not all(pd.api.types.is_numeric_dtype(frame[feature]) for feature in REQUIRED_FEATURES):
        raise ValueError("Patient features must be numeric.")

    for feature, (low, high) in FEATURE_BOUNDS.items():
        value = frame.at[0, feature]
        if not math.isfinite(float(value)):
            raise ValueError(f"{feature} must be a finite number.")
        if not low <= value <= high:
            raise ValueError(f"{feature} must be between {low} and {high} (software-demo bounds).")
    return frame


def feature_vector(patient: dict | pd.DataFrame) -> list[float]:
    """Return model features in REQUIRED_FEATURES order after validation."""
    frame = patient if isinstance(patient, pd.DataFrame) else validate_patient(patient)
    if list(frame[REQUIRED_FEATURES].columns) != REQUIRED_FEATURES or len(frame) != 1:
        raise ValueError("Expected one validated patient row with all required features.")
    return [float(frame.iloc[0][name]) for name in REQUIRED_FEATURES]
