"""Score one irrigation reading inside Azure Machine Learning.

The registered model stays on this endpoint. The laptop sends four sensor
numbers and receives the prediction. This script does not control equipment.
"""

from __future__ import annotations

import json
import os

import mlflow.sklearn
import pandas as pd

FEATURE_COLUMNS = ["soil_moisture", "temperature", "humidity", "rainfall"]
MODEL = None


def _model_dir() -> str:
    root = os.environ["AZUREML_MODEL_DIR"]
    for current, _dirs, files in os.walk(root):
        if "MLmodel" in files:
            return current
    raise FileNotFoundError(f"No MLflow model was found under {root}.")


def init() -> None:
    global MODEL
    MODEL = mlflow.sklearn.load_model(_model_dir())


def _rows(payload: dict) -> pd.DataFrame:
    if all(name in payload for name in FEATURE_COLUMNS):
        values = [[float(payload[name]) for name in FEATURE_COLUMNS]]
        return pd.DataFrame(values, columns=FEATURE_COLUMNS)

    table = payload.get("input_data", payload)
    columns = table.get("columns", FEATURE_COLUMNS)
    data = table.get("data")
    if data is None:
        raise ValueError(
            "Send soil_moisture, temperature, humidity, and rainfall, "
            "or an input_data table with those columns."
        )
    frame = pd.DataFrame(data, columns=columns)
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"The request is missing columns: {missing}")
    return frame[FEATURE_COLUMNS].astype(float)


def run(raw_data: str) -> dict:
    if MODEL is None:
        init()
    payload = json.loads(raw_data)
    frame = _rows(payload)
    predicted = int(MODEL.predict(frame)[0])
    class_list = list(MODEL.classes_)
    required_index = class_list.index(1)
    probability_required = float(MODEL.predict_proba(frame)[0][required_index])
    confidence = probability_required if predicted == 1 else 1.0 - probability_required
    return {
        "irrigation_required": predicted,
        "label": "REQUIRED" if predicted == 1 else "NOT REQUIRED",
        "probability_required": probability_required,
        "confidence": confidence,
    }
