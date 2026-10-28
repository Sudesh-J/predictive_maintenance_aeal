"""
Score one sensor reading with a scikit-learn model.

The dashboard sends readings to the Azure Machine Learning endpoint.
This module never switches a pump, valve, or relay.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "irrigation_rf_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Column order is part of the trained model. Do not sort or rename these.
FEATURE_COLUMNS = ["soil_moisture", "temperature", "humidity", "rainfall"]
TARGET_COLUMN = "irrigation_required"


class IrrigationPredictor:
    def __init__(self, model) -> None:
        self.model = model
        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and int(expected) != len(FEATURE_COLUMNS):
            raise ValueError(
                "The saved model expects "
                f"{expected} features, but this project uses {FEATURE_COLUMNS}. "
                "Retrain it with train_model.py."
            )

        classes = list(getattr(self.model, "classes_", []))
        if 1 not in classes:
            raise ValueError(
                "The saved model has no class 1 (irrigation required). "
                "Regenerate the dataset and train again."
            )
        self._required_index = classes.index(1)

    @classmethod
    def from_estimator(cls, model) -> "IrrigationPredictor":
        return cls(model)

    @classmethod
    def from_joblib(cls, model_path: Path = MODEL_PATH) -> "IrrigationPredictor":
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                "Trained model not found at "
                f"{path}. The dashboard loads the model from Azure ML instead."
            )
        return cls(joblib.load(path))

    def predict_reading(self, reading: dict) -> dict:
        missing = [column for column in FEATURE_COLUMNS if column not in reading]
        if missing:
            raise ValueError(f"Sensor reading is missing values for: {missing}")

        features = pd.DataFrame(
            [{column: float(reading[column]) for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        predicted = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        probability_required = float(probabilities[self._required_index])
        confidence = probability_required if predicted == 1 else 1.0 - probability_required

        return {
            "irrigation_required": predicted,
            "label": "REQUIRED" if predicted == 1 else "NOT REQUIRED",
            "probability_required": probability_required,
            "confidence": confidence,
        }


def load_metrics(path: Path = METRICS_PATH) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Score one fresh simulated reading from each scenario with the Azure model."""
    from azure_source import load_azure_model_and_metrics
    from sensor_simulator import SCENARIOS, SensorSimulator

    print("Sending sample readings to the Azure Machine Learning endpoint.")
    predictor, _metrics, info = load_azure_model_and_metrics()
    print(
        f"Endpoint {info['endpoint_name']} in workspace {info['workspace_name']} "
        f"is serving {info['model_name']} version {info['model_version']}."
    )
    simulator = SensorSimulator(seed=7)

    for scenario in SCENARIOS:
        reading = simulator.read(scenario)
        result = predictor.predict_reading(reading)
        print(
            f"{scenario:16} moisture={reading['soil_moisture']:5.1f}%  "
            f"temp={reading['temperature']:4.1f}C  "
            f"humidity={reading['humidity']:5.1f}%  "
            f"rain={reading['rainfall']:5.2f}mm  ->  "
            f"{result['label']}  "
            f"(P(required)={result['probability_required']:.1%})"
        )


if __name__ == "__main__":
    main()
