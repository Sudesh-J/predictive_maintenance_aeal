"""
Load the saved Random Forest and score one sensor reading on this laptop.

Prediction uses the local joblib file. MLflow does not need to be running,
and no cloud prediction service is called. The result is a recommendation
only: this module never switches a pump, valve, or relay.
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
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Trained model not found at "
                f"{self.model_path}. From the project folder run:\n"
                "  .\\.venv\\Scripts\\python.exe generate_dataset.py\n"
                "  .\\.venv\\Scripts\\python.exe train_model.py"
            )

        self.model = joblib.load(self.model_path)
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
    """Score one fresh simulated reading from each scenario."""
    from sensor_simulator import SCENARIOS, SensorSimulator

    print("Local edge inference demo. No hardware is connected.")
    predictor = IrrigationPredictor()
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
