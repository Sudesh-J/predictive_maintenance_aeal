"""
Train a Random Forest on the simulated dataset and track the run in MLflow.

The fit uses only the training split. Metrics are calculated on the held-out
test split. A joblib copy is saved under models/ so the dashboard can predict
without starting the MLflow server.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from edge_inference import FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "agriculture_sensor_data.csv"
INFO_PATH = ROOT / "data" / "dataset_info.json"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "irrigation_rf_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "smart_agriculture_irrigation"
REGISTERED_MODEL_NAME = "smart-agri-irrigation-predictor"

MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    # One worker keeps the result stable across machines with different CPUs.
    "n_jobs": 1,
}


def _load_training_frame() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise SystemExit(
            "Dataset not found at "
            f"{DATA_PATH}. From the project folder run:\n"
            "  .\\.venv\\Scripts\\python.exe generate_dataset.py"
        )

    data = pd.read_csv(DATA_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(
            f"Dataset is missing columns {missing}. "
            "Run generate_dataset.py again so the CSV matches the trainer."
        )
    return data


def _setup_mlflow() -> str:
    """Point MLflow at a local SQLite file so the Model Registry works offline."""
    database = (ROOT / "mlflow.db").resolve()
    # Spaces must stay as spaces. Encoding them makes SQLite look for a
    # folder named with "%20", which does not exist.
    tracking_uri = "sqlite:///" + database.as_posix()
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_location = (ROOT / "mlartifacts").resolve().as_uri()
        client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    elif experiment.lifecycle_stage == "deleted":
        raise SystemExit(
            "The MLflow experiment was deleted. Remove mlflow.db and train again."
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def _register_model(model_uri: str) -> str | None:
    """Register the run in the local MLflow Model Registry.

    Returns the version string, or None when this MLflow install cannot
    register models. The joblib file is already saved in either case.
    """
    try:
        version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    except Exception as exc:
        print("MLflow Model Registry is not available in this setup.")
        print(f"The experiment run is still logged. Details: {exc}")
        return None

    client = MlflowClient()
    description = (
        "Random Forest that predicts whether simulated farm conditions need "
        "irrigation. Trained for a college IoT and edge-computing demonstration. "
        "Not for controlling real irrigation equipment."
    )
    try:
        client.update_registered_model(REGISTERED_MODEL_NAME, description=description)
        client.update_model_version(
            REGISTERED_MODEL_NAME, version.version, description=description
        )
    except Exception as exc:
        print(f"Registered version {version.version}, but could not set the description: {exc}")

    try:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", version.version
        )
        print(f"Registered model alias 'champion' -> version {version.version}.")
    except Exception as exc:
        print(f"Version {version.version} was registered, but the alias was not set: {exc}")

    return str(version.version)


def train() -> dict:
    data = _load_training_frame()
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].astype(int)

    # The split happens before fit. Only the training rows are used to build trees.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)
    importances = {
        name: float(value)
        for name, value in zip(FEATURE_COLUMNS, model.feature_importances_)
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model_name": REGISTERED_MODEL_NAME,
        "algorithm": "RandomForestClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "positive_class_meaning": "irrigation required",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_layout": "rows are actual [0, 1], columns are predicted [0, 1]",
        "feature_importances": {key: round(value, 4) for key, value in importances.items()},
        "parameters": MODEL_PARAMS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_note": "Metrics are from a held-out split of simulated, rule-labeled data. They are not real-farm accuracy.",
        "hardware_note": "This model does not connect to irrigation equipment.",
        "model_file": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tracking_uri = _setup_mlflow()
    with mlflow.start_run(run_name="random_forest_irrigation") as run:
        mlflow.log_params(
            {
                **{key: value for key, value in MODEL_PARAMS.items()},
                "test_size": TEST_SIZE,
                "n_samples": int(len(data)),
                "n_features": len(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            }
        )
        for name, value in importances.items():
            mlflow.log_metric(f"importance_{name}", float(value))

        mlflow.set_tag("data_source", "simulated")
        mlflow.set_tag("deployment", "local_laptop")
        mlflow.set_tag("task", "irrigation_required_classification")

        mlflow.log_dict(
            {
                "confusion_matrix": metrics["confusion_matrix"],
                "layout": metrics["confusion_matrix_layout"],
            },
            "confusion_matrix.json",
        )
        if INFO_PATH.is_file():
            mlflow.log_artifact(str(INFO_PATH), artifact_path="dataset")
        mlflow.log_artifact(str(DATA_PATH), artifact_path="dataset")

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="irrigation_model",
            signature=signature,
            input_example=x_train.head(2),
        )
        model_uri = f"runs:/{run.info.run_id}/irrigation_model"
        version = _register_model(model_uri)
        metrics["mlflow_run_id"] = run.info.run_id
        metrics["mlflow_model_version"] = version
        metrics["mlflow_tracking_uri"] = tracking_uri

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_report(metrics, matrix)
    return metrics


def _print_report(metrics: dict, matrix) -> None:
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    print("")
    print("Training finished. Metrics are from the held-out test set only.")
    print("The model was not fit on those rows.")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}  (class 1 = irrigation required)")
    print(f"Recall:    {metrics['recall']:.4f}  (class 1 = irrigation required)")
    print(f"F1 score:  {metrics['f1']:.4f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 Pred 0   Pred 1")
    print(f"Actual 0         {tn:6d}   {fp:6d}")
    print(f"Actual 1         {fn:6d}   {tp:6d}")
    print("Feature importances:")
    for name, value in metrics["feature_importances"].items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved local model to {metrics['model_file']}")
    if metrics.get("mlflow_model_version"):
        print(
            f"MLflow registered '{REGISTERED_MODEL_NAME}' "
            f"version {metrics['mlflow_model_version']} "
            f"(run {metrics['mlflow_run_id']})."
        )
    print("")
    print("The dashboard loads the joblib file and does not need MLflow to stay open.")
    print("To inspect the experiment and registered model, from this folder run:")
    print(
        "  .\\.venv\\Scripts\\python.exe -m mlflow ui "
        "--backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000"
    )
    print("Then open http://127.0.0.1:5000")


if __name__ == "__main__":
    train()
