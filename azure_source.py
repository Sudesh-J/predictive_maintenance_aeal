"""
Send sensor readings to the Azure Machine Learning online endpoint.

The model stays in the student workspace. This module does not download it
and does not read models/irrigation_rf_model.joblib.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from edge_inference import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "azure" / "workspace.json"
ENDPOINT_NAME = "aeal-irrig-score"
DEPLOYMENT_NAME = "blue"


class AzureSourceError(RuntimeError):
    """The dashboard could not use the student Azure ML endpoint."""


class AzureEndpointClient:
    """Posts one reading to the deployed model and returns its prediction."""

    def __init__(self, scoring_uri: str, api_key: str) -> None:
        self.scoring_uri = scoring_uri
        self.api_key = api_key

    def predict_reading(self, reading: dict) -> dict:
        payload = {name: float(reading[name]) for name in FEATURE_COLUMNS}
        request = urllib.request.Request(
            self.scoring_uri,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AzureSourceError(
                f"The Azure endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AzureSourceError(
                "The dashboard could not reach the Azure ML endpoint. "
                f"{exc.reason}"
            ) from exc

        result = json.loads(body)
        if isinstance(result, str):
            result = json.loads(result)
        required = ["irrigation_required", "label", "probability_required", "confidence"]
        missing = [name for name in required if name not in result]
        if missing:
            raise AzureSourceError(
                f"The Azure endpoint response is missing {missing}: {result}"
            )
        return result


def load_workspace_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise AzureSourceError(f"Azure workspace config was not found at {CONFIG_PATH}.")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _az_executable() -> str:
    found = shutil.which("az")
    if not found:
        raise AzureSourceError(
            "Azure CLI is not installed. Install it, then sign in with "
            "the Azure for Students account."
        )
    return found


def _az(args: list[str]) -> str:
    try:
        result = subprocess.run(
            [_az_executable(), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AzureSourceError(
            "Azure CLI is not installed. Install it, then sign in with "
            "the Azure for Students account."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AzureSourceError(detail or "An Azure CLI command failed.")
    return result.stdout


def require_student_account() -> dict:
    """Stop unless the active CLI subscription is the student one."""
    account = json.loads(_az(["account", "show", "-o", "json"]))
    config = load_workspace_config()
    expected = config["subscription_name"]
    actual = account.get("name")
    if actual != expected:
        raise AzureSourceError(
            f"The Azure CLI is using '{actual}', not '{expected}'. "
            "The dashboard only talks to the student subscription. Run:\n"
            f'az account set --subscription "{expected}"'
        )
    user = (account.get("user") or {}).get("name", "")
    if "students.mituniversity.edu.in" not in user:
        raise AzureSourceError(
            f"The signed-in user is '{user}'. Sign in with the student account "
            "before opening the dashboard."
        )
    return account


def _workspace_args(config: dict, subscription_id: str) -> list[str]:
    return [
        "--resource-group",
        config["resource_group"],
        "--workspace-name",
        config["workspace_name"],
        "--subscription",
        subscription_id,
    ]


def _fetch_metrics(config: dict, subscription_id: str) -> dict:
    run_url = (
        f"https://{config['location']}.api.azureml.ms/mlflow/v2.0/"
        f"subscriptions/{subscription_id}/resourceGroups/{config['resource_group']}/"
        "providers/Microsoft.MachineLearningServices/workspaces/"
        f"{config['workspace_name']}/api/2.0/mlflow/runs/get"
        f"?run_id={config['job_name']}"
    )
    payload = json.loads(
        _az(["rest", "--method", "get", "--url", run_url, "--resource", "https://ml.azure.com"])
    )
    metric_rows = payload["run"]["data"]["metrics"]
    values = {row["key"]: row["value"] for row in metric_rows}
    required = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise AzureSourceError(f"The Azure ML run is missing metrics: {missing}")
    return {
        "accuracy": float(values["accuracy"]),
        "precision": float(values["precision"]),
        "recall": float(values["recall"]),
        "f1": float(values["f1"]),
        "confusion_matrix": [
            [int(values["true_negative"]), int(values["false_positive"])],
            [int(values["false_negative"]), int(values["true_positive"])],
        ],
        "confusion_matrix_layout": "rows are actual [0, 1], columns are predicted [0, 1]",
        "model_name": config["model_name"],
        "mlflow_model_version": config["model_version"],
        "job_name": config["job_name"],
        "workspace_name": config["workspace_name"],
        "resource_group": config["resource_group"],
        "source": "azure_ml",
    }


def load_azure_model_and_metrics() -> tuple[AzureEndpointClient, dict, dict]:
    """Connect to the deployed model and read its test metrics from Azure ML."""
    config = load_workspace_config()
    account = require_student_account()
    subscription_id = account["id"]
    common = _workspace_args(config, subscription_id)
    endpoint = json.loads(
        _az(["ml", "online-endpoint", "show", "--name", ENDPOINT_NAME, *common, "-o", "json"])
    )
    if endpoint.get("provisioning_state") != "Succeeded":
        raise AzureSourceError(
            f"Endpoint {ENDPOINT_NAME} is {endpoint.get('provisioning_state')}. "
            "Wait until the Azure deployment finishes, then refresh the dashboard."
        )
    credentials = json.loads(
        _az(["ml", "online-endpoint", "get-credentials", "--name", ENDPOINT_NAME, *common, "-o", "json"])
    )
    scoring_uri = endpoint.get("scoring_uri")
    api_key = credentials.get("primaryKey") or credentials.get("key")
    if not scoring_uri or not api_key:
        raise AzureSourceError(
            f"Endpoint {ENDPOINT_NAME} does not have a scoring URI yet. "
            "The deployment may still be starting."
        )
    client = AzureEndpointClient(scoring_uri, api_key)
    metrics = _fetch_metrics(config, subscription_id)
    info = {
        "subscription_name": account["name"],
        "user": account.get("user", {}).get("name", ""),
        "workspace_name": config["workspace_name"],
        "resource_group": config["resource_group"],
        "model_name": config["model_name"],
        "model_version": config["model_version"],
        "job_name": config["job_name"],
        "endpoint_name": ENDPOINT_NAME,
        "deployment_name": DEPLOYMENT_NAME,
        "scoring_uri": scoring_uri,
    }
    return client, metrics, info
