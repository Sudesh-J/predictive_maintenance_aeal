"""
Streamlit dashboard for the irrigation demo.

Sensor values are simulated on this laptop. Each reading is sent to the
model deployed in Azure Machine Learning, and the prediction comes back
from that endpoint.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from azure_source import AzureEndpointClient, AzureSourceError, load_azure_model_and_metrics
from sensor_simulator import SCENARIOS, SensorSimulator

HISTORY_LIMIT = 45
READING_INTERVAL = timedelta(seconds=2)


st.set_page_config(
    page_title="Smart Agriculture Irrigation",
    layout="wide",
)


@st.cache_resource
def load_azure_context() -> tuple[AzureEndpointClient, dict, dict]:
    """Connect once per dashboard process. Each reading is still sent to Azure."""
    return load_azure_model_and_metrics()


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator()


def append_reading(predictor: AzureEndpointClient, scenario: str) -> None:
    reading = st.session_state.simulator.read(scenario)
    prediction = predictor.predict_reading(reading)
    st.session_state.history.append({**reading, **prediction})
    if len(st.session_state.history) > HISTORY_LIMIT:
        st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]


def sensor_chart(history: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=(
            "Soil moisture (%)",
            "Temperature (°C)",
            "Humidity (%)",
            "Rainfall (mm)",
        ),
    )
    times = history["timestamp"]
    traces = [
        ("soil_moisture", 1, 1, "#1b7f4e"),
        ("temperature", 1, 2, "#e07a3d"),
        ("humidity", 2, 1, "#3d7ea6"),
        ("rainfall", 2, 2, "#5c6bc0"),
    ]
    for column, row, col, color in traces:
        figure.add_trace(
            go.Scatter(
                x=times,
                y=history[column],
                mode="lines+markers",
                name=column.replace("_", " "),
                line={"color": color, "width": 2},
                marker={"size": 6},
            ),
            row=row,
            col=col,
        )
    figure.update_layout(
        height=560,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        template="plotly_white",
        showlegend=False,
    )
    return figure


def confusion_chart(matrix: list) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted: not required", "Predicted: required"],
            y=["Actual: not required", "Actual: required"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Greens",
            showscale=False,
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        template="plotly_white",
    )
    return figure


@st.fragment(run_every=READING_INTERVAL)
def live_panel() -> None:
    """Refresh the sensor panel every 2 seconds without restarting the page."""
    predictor, _metrics, _info = load_azure_context()
    try:
        render_live(predictor, st.session_state.scenario, st.session_state.running)
    except AzureSourceError as exc:
        st.error(str(exc))


def render_evaluation(metrics: dict | None) -> None:
    st.subheader("Held-out test results")
    st.caption(
        "These scores were logged by Azure ML job "
        f"{metrics['job_name']} in workspace {metrics['workspace_name']}. "
        "They are not accuracy on a real farm."
    )
    accuracy, precision, recall, f1 = st.columns(4)
    accuracy.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    precision.metric("Precision", f"{metrics['precision']:.1%}")
    recall.metric("Recall", f"{metrics['recall']:.1%}")
    f1.metric("F1 score", f"{metrics['f1']:.1%}")
    st.plotly_chart(
        confusion_chart(metrics["confusion_matrix"]),
        use_container_width=True,
        key="confusion_matrix",
    )
    st.caption(metrics.get("confusion_matrix_layout", ""))


def render_live(predictor: AzureEndpointClient, scenario: str, running: bool) -> None:
    if running:
        append_reading(predictor, scenario)

    history = st.session_state.history
    if not history:
        st.info(
            "Turn on live simulation in the sidebar. A new simulated reading "
            "is sent to the Azure ML endpoint every 2 seconds."
        )
        return

    latest = history[-1]
    if latest["irrigation_required"] == 1:
        st.error(
            "IRRIGATION REQUIRED. This is a simulated recommendation only. "
            "No pump, valve, or other equipment is activated."
        )
    else:
        st.success("Irrigation is NOT REQUIRED for this simulated reading.")

    moisture, temperature, humidity, rainfall = st.columns(4)
    moisture.metric("Soil moisture", f"{latest['soil_moisture']:.1f}%")
    temperature.metric("Temperature", f"{latest['temperature']:.1f} °C")
    humidity.metric("Humidity", f"{latest['humidity']:.1f}%")
    rainfall.metric("Rainfall", f"{latest['rainfall']:.2f} mm")

    status, probability, confidence = st.columns(3)
    status.metric("Predicted irrigation", latest["label"])
    probability.metric(
        "Probability required",
        f"{latest['probability_required']:.1%}",
    )
    confidence.metric("Prediction confidence", f"{latest['confidence']:.1%}")
    st.caption(
        "Sensor values are simulated on this laptop and sent to the deployed model. "
        "The prediction is the response from Azure Machine Learning. "
        f"Scenario: {SCENARIOS[latest['scenario']]['title']}. "
        f"Reading time: {latest['timestamp']}."
    )

    frame = pd.DataFrame(history)
    st.subheader("Sensor history")
    st.plotly_chart(sensor_chart(frame), use_container_width=True, key="sensor_chart")

    st.subheader("Recent predictions")
    recent = frame.iloc[::-1][
        [
            "timestamp",
            "scenario",
            "soil_moisture",
            "temperature",
            "humidity",
            "rainfall",
            "label",
            "probability_required",
            "data_source",
        ]
    ].head(10)
    recent = recent.copy()
    recent["probability_required"] = recent["probability_required"].map(
        lambda value: f"{float(value):.1%}"
    )
    recent = recent.rename(
        columns={
            "timestamp": "Time",
            "scenario": "Scenario",
            "soil_moisture": "Moisture %",
            "temperature": "Temp °C",
            "humidity": "Humidity %",
            "rainfall": "Rain mm",
            "label": "Prediction",
            "probability_required": "P(required)",
            "data_source": "Source",
        }
    )
    st.dataframe(recent, use_container_width=True, hide_index=True, key="recent_table")


def main() -> None:
    init_state()

    st.title("Smart Agriculture AI Irrigation Prediction System")
    st.markdown(
        "Predict whether a field needs irrigation from **simulated** IoT readings. "
        "Each reading is sent to the model **deployed in Azure Machine Learning**. "
        "The model is not downloaded onto this laptop."
    )
    st.warning(
        "Simulated data only. These are not measurements from real sensors or a "
        "real farm, and this dashboard does not control irrigation equipment."
    )

    try:
        _predictor, metrics, azure_info = load_azure_context()
    except (AzureSourceError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.code(
            'az account set --subscription "Azure for Students"\n'
            "az account show --query \"{subscription:name, user:user.name}\" --output table",
            language="powershell",
        )
        st.stop()

    with st.sidebar:
        st.header("Simulation controls")
        st.toggle(
            "Run live simulation",
            value=False,
            key="running",
            help="While this is on, a new simulated reading arrives every 2 seconds.",
        )
        scenario = st.selectbox(
            "Sensor scenario",
            options=list(SCENARIOS),
            format_func=lambda key: SCENARIOS[key]["title"],
            key="scenario",
        )
        st.caption(SCENARIOS[scenario]["description"])
        st.caption("Simulation is paused." if not st.session_state.running else "Simulation is running.")
        if st.button("Clear chart history"):
            st.session_state.history = []
            st.session_state.simulator = SensorSimulator()

        st.header("Azure ML endpoint")
        st.write(f"Subscription: {azure_info['subscription_name']}")
        st.write(f"Workspace: {azure_info['workspace_name']}")
        st.write(f"Endpoint: {azure_info['endpoint_name']}")
        st.write(f"Deployment: {azure_info['deployment_name']}")
        st.write(f"Model: {azure_info['model_name']} v{azure_info['model_version']}")
        st.write(f"Test accuracy: {metrics['accuracy']:.1%}")
        st.caption("Each live reading is posted to this endpoint. The model stays in Azure.")

    live_panel()

    with st.expander("Model evaluation on the test set", expanded=False):
        render_evaluation(metrics)


if __name__ == "__main__":
    main()
