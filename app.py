"""
Streamlit dashboard for the local irrigation demo.

Sensor values come from sensor_simulator.py. Predictions come from the joblib
model through edge_inference.py. MLflow can stay closed while this app runs.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from edge_inference import IrrigationPredictor, load_metrics
from sensor_simulator import SCENARIOS, SensorSimulator

HISTORY_LIMIT = 45
READING_INTERVAL = timedelta(seconds=2)


st.set_page_config(
    page_title="Smart Agriculture Irrigation",
    layout="wide",
)


@st.cache_resource
def load_predictor(model_mtime: float) -> IrrigationPredictor:
    # model_mtime is part of the cache key, so a retrain is picked up on refresh.
    del model_mtime
    return IrrigationPredictor()


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator()


def append_reading(predictor: IrrigationPredictor, scenario: str) -> None:
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
    from edge_inference import MODEL_PATH

    predictor = load_predictor(MODEL_PATH.stat().st_mtime)
    render_live(predictor, st.session_state.scenario, st.session_state.running)


def render_evaluation(metrics: dict | None) -> None:
    st.subheader("Held-out test results")
    if metrics is None:
        st.info("metrics.json was not found. Run train_model.py to create it.")
        return

    st.caption(
        "These scores measure the model on simulated rows it did not train on. "
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


def render_live(predictor: IrrigationPredictor, scenario: str, running: bool) -> None:
    if running:
        append_reading(predictor, scenario)

    history = st.session_state.history
    if not history:
        st.info(
            "Turn on live simulation in the sidebar. A new simulated reading "
            "is created every 2 seconds, then scored on this laptop."
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
        f"Source: {latest['data_source']} sensor values. "
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
        "The Random Forest runs on this laptop."
    )
    st.warning(
        "Simulated data only. These are not measurements from real sensors or a "
        "real farm, and this dashboard does not control irrigation equipment."
    )

    from edge_inference import MODEL_PATH

    if not MODEL_PATH.is_file():
        st.error(
            "Trained model not found at models/irrigation_rf_model.joblib. "
            "Generate the dataset and train the model first."
        )
        st.code(
            ".\\.venv\\Scripts\\python.exe generate_dataset.py\n"
            ".\\.venv\\Scripts\\python.exe train_model.py",
            language="powershell",
        )
        st.stop()

    try:
        load_predictor(MODEL_PATH.stat().st_mtime)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    metrics = load_metrics()

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

        st.header("Model")
        st.write("Algorithm: Random Forest")
        if metrics:
            st.write(f"Test accuracy: {metrics['accuracy']:.1%}")
            version = metrics.get("mlflow_model_version")
            if version:
                st.write(f"MLflow model version: {version}")
                st.caption("MLflow alias: champion")
        st.caption("Predictions load models/irrigation_rf_model.joblib. MLflow can be closed.")

    live_panel()

    with st.expander("Model evaluation on the test set", expanded=False):
        render_evaluation(metrics)


if __name__ == "__main__":
    main()
