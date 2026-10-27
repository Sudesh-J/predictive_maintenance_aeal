"""
Simulated soil and weather sensors for the live dashboard.

Nothing in this file reads hardware. Every value is drawn on the laptop so a
demonstration can show normal, dry, and wet conditions on demand.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

SCENARIOS = {
    "normal": {
        "title": "Normal conditions",
        "description": (
            "Comfortable moisture, mild temperature, and little or no rain. "
            "Irrigation is usually not required."
        ),
        "ranges": {
            "soil_moisture": (45.0, 58.0),
            "temperature": (24.0, 31.0),
            "humidity": (52.0, 72.0),
            "rainfall": (0.0, 0.8),
        },
    },
    "dry_soil": {
        "title": "Dry soil",
        "description": (
            "Low soil moisture, hot dry air, and almost no rain. "
            "Irrigation is usually required."
        ),
        "ranges": {
            "soil_moisture": (10.0, 24.0),
            "temperature": (33.0, 40.0),
            "humidity": (22.0, 40.0),
            "rainfall": (0.0, 0.3),
        },
    },
    "high_moisture": {
        "title": "High moisture / rainfall",
        "description": (
            "Wet soil after rainfall. Irrigation is usually not required."
        ),
        "ranges": {
            "soil_moisture": (68.0, 90.0),
            "temperature": (20.0, 28.0),
            "humidity": (75.0, 95.0),
            "rainfall": (6.0, 18.0),
        },
    },
}

FEATURE_ORDER = ["soil_moisture", "temperature", "humidity", "rainfall"]


def _round_feature(name: str, value: float) -> float:
    if name == "rainfall":
        return round(float(value), 2)
    return round(float(value), 1)


class SensorSimulator:
    """Walk sensor values inside the selected scenario.

    Changing the scenario jumps to the new range on the next reading, so the
    demonstration does not have to wait for a slow drift.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.previous: dict | None = None
        self.previous_scenario: str | None = None

    def read(self, scenario: str) -> dict:
        if scenario not in SCENARIOS:
            known = ", ".join(SCENARIOS)
            raise ValueError(f"Unknown scenario '{scenario}'. Choose one of: {known}.")

        ranges = SCENARIOS[scenario]["ranges"]
        jump = self.previous is None or scenario != self.previous_scenario
        reading = {}

        for name in FEATURE_ORDER:
            low, high = ranges[name]
            if jump or self.previous is None:
                value = float(self.rng.uniform(low, high))
            else:
                center = (low + high) / 2
                span = (high - low) / 2
                previous_value = float(self.previous[name])
                noise = float(self.rng.normal(0, max(span * 0.12, 0.05)))
                value = previous_value + (center - previous_value) * 0.35 + noise
            value = float(np.clip(value, low, high))
            reading[name] = _round_feature(name, value)

        reading["timestamp"] = datetime.now().isoformat(timespec="seconds")
        reading["scenario"] = scenario
        reading["data_source"] = "simulated"
        self.previous = reading
        self.previous_scenario = scenario
        return reading
