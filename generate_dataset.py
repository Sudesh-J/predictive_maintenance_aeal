"""
Create a simulated farm dataset for the irrigation classifier.

The rows are invented on this computer. They are not readings from a real
farm, weather station, or soil sensor. Labels follow simple crop-water rules,
with a small amount of random noise so the task is not perfectly separable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "agriculture_sensor_data.csv"
INFO_PATH = DATA_DIR / "dataset_info.json"

RANDOM_SEED = 42
N_SAMPLES = 4000
LABEL_NOISE_RATE = 0.02

# These names are the model inputs. Keep them identical in train_model.py
# and edge_inference.py.
FEATURE_COLUMNS = ["soil_moisture", "temperature", "humidity", "rainfall"]
TARGET_COLUMN = "irrigation_required"

# Each regime is a plausible weather and soil situation. Sampling inside a
# regime is what makes the columns move together, instead of being fully random.
REGIMES = {
    "normal": {
        "weight": 0.30,
        "soil_moisture": (40.0, 58.0),
        "temperature": (22.0, 32.0),
        "humidity": (48.0, 75.0),
        "rainfall": (0.0, 2.5),
        "rainfall_zero_chance": 0.45,
    },
    "dry": {
        "weight": 0.20,
        "soil_moisture": (8.0, 28.0),
        "temperature": (28.0, 40.0),
        "humidity": (18.0, 45.0),
        "rainfall": (0.0, 1.2),
        "rainfall_zero_chance": 0.75,
    },
    "hot_dry": {
        "weight": 0.14,
        "soil_moisture": (30.0, 44.0),
        "temperature": (34.0, 42.0),
        "humidity": (15.0, 40.0),
        "rainfall": (0.0, 0.8),
        "rainfall_zero_chance": 0.80,
    },
    "rainy": {
        "weight": 0.14,
        "soil_moisture": (55.0, 92.0),
        "temperature": (18.0, 30.0),
        "humidity": (72.0, 98.0),
        "rainfall": (5.0, 25.0),
        "rainfall_zero_chance": 0.0,
    },
    "cool_wet_soil": {
        "weight": 0.10,
        "soil_moisture": (62.0, 88.0),
        "temperature": (16.0, 26.0),
        "humidity": (60.0, 90.0),
        "rainfall": (0.0, 4.0),
        "rainfall_zero_chance": 0.35,
    },
    # Soil can still look dry just after rain starts. The label rules then say
    # not to irrigate, so the model has to use rainfall and not only moisture.
    "recent_rain": {
        "weight": 0.12,
        "soil_moisture": (18.0, 50.0),
        "temperature": (20.0, 34.0),
        "humidity": (55.0, 92.0),
        "rainfall": (6.0, 20.0),
        "rainfall_zero_chance": 0.0,
    },
}


def irrigation_required(soil_moisture, temperature, humidity, rainfall) -> int:
    """Return 1 when a typical field crop would be watered, else 0.

    Rules are applied in order:
    1. Wet soil or a meaningful rain event means water is already available.
    2. Dry soil needs water unless rule 1 already handled heavy rain.
    3. Moderately dry soil still needs water in hot, dry air with almost no rain.
    4. Otherwise do not irrigate.
    """
    if soil_moisture >= 60 or rainfall >= 5:
        return 0
    if soil_moisture < 30:
        return 1
    if (
        soil_moisture < 45
        and temperature >= 34
        and humidity < 45
        and rainfall < 1
    ):
        return 1
    return 0


def _check_rules() -> None:
    """Fail fast if the written rules and the function drift apart."""
    examples = [
        (18, 36, 30, 0.0, 1),
        (75, 24, 80, 10.0, 0),
        (50, 27, 60, 0.2, 0),
        (38, 36, 30, 0.0, 1),
        (25, 22, 70, 8.0, 0),
        (44, 30, 55, 0.4, 0),
    ]
    for soil, temp, humidity, rain, expected in examples:
        actual = irrigation_required(soil, temp, humidity, rain)
        if actual != expected:
            raise RuntimeError(
                "Label rule check failed for "
                f"soil={soil}, temp={temp}, humidity={humidity}, rain={rain}: "
                f"expected {expected}, got {actual}."
            )


def _sample_regime(rng: np.random.Generator, name: str, count: int) -> pd.DataFrame:
    spec = REGIMES[name]
    frame = pd.DataFrame({"regime": name}, index=range(count))
    for column in FEATURE_COLUMNS:
        low, high = spec[column]
        frame[column] = rng.uniform(low, high, size=count)

    dry_rain = rng.random(count) < spec["rainfall_zero_chance"]
    frame.loc[dry_rain, "rainfall"] = 0.0

    frame["soil_moisture"] = frame["soil_moisture"].clip(5, 95).round(1)
    frame["temperature"] = frame["temperature"].clip(10, 45).round(1)
    frame["humidity"] = frame["humidity"].clip(10, 100).round(1)
    frame["rainfall"] = frame["rainfall"].clip(0, 40).round(2)
    return frame


def generate_dataset(
    n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    _check_rules()
    rng = np.random.default_rng(seed)

    regime_names = list(REGIMES)
    weights = np.array([REGIMES[name]["weight"] for name in regime_names], dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(regime_names, size=n_samples, p=weights)

    parts = [
        _sample_regime(rng, name, int(np.sum(chosen == name)))
        for name in regime_names
        if np.sum(chosen == name) > 0
    ]
    data = pd.concat(parts, ignore_index=True)
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    labels = [
        irrigation_required(
            row.soil_moisture, row.temperature, row.humidity, row.rainfall
        )
        for row in data.itertuples(index=False)
    ]
    labels = np.array(labels, dtype=int)

    # A few labels are flipped on purpose. This stands in for an uncertain
    # field decision and keeps the classifier from scoring a perfect 100%.
    flip = rng.random(n_samples) < LABEL_NOISE_RATE
    labels = np.where(flip, 1 - labels, labels)

    data.insert(0, "sample_id", np.arange(1, n_samples + 1))
    data[TARGET_COLUMN] = labels
    data["data_source"] = "simulated_not_real_farm"
    return data


def save_dataset(data: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)

    info = {
        "data_source": "simulated",
        "real_farm_data": False,
        "real_sensor_hardware": False,
        "n_samples": int(len(data)),
        "random_seed": RANDOM_SEED,
        "label_noise_rate": LABEL_NOISE_RATE,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "units": {
            "soil_moisture": "percent of field capacity, 0-100",
            "temperature": "degrees Celsius",
            "humidity": "percent relative humidity",
            "rainfall": "millimetres in the recent period",
        },
        "label_rules_in_order": [
            "If soil moisture >= 60% OR rainfall >= 5 mm -> irrigation not required.",
            "Else if soil moisture < 30% -> irrigation required.",
            "Else if soil moisture < 45% AND temperature >= 34 C AND humidity < 45% AND rainfall < 1 mm -> irrigation required.",
            "Else irrigation not required.",
            f"Then about {LABEL_NOISE_RATE:.0%} of labels are flipped at random.",
        ],
        "note": (
            "This file was generated by generate_dataset.py for a college "
            "demonstration. It was not collected from a real farm."
        ),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    print("Generating a SIMULATED agricultural dataset.")
    print("These rows are not measurements from a real farm or real IoT hardware.")
    data = generate_dataset()
    save_dataset(data)

    counts = data[TARGET_COLUMN].value_counts().sort_index()
    print(f"Saved {len(data)} rows to {DATA_PATH.relative_to(ROOT)}")
    print(f"Also wrote {INFO_PATH.relative_to(ROOT)}")
    print("Class counts (0 = not required, 1 = required):")
    for label, count in counts.items():
        print(f"  {label}: {count} ({count / len(data):.1%})")
    print("Feature ranges:")
    print(data[FEATURE_COLUMNS].agg(["min", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
