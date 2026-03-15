import argparse
import csv
import logging
import math
import random
from pathlib import Path

LOG_PATH = Path("logs/data_creation.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

FIELDNAMES = ["day", "seasonality", "trend", "noise", "has_anomaly", "temperature", "target_temperature"]


def _generate_temperature_dataset(
    *,
    seed: int,
    n_days: int,
    base_temp: float,
    trend_per_day: float,
    noise_std: float,
    anomaly_probability: float,
    anomaly_scale: float,
):
    rng = random.Random(seed)
    rows = []

    temperatures = []
    for day in range(n_days):
        seasonality = 8.0 * math.sin(2.0 * math.pi * day / 30.0)
        trend = trend_per_day * day
        noise = rng.gauss(0.0, noise_std)

        temperature = base_temp + seasonality + trend + noise
        has_anomaly = 1 if rng.random() < anomaly_probability else 0
        if has_anomaly:
            temperature += rng.gauss(0.0, anomaly_scale)

        temperatures.append((day, seasonality, trend, noise, has_anomaly, temperature))

    for i in range(len(temperatures) - 1):
        day, seasonality, trend, noise, has_anomaly, temperature = temperatures[i]
        target_temperature = temperatures[i + 1][5]
        rows.append(
            {
                "day": day,
                "seasonality": seasonality,
                "trend": trend,
                "noise": noise,
                "has_anomaly": has_anomaly,
                "temperature": temperature,
                "target_temperature": target_temperature,
            }
        )
    return rows


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def create_data(save_dir: str):
    save_path = Path(save_dir)
    train_dir = save_path / "train"
    test_dir = save_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    datasets_config = [
        {
            "name": "clean_signal",
            "seed": 42,
            "n_days": 160,
            "base_temp": 12.0,
            "trend_per_day": 0.03,
            "noise_std": 0.7,
            "anomaly_probability": 0.00,
            "anomaly_scale": 0.0,
        },
        {
            "name": "noisy_signal",
            "seed": 123,
            "n_days": 160,
            "base_temp": 10.0,
            "trend_per_day": 0.02,
            "noise_std": 2.4,
            "anomaly_probability": 0.00,
            "anomaly_scale": 0.0,
        },
        {
            "name": "anomalous_signal",
            "seed": 777,
            "n_days": 160,
            "base_temp": 11.0,
            "trend_per_day": 0.015,
            "noise_std": 1.1,
            "anomaly_probability": 0.08,
            "anomaly_scale": 14.0,
        },
    ]

    for cfg in datasets_config:
        name = cfg["name"]
        rows = _generate_temperature_dataset(**{k: v for k, v in cfg.items() if k != "name"})
        split_idx = int(len(rows) * 0.8)
        train_rows = rows[:split_idx]
        test_rows = rows[split_idx:]

        train_file = train_dir / f"{name}.csv"
        test_file = test_dir / f"{name}.csv"

        _write_csv(train_file, train_rows)
        _write_csv(test_file, test_rows)

        logging.info("Created %s: train=%s rows, test=%s rows", name, len(train_rows), len(test_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data creation script")
    parser.add_argument("--save_dir", help="Path to save generated data.", required=True)
    args = parser.parse_args()

    create_data(args.save_dir)
