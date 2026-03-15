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
) -> list[dict]:
    rng = random.Random(seed)
    temperature_points = []

    for day in range(n_days):
        seasonality = 8.0 * math.sin(2.0 * math.pi * day / 30.0)
        trend = trend_per_day * day
        noise = rng.gauss(0.0, noise_std)

        temperature = base_temp + seasonality + trend + noise
        has_anomaly = 1 if rng.random() < anomaly_probability else 0
        if has_anomaly:
            temperature += rng.gauss(0.0, anomaly_scale)

        temperature_points.append((day, seasonality, trend, noise, has_anomaly, temperature))

    rows = []
    for i in range(len(temperature_points) - 1):
        day, seasonality, trend, noise, has_anomaly, temperature = temperature_points[i]
        target_temperature = temperature_points[i + 1][5]
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _clean_old_csv(dir_path: Path) -> None:
    for file in dir_path.glob("*.csv"):
        file.unlink()


def create_data(save_dir: str) -> tuple[Path, Path]:
    save_path = Path(save_dir)
    train_dir = save_path / "train"
    test_dir = save_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    _clean_old_csv(train_dir)
    _clean_old_csv(test_dir)

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

        _write_csv(train_dir / f"{name}.csv", rows[:split_idx])
        _write_csv(test_dir / f"{name}.csv", rows[split_idx:])

        logging.info("Created dataset %s with %s rows", name, len(rows))

    return train_dir, test_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data creation script")
    parser.add_argument("--save_dir", help="Path to save generated data.", required=True)
    args = parser.parse_args()

    create_data(args.save_dir)
