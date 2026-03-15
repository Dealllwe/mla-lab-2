import argparse
import csv
import json
import logging
import math
from pathlib import Path

LOG_PATH = Path("logs/data_preprocessing.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

FEATURE_COLUMNS = ["day", "seasonality", "trend", "noise", "has_anomaly", "temperature"]
TARGET_COLUMN = "target_temperature"


def _csv_files(data_dir: Path):
    files = sorted([f for f in data_dir.iterdir() if f.suffix == ".csv"])
    if not files:
        raise FileNotFoundError(f"No .csv files found in {data_dir}")
    return files


def _read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _train_scaler(train_dir: Path, scaler_save_path: Path):
    values = {col: [] for col in FEATURE_COLUMNS}

    for file in _csv_files(train_dir):
        for row in _read_rows(file):
            for col in FEATURE_COLUMNS:
                values[col].append(float(row[col]))

    scaler = {"mean": {}, "std": {}}
    for col in FEATURE_COLUMNS:
        mean = sum(values[col]) / len(values[col])
        variance = sum((x - mean) ** 2 for x in values[col]) / len(values[col])
        std = math.sqrt(variance)
        scaler["mean"][col] = mean
        scaler["std"][col] = std if std > 1e-12 else 1.0

    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_save_path.open("w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    logging.info("Scaler fitted and saved to %s", scaler_save_path)
    return scaler


def _preprocess_split(split_dir: Path, scaler):
    output_dir = split_dir.parent / f"{split_dir.name}_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in _csv_files(split_dir):
        out_path = output_dir / file.name
        rows = _read_rows(file)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS + [TARGET_COLUMN])
            writer.writeheader()

            for row in rows:
                out_row = {}
                for col in FEATURE_COLUMNS:
                    raw = float(row[col])
                    out_row[col] = (raw - scaler["mean"][col]) / scaler["std"][col]
                out_row[TARGET_COLUMN] = float(row[TARGET_COLUMN])
                writer.writerow(out_row)

        logging.info("Preprocessed %s -> %s", file, out_path)

    return output_dir


def preprocess(data_dir: str, standard_scaler_path: str):
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"

    scaler = _train_scaler(train_dir, Path(standard_scaler_path))
    _preprocess_split(train_dir, scaler)
    _preprocess_split(test_dir, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument("--data_dir", help="Path to train/test directories.", required=True)
    parser.add_argument("--standard_scaler_path", help="Path to save scaler json.", required=True)
    args = parser.parse_args()

    preprocess(args.data_dir, args.standard_scaler_path)
