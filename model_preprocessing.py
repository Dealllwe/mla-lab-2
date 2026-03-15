import argparse
import csv
import logging
import math
import pickle
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


def _csv_files(data_dir: Path) -> list[Path]:
    files = sorted([f for f in data_dir.iterdir() if f.suffix == ".csv"])
    if not files:
        raise FileNotFoundError(f"No .csv files found in {data_dir}")
    return files


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fit_scaler(train_dir: Path) -> dict:
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
    return scaler


def _save_scaler(scaler: dict, scaler_save_path: Path) -> None:
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_save_path.open("wb") as f:
        pickle.dump(scaler, f)


def _preprocess_split(split_dir: Path, scaler: dict) -> Path:
    output_dir = split_dir.parent / f"{split_dir.name}_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("*.csv"):
        old_file.unlink()

    for file in _csv_files(split_dir):
        rows = _read_rows(file)
        out_path = output_dir / file.name

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS + [TARGET_COLUMN])
            writer.writeheader()

            for row in rows:
                out_row = {
                    col: (float(row[col]) - scaler["mean"][col]) / scaler["std"][col]
                    for col in FEATURE_COLUMNS
                }
                out_row[TARGET_COLUMN] = float(row[TARGET_COLUMN])
                writer.writerow(out_row)

        logging.info("Preprocessed %s -> %s", file, out_path)

    return output_dir


def preprocess(data_dir: str, standard_scaler_path: str) -> tuple[Path, Path]:
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"

    scaler = _fit_scaler(train_dir)
    _save_scaler(scaler, Path(standard_scaler_path))
    logging.info("Scaler fitted and saved to %s", standard_scaler_path)

    train_out = _preprocess_split(train_dir, scaler)
    test_out = _preprocess_split(test_dir, scaler)
    return train_out, test_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument("--data_dir", help="Path to train/test directories.", required=True)
    parser.add_argument("--standard_scaler_path", help="Path to save scaler parameters (.joblib).", required=True)
    args = parser.parse_args()

    preprocess(args.data_dir, args.standard_scaler_path)
