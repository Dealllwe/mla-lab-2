import argparse
import csv
import json
import logging
import math
from pathlib import Path

LOG_PATH = Path("logs/model_testing.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

TARGET_COLUMN = "target_temperature"


def _read_rows_from_path(path: Path):
    rows = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    elif path.is_dir():
        for file in sorted([f for f in path.iterdir() if f.suffix == ".csv"]):
            with file.open("r", newline="", encoding="utf-8") as f:
                rows.extend(list(csv.DictReader(f)))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _predict(model, row):
    return sum(float(row[f]) * w for f, w in zip(model["feature_names"], model["weights"])) + model["bias"]


def test(test_data_path: str, model_save_path: str):
    rows = _read_rows_from_path(Path(test_data_path))

    with Path(model_save_path).open("r", encoding="utf-8") as f:
        model = json.load(f)

    y_true = [float(r[TARGET_COLUMN]) for r in rows]
    y_pred = [_predict(model, r) for r in rows]

    n = len(y_true)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n)

    mean_y = sum(y_true) / n
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": n}

    logging.info("Model evaluation metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing")
    parser.add_argument(
        "--test_data_path",
        help="Path to preprocessed test directory or csv file.",
        required=True,
    )
    parser.add_argument("--model_save_path", help="Path to saved model weights", required=True)
    args = parser.parse_args()

    test(args.test_data_path, args.model_save_path)
