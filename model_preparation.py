import argparse
import csv
import json
import logging
from pathlib import Path

LOG_PATH = Path("logs/model_preparation.log")
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


def _train_linear_regression(X, y, lr=0.01, epochs=2000):
    n_samples = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        dw = [0.0] * n_features
        db = 0.0

        for i in range(n_samples):
            pred = sum(weights[j] * X[i][j] for j in range(n_features)) + bias
            err = pred - y[i]
            for j in range(n_features):
                dw[j] += err * X[i][j]
            db += err

        for j in range(n_features):
            weights[j] -= lr * (2.0 / n_samples) * dw[j]
        bias -= lr * (2.0 / n_samples) * db

    return weights, bias


def train(train_data_path: str, model_save_path: str):
    train_path = Path(train_data_path)
    model_path = Path(model_save_path)

    rows = _read_rows_from_path(train_path)
    feature_names = [k for k in rows[0].keys() if k != TARGET_COLUMN]

    X = [[float(row[f]) for f in feature_names] for row in rows]
    y = [float(row[TARGET_COLUMN]) for row in rows]

    weights, bias = _train_linear_regression(X, y)

    model = {"feature_names": feature_names, "weights": weights, "bias": bias}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    logging.info("Model trained on %s rows and saved to %s", len(rows), model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument(
        "--train_data_path",
        help="Path to preprocessed train directory or csv file.",
        required=True,
    )
    parser.add_argument("--model_save_path", help="Path to save model (.json/.joblib).", required=True)
    args = parser.parse_args()

    train(args.train_data_path, args.model_save_path)
