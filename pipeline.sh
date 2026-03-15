#!/bin/bash
set -euo pipefail

LOG_DIR="./logs"
DATA_DIR="./data"

mkdir -p "$LOG_DIR"

echo "Logs will be saved to $LOG_DIR"
echo "Running data creation..."
python data_creation.py --save_dir "$DATA_DIR"

echo "Running preprocessing..."
python model_preprocessing.py --data_dir "$DATA_DIR" --standard_scaler_path "$DATA_DIR/scaler.joblib"

echo "Running model training..."
python model_preparation.py --train_data_path "$DATA_DIR/train_preprocessed" --model_save_path "$DATA_DIR/model.joblib"

echo "Running model testing..."
python model_testing.py --test_data_path "$DATA_DIR/test_preprocessed" --model_save_path "$DATA_DIR/model.joblib"

echo "Pipeline finished successfully"
