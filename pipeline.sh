#!/bin/bash

DIR="./logs"

if [ -d "$DIR" ]; then
    echo "Directory $DIR exists."
else
    echo "Directory $DIR does not exist. Creating..."
    mkdir $DIR
fi

echo "Logs will be saved to $DIR"

echo "Run data_creation.py"
python data_creation.py --save_dir data 

echo "Run model_preprocessing.py"
python model_preprocessing.py --data_dir data --standard_scaler_path data/scaler.joblib


echo "Run model_preparation.py"
python model_preparation.py --train_data_path data/train/data.csv --model_save_path data/model.joblib


echo "Run model_testing.py"
python model_testing.py --test_data_path data/test/data.csv --model_save_path data/model.joblib