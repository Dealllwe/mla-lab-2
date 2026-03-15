import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def _train_scaler(data_dir: Path, standard_scaler_path: str) -> StandardScaler:
    scaler = StandardScaler()

    data = []
    for file in data_dir.iterdir():
        if not file.name.endswith('.csv'):
            logging.info(f'Wrong file format in {data_dir} : {file.name}')

        data.append(
            pd.read_csv(file).values.reshape(-1, 1)
        )
    
    np_data = np.concatenate(data)
    scaler.fit(np_data)

    joblib.dump(scaler, standard_scaler_path)

    return scaler


def _preprocess_data(data_dir: Path, scaler: StandardScaler):

    if not data_dir.exists():
        raise Exception(f'Path {data_dir} does not exist.')
    
    for file in data_dir.iterdir():
        if not file.name.endswith('.csv'):
            logging.info(f'Wrong file format in {data_dir} : {file.name}')
        
        save_dir = data_dir.parent / (data_dir.name + '_preprocessed')
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / file.name

        pd.DataFrame(
            scaler.transform(
                 pd.read_csv(file).values.reshape(-1, 1)
            )
        ).to_csv(
            save_path,
            index=False
        )

        logging.info(f'Data from {file} preprocessed and saved to {save_path}')





def preprocess(data_dir: str, standard_scaler_path: str):
    data_dir = Path(data_dir)

    scaler = _train_scaler(data_dir/'train', standard_scaler_path)

    _preprocess_data(data_dir/'train', scaler)
    _preprocess_data(data_dir/'test', scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument("--data_dir", help="Path to train/test directories.", required=True)
    parser.add_argument("--standard_scaler_path", help="Path to save StandardScaler.", required=True)
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.standard_scaler_path)