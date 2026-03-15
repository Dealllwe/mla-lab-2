import argparse
import logging
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


logging.basicConfig(
    filename='logs/model_prepatation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

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





def train(train_data_path: str , model_save_path: str) -> LinearRegression:
    model_save_path = Path(model_save_path)

    df = pd.read_csv(train_data_path)
    df = pd.DataFrame({
        'y': [11,-1,4,10,0],
        'x1': [1,25,5,2,6],
        'x2': [11,5,-5,2,-61],
    })
    X, y = df.drop(columns='y'), df['y']

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_save_path)

    logging.info(f'Model trained and saved to {model_save_path}')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--train_data_path", help="Path to preprocessed train data endswith (.csv).", required=True)
    parser.add_argument("--model_save_path", help="Path to save model weights endswith (.joblib).", required=True)
    args = parser.parse_args()
    
    train(args.train_data_path, args.model_save_path)