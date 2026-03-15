import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    filename='logs/data_creation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def _create_dummy_data():
    return pd.DataFrame([1,2,3]), pd.DataFrame([4,5])

def create_data(save_dir: str) -> pd.DataFrame:
    save_path = Path(save_dir)

    train_dir = save_path/'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = save_path/'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    

    df_train, df_test = _create_dummy_data()
    logging.info(f'Data created. {df_train.head(3)}')

    df_train.to_csv(train_dir / 'data.csv', index=False)
    df_test.to_csv(test_dir / 'data.csv', index=False)

    logging.info(f'Data saved to dirs {train_dir} and {test_dir}')

    return df_train, df_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data creation script")
    parser.add_argument("--save_dir", help="Path to saved data.", required=True)
    args = parser.parse_args()
    
    create_data(args.save_dir)