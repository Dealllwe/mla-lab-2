import argparse
import logging

logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def test(test_data_path: str, model_save_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing")
    parser.add_argument("--test_data_path", help="Path to preprocessed test data endswith (.csv).", required=True)
    parser.add_argument("--model_save_path", help="Path to saved model weights; endswith (.joblib)", required=True)
    args = parser.parse_args()
    
    test(args.test_data_path, args.model_save_path)