from src.config import load_yaml_config
from src.data import load_data_from_json
from src.cleaning import clean_dataset
from src.preprocessing import prepare_city_stream

CITYA = "Lisboa"
CITYB = "Porto"

def main():
    config = load_yaml_config()

    # Load the datasets
    df_raw_lisboa = load_data_from_json(CITYA, config)
    df_raw_porto = load_data_from_json(CITYB, config)

    # Clean the datasets
    df_cleaned_lisboa = clean_dataset(df_raw_lisboa, config["cleaning"])
    df_cleaned_porto = clean_dataset(df_raw_porto, config["cleaning"])

    lisboa_stream = prepare_city_stream(df_cleaned_lisboa)
    porto_stream = prepare_city_stream(df_cleaned_porto)

    X_A_lisboa = lisboa_stream["X_A"]
    X_B_lisboa = lisboa_stream["X_B"]
    y_lisboa = lisboa_stream["y"]

    X_A_porto = porto_stream["X_A"]
    X_B_porto = porto_stream["X_B"]
    y_porto = porto_stream["y"]