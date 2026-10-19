import string
import secrets
import requests
import json
import pandas as pd
from pathlib import Path

from .json_utils import fetch_json
from .config import find_project_root


def create_random_suffix(length=8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def download_data(data_config: dict, city: str) -> None:
    """
    Uses the parameters from the config to query the API for a specific city, and if successful, writes the results to a json file.
    """
    api_params = data_config["api_parameters"]
    open_meteo_params = {
        "latitude": api_params[city]["latitude"],
        "longitude": api_params[city]["longitude"],
        "start_date": api_params["start_date"],
        "end_date": api_params["end_date"],
        "hourly": ",".join(api_params["hourly"]),
        "timezone": api_params["timezone"]
    }

    # Send request to the Open-Meteo API
    try:
        open_meteo_json, response = fetch_json(api_params["url"], params=open_meteo_params)
        print(f"HTTP status: {response.status_code}")
        print(f"Request URL: {response.url}")
    except requests.RequestException as exc:
        print(f"Could not reach Open-Meteo. Reason: {exc}")
        return None

    raw_dir = Path(find_project_root() / data_config["raw_directory"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / f"{city}_openmeteo_raw.json"
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(open_meteo_json, file, indent=2)


def load_data_from_json(data_config: dict, city: str) -> pd.DataFrame:
    """
    Loads the raw data for a city from a local json file.
    """
    raw_dir = Path(find_project_root() / data_config["raw_directory"])
    file_path = raw_dir / f"{city}_openmeteo_raw.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df


def check_hourly_continuity(df: pd.DataFrame) -> pd.Series:
    timestamps = (
        pd.to_datetime(df["time"])
        .sort_values()
        .reset_index(drop=True)
    )

    differences = timestamps.diff()

    irregular = differences[
        differences.notna() &
        (differences != pd.Timedelta(hours=1))
    ]

    print(f"Number of observations: {len(timestamps)}")
    print(f"Duplicate timestamps: {timestamps.duplicated().sum()}")
    print(f"Non-hourly transitions: {len(irregular)}")

    if irregular.empty:
        print("The stream is continuous at hourly intervals.")
    else:
        print("\nIrregular transitions:")
        for i in irregular.index:
            print(
                f"{timestamps.iloc[i-1]} -> "
                f"{timestamps.iloc[i]} "
                f"({differences.iloc[i]})"
            )

    return irregular