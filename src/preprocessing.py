import pandas as pd

FEATURES_A = [
    "temperature_lag1",
    "humidity_lag1",
    "precipitation_lag1",
    "wind_lag1",
]

FEATURES_B = FEATURES_A + [
    "humidity_delta",
    "precipitation_delta",
    "humidity_mean6",
    "precipitation_mean6",
    "humidity_std6",
    "precipitation_std6",
    "rainy_hours24",
]


def prepare_city_stream(df: pd.DataFrame) -> dict:
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Ensure that shift(1) really means one hour earlier.
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if (df["time"].diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("The stream has missing or irregular hours.")

    raw_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ]

    if df[raw_features].isna().any().any():
        raise ValueError("Missing weather values must be handled first.")

    # Target
    df["rain_event"] = (df["precipitation"] >= 0.1).astype(int)

    # Representation A
    df["temperature_lag1"] = df["temperature_2m"].shift(1)
    df["humidity_lag1"] = df["relative_humidity_2m"].shift(1)
    df["precipitation_lag1"] = df["precipitation"].shift(1)
    df["wind_lag1"] = df["wind_speed_10m"].shift(1)

    # Representation B
    for name, column in [
        ("humidity", "relative_humidity_2m"),
        ("precipitation", "precipitation"),
    ]:
        series = df[column]
        past = series.shift(1)

        df[f"{name}_delta"] = past - series.shift(2)
        df[f"{name}_mean6"] = past.rolling(6).mean()
        df[f"{name}_std6"] = past.rolling(6).std(ddof=1)

    df["rainy_hours24"] = df["rain_event"].shift(1).rolling(24).sum()

    # Use identical eligible timestamps for A and B
    df = df.dropna(subset=FEATURES_B).reset_index(drop=True)

    return {
        "timestamp": df["time"],
        "X_A": df[FEATURES_A],
        "X_B": df[FEATURES_B],
        "y": df["rain_event"],
    }