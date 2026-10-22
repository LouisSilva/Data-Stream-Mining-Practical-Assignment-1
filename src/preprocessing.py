import re
import pandas as pd

FEATURES_A = [
    "temperature_lag1",
    "humidity_lag1",
    "precipitation_lag1",
    "wind_lag1",
]

RAW_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

_WEATHER_FEATURE = re.compile(
    r"^(temperature_2m|relative_humidity_2m|precipitation|wind_speed_10m)_"
    r"(delta|mean[1-9][0-9]*|std[1-9][0-9]*)$"
)
_RAIN_FREQUENCY = re.compile(r"^rainy_hours([1-9][0-9]*)$")


def _add_temporal_feature(df: pd.DataFrame, feature: str) -> None:
    """Add one specified, past-only temporal feature to df."""
    # The C^(N)_t rolling window
    rain_match = _RAIN_FREQUENCY.fullmatch(feature)
    if rain_match:
        window = int(rain_match.group(1))
        df[feature] = df["rain_event"].shift(1).rolling(window).sum()
        return

    match = _WEATHER_FEATURE.fullmatch(feature)
    if not match:
        raise ValueError(
            f"Unsupported feature {feature!r}. Use <raw_column>_delta, "
            "<raw_column>_meanN, <raw_column>_stdN, or rainy_hoursN, "
            f"where raw_column is one of {RAW_FEATURES}."
        )

    raw_column, operation = match.groups()
    past = df[raw_column].shift(1)

    if operation == "delta":
        df[feature] = past - df[raw_column].shift(2)
        
    elif operation.startswith("mean"):
        window = int(operation[len("mean"):])
        df[feature] = past.rolling(window).mean()
        
    elif operation.startswith("std"):
        window = int(operation[len("std"):])
        if window < 2:
            raise ValueError(f"{feature}: sample standard deviation needs at least 2 observations")
        df[feature] = past.rolling(window).std(ddof=1)


def add_rain_event(df: pd.DataFrame) -> None:
    df["rain_event"] = (df["precipitation"] >= 0.1).astype(int)


def prepare_city_stream(preprocessing_config: dict, df: pd.DataFrame) -> dict:
    """Returns (timestamp, X_A, X_B, y)."""
    features = preprocessing_config.get("features")
    if not isinstance(features, list) or any(not isinstance(f, str) for f in features):
        raise ValueError("preprocessing.features must be a list of feature-name strings")
        
    if len(features) != len(set(features)):
        raise ValueError("preprocessing.features contains duplicate feature names")
        
    if any(f in FEATURES_A for f in features):
        raise ValueError("Representation A features are already included automatically")
    
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Ensure that shift(1) really means one hour earlier.
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if (df["time"].diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("The stream has missing or irregular hours.")

    if df[RAW_FEATURES].isna().any().any():
        raise ValueError("Missing weather values must be handled.")

    # Target
    if "rain_event" not in df.columns:
        add_rain_event(df)

    # Representation A
    for feature_name, raw_column in zip(FEATURES_A, RAW_FEATURES):
        df[feature_name] = df[raw_column].shift(1)

    # Configuration-controlled Representation B
    for feature in features:
        _add_temporal_feature(df, feature)

    # Use identical eligible timestamps for A and B
    features_b = FEATURES_A + features
    df = df.dropna(subset=features_b).reset_index(drop=True)

    return {
        "timestamp": df["time"].copy(),
        "X_A": df[FEATURES_A].copy(),
        "X_B": df[features_b].copy(),
        "y": df["rain_event"].copy(),
    }