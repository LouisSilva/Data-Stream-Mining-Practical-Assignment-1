"""Chronological, prequential classification evaluation.

Input for each city is the dictionary returned by prepare_city_stream():
    {"timestamp": pd.Series, "X_A": pd.DataFrame,
     "X_B": pd.DataFrame, "y": pd.Series}

No model is fitted in batches. Warm-up: learn only. Evaluation:
PREDICT -> EVALUATE -> LEARN, once per observation.
"""
from __future__ import annotations

from collections import deque
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
import json

import numpy as np
import pandas as pd

from .models import MODEL_NAMES, build_model
from .data import create_random_suffix

# These CURRENT-time weather variables must not be model predictors.
_FORBIDDEN_X = {
    "rain_event", "y", "timestamp", "time",
    "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m",
}


def _div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0



# TODO: add dataclass for storing the [TP, TN, FP, FN] list, and use that instead of the `recent_counts`


class BinaryMetrics:
    """Incremental confusion matrix for y=1 (rain) and y=0 (no rain)."""

    def __init__(self, recent_window: int = 200):
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
            
        self.tp = self.tn = self.fp = self.fn = 0
        self.recent = deque(maxlen=recent_window)
        self.recent_counts = [0, 0, 0, 0]  # TP, TN, FP, FN

    @staticmethod
    def _cell(y: int, y_pred: int) -> int:
        if y == 1 and y_pred == 1:
            return 0
        if y == 0 and y_pred == 0:
            return 1
        if y == 0 and y_pred == 1:
            return 2
        return 3

    def update(self, y: int, y_pred: int) -> None:
        cell = self._cell(y, y_pred)
        if cell == 0:
            self.tp += 1
        elif cell == 1:
            self.tn += 1
        elif cell == 2:
            self.fp += 1
        else:
            self.fn += 1

        if len(self.recent) == self.recent.maxlen:
            old_cell = self.recent.popleft()
            self.recent_counts[old_cell] -= 1
            
        self.recent.append(cell)
        self.recent_counts[cell] += 1

    @staticmethod
    def _scores(tp: int, tn: int, fp: int, fn: int) -> dict:
        n = tp + tn + fp + fn
        f1_rain = _div(2 * tp, 2 * tp + fp + fn)
        f1_dry = _div(2 * tn, 2 * tn + fp + fn)
        
        return {
            "n_evaluated": n,
            "accuracy": _div(tp + tn, n),
            "macro_f1": (f1_rain + f1_dry) / 2,
            "precision_event": _div(tp, tp + fp),
            "recall_event": _div(tp, tp + fn),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        }

    def cumulative(self) -> dict:
        return self._scores(self.tp, self.tn, self.fp, self.fn)

    def window(self) -> dict:
        tp, tn, fp, fn = self.recent_counts
        s = self._scores(tp, tn, fp, fn)
        return {
            "window_accuracy": s["accuracy"],
            "window_macro_f1": s["macro_f1"],
        }


def _validate_inputs(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series) -> pd.Series:
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a DataFrame and y a Series")
        
    if len(X) != len(y) or len(X) != len(timestamps) or len(X) < 2:
        raise ValueError("X, y, and timestamps must be aligned and have length >= 2")
        
    if X.columns.has_duplicates or X.shape[1] == 0:
        raise ValueError("X must have unique feature names and at least one feature")
        
    if _FORBIDDEN_X.intersection(X.columns):
        raise ValueError(f"Unsafe current-time/target columns in X: "
                         f"{sorted(_FORBIDDEN_X.intersection(X.columns))}")
        
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All X features must be numeric")
        
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or non-finite values; clean/align it first")
        
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("y must consist solely of binary 0/1 rain-event labels")

    # Timestamps are metadata used for ordering and plots; not predictors.
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="raise")
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be unique, non-missing, chronological")
        
    return ts


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    *,
    warmup_rows: int,
    recent_window: int = 200,
    checkpoint_every: int = 1000,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate a new model on exactly one city and one representation.

    First warmup_rows are learned without being evaluated. Every later label
    is scored before the model is allowed to learn from that observation.
    Return (final_metrics_dict, cumulative/window_history_dataframe).
    """
    ts = _validate_inputs(X, y, timestamps)
    if not 1 <= warmup_rows < len(X):
        raise ValueError("warmup_rows must be between 1 and len(X)-1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")

    evaluator = BinaryMetrics(recent_window=recent_window)
    feature_names = list(X.columns)
    prediction_ns = update_ns = 0
    history = []
    last_timestamp = None

    # itertuples avoids converting an entire historical table into a list of
    # 50,000 dictionaries. Observations are replayed one at a time.
    for i, (values, label, timestamp) in enumerate(
        zip(X.itertuples(index=False, name=None), y.to_numpy(), ts)
    ):
        x = {name: float(value) for name, value in zip(feature_names, values)}
        label = int(label)

        if i < warmup_rows:
            # Give every model the same initial history
            model.learn_one(x, label)
            continue

        # Predict using only the information available before hour 1
        before = perf_counter_ns()
        pred = model.predict_one(x)
        prediction_ns += perf_counter_ns() - before
        
        if pred is None or pred not in (0, 1, False, True):
            raise RuntimeError(f"Model did not return a binary prediction at {timestamp}: {pred!r}")
        pred = int(pred)

        # Evaluate the prediction before exposing this label to the model
        evaluator.update(label, pred)

        # Update the model with the true label
        before = perf_counter_ns()
        model.learn_one(x, label)
        update_ns += perf_counter_ns() - before

        last_timestamp = timestamp
        n_scored = evaluator.tp + evaluator.tn + evaluator.fp + evaluator.fn
        if n_scored % checkpoint_every == 0:
            history.append({"timestamp": timestamp, **evaluator.cumulative(),
                            **evaluator.window()})

    result = evaluator.cumulative()
    n_scored = result["n_evaluated"]
    result.update({
        "warmup_rows": warmup_rows,
        "evaluation_start": str(ts.iloc[warmup_rows]),
        "evaluation_end": str(last_timestamp),
        "prediction_ms_per_event": prediction_ns / n_scored / 1_000_000,
        "update_ms_per_event": update_ns / n_scored / 1_000_000,
    })

    if not history or history[-1]["n_evaluated"] != n_scored:
        history.append({"timestamp": last_timestamp, **evaluator.cumulative(),
                        **evaluator.window()})
        
    return result, pd.DataFrame(history)


def run_experiments(
    model_config: dict,
    city_streams: dict,
    *,
    warmup_fraction: float = 0.25,
    recent_window: int = 200,
    checkpoint_every: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run five models on each city's A and B representation (20 runs for 2 cities).

    city_streams = {"Lisboa": prepare_city_stream(df_lisboa),
                    "Porto": prepare_city_stream(df_porto)}
    Each city must expose identical, chronologically ordered eligible timestamps.
    Models are constructed anew for every experimental run.
    """
    if not 0 < warmup_fraction < 1:
        raise ValueError("warmup_fraction must be between 0 and 1")
    if not city_streams:
        raise ValueError("Pass at least one city stream")

    common_ts = None
    common_warmup_rows = None
    for city, stream in city_streams.items():
        if not all(k in stream for k in ("timestamp", "X_A", "X_B", "y")):
            raise ValueError(f"{city}: expected timestamp, X_A, X_B, and y")
            
        ts = _validate_inputs(stream["X_A"], stream["y"], stream["timestamp"])
        _validate_inputs(stream["X_B"], stream["y"], stream["timestamp"])
        
        if not set(stream["X_B"].columns).issubset(stream["X_B"].columns):
            raise ValueError(f"{city}: Representation B must contain A")
            
        pd.testing.assert_frame_equal(
            stream["X_B"].reset_index(drop=True),
            stream["X_B"][list(stream["X_B"].columns)].reset_index(drop=True),
            check_dtype=False,
        )
        
        if common_ts is not None and not ts.equals(common_ts):
            raise ValueError("Cities must have identical eligible evaluation timestamps")
            
        common_ts = ts
        n_warmup = int(len(ts) * warmup_fraction)
        if not 1 <= n_warmup < len(ts):
            raise ValueError("Not enough observations for warm-up and evaluation")
            
        if common_warmup_rows is not None and common_warmup_rows != n_warmup:
            raise ValueError("Warm-up differs between cities")
            
        common_warmup_rows = n_warmup

    summaries, histories = [], []
    for city, stream in city_streams.items():
        for representation in ("X_A", "X_B"):
            X = stream[representation]

            for model_name in MODEL_NAMES:
                model = build_model(model_config, selected_model=model_name)

                summary, history = evaluate_model(
                    model, X, stream["y"], stream["timestamp"],
                    warmup_rows=common_warmup_rows,
                    recent_window=recent_window,
                    checkpoint_every=checkpoint_every,
                )
                
                summaries.append({"city": city, "representation": representation,
                             "model": model_name, **summary})
                history.insert(0, "city", city)
                history.insert(1, "representation", representation)
                history.insert(2, "model", model_name)
                histories.append(history)
                
                print(f"{city:7} {representation} {model_name:21} "
                      f"Acc={summary['accuracy']:.3f} "
                      f"MacroF1={summary['macro_f1']:.3f} "
                      f"RainRecall={summary['recall_event']:.3f}")

    return pd.DataFrame(summaries), pd.concat(histories, ignore_index=True)


def save_results(
    model_config: dict, 
    summary: pd.DataFrame, 
    history: pd.DataFrame, 
    out_dir: str | Path,
    *, 
    warmup_fraction: float = 0.25, 
    recent_window: int = 200,
    checkpoint_every: int = 1000
) -> None:
    """Save tables and explicit hyperparameters for the report."""
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random_suffix = create_random_suffix()
    summary.to_csv(output_path / f"metrics_{random_suffix}.csv", index=False)
    history.to_csv(output_path / f"metric_history_{random_suffix}.csv", index=False)

    configuration = {
        "warmup_fraction": warmup_fraction,
        "recent_window": recent_window,
        "checkpoint_every": checkpoint_every,
        "models": list(MODEL_NAMES),
        "model_configuration": model_config,
        "river_version": version("river"),
        "pandas_version": pd.__version__,
    }
    (output_path / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
