"""Chronological, prequential classification evaluation.

Input for each city is the dictionary returned by prepare_city_stream():
    {"timestamp": pd.Series, "X_A": pd.DataFrame,
     "X_B": pd.DataFrame, "y": pd.Series}

No model is fitted in batches. Warm-up: learn only. 
Evaluation: PREDICT -> EVALUATE -> LEARN, once per observation.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import deque
from time import perf_counter_ns
import pandas as pd


# These CURRENT-time weather variables must not be model predictors.
_FORBIDDEN_FEATURES = {
    "rain_event", "y", "timestamp", "time",
    "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m",
}


class ConfusionOutcome(Enum):
    """The four possible outcomes of a binary prediction."""
    TP = "tp"
    TN = "tn"
    FP = "fp"
    FN = "fn"


@dataclass
class ConfusionMatrix:
    """Confusion-matrix counters and metrics for the positive class y=1."""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, outcome: ConfusionOutcome, amount: int = 1) -> None:
        """Add one outcome, or remove it with ``amount=-1`` on window eviction."""
        if outcome is ConfusionOutcome.TP:
            self.tp += amount
        elif outcome is ConfusionOutcome.TN:
            self.tn += amount
        elif outcome is ConfusionOutcome.FP:
            self.fp += amount
        elif outcome is ConfusionOutcome.FN:
            self.fn += amount
        else:
            raise ValueError(f"Unknown confusion outcome: {outcome!r}")

    @property
    def total(self) -> int:
        """Number of evaluated predictions represented by this matrix."""
        return self.tp + self.tn + self.fp + self.fn

    def scores(self) -> dict:
        """Calculate Accuracy, Macro-F1 and target-class metrics."""
        def divide(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        # Calculate F1 for the positive class and for the negative class.
        positive_f1 = divide(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        negative_f1 = divide(2 * self.tn, 2 * self.tn + self.fp + self.fn)
        return {
            "n_evaluated": self.total,
            "accuracy": divide(self.tp + self.tn, self.total),
            "macro_f1": (positive_f1 + negative_f1) / 2,
            "precision": divide(self.tp, self.tp + self.fp),
            "recall": divide(self.tp, self.tp + self.fn),
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
        }



class StreamingBinaryMetrics:
    """Tracks classification metrics for all predictions and a sliding window.

    ``cumulative`` counts every evaluated prediction and never forgets.
    ``window`` counts only the latest ``window_size`` evaluated predictions.
    ``window_outcomes`` is a queue of their outcomes, used to remove the oldest
    prediction from ``window`` when a new one arrives.
    """
    def __init__(self, window_size: int = 200):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.cumulative = ConfusionMatrix()
        self.window = ConfusionMatrix()
        self.window_outcomes: deque[ConfusionOutcome] = deque(maxlen=window_size)

    @staticmethod
    def _classify(y: int, pred: int) -> ConfusionOutcome:
        """Map an actual/predicted 0-or-1 pair to a named outcome."""
        if y not in (0, 1) or pred not in (0, 1): raise ValueError("Both actual and predicted labels must be 0 or 1")
        if y == 1 and pred == 1: return ConfusionOutcome.TP
        if y == 0 and pred == 0: return ConfusionOutcome.TN
        if y == 0 and pred == 1: return ConfusionOutcome.FP
        return ConfusionOutcome.FN

    def update(self, y: int, pred: int) -> None:
        """Record one prediction, evaluated before learning its true label."""
        outcome = self._classify(y, pred)
        self.cumulative.add(outcome)

        if len(self.window_outcomes) == self.window_outcomes.maxlen:
            # Remove the oldest result from the sliding window only.
            self.window.add(self.window_outcomes.popleft(), amount=-1)

        self.window_outcomes.append(outcome)
        self.window.add(outcome)
        
    def snapshot(self) -> dict:
        """Return cumulative metrics plus window Accuracy and Macro-F1."""
        full_scores = self.cumulative.scores()
        window_scores = self.window.scores()

        # Set windows that have only dry/rain periods to NaNs, because otherwise
        # they have exactly 0.5 macro-F1, which looks weird on the graph and looks
        # like the model performance is deteriorating, when it isn't.
        has_rain = (self.window.tp + self.window.fn) > 0
        has_dry = (self.window.tn + self.window.fp) > 0
        
        window_macro_f1 = (
            window_scores["macro_f1"]
            if has_rain and has_dry
            else float("nan")
        )
        
        return {
            **full_scores,
            "window_accuracy": window_scores["accuracy"],
            "window_macro_f1": window_macro_f1,
        }


def _validate_stream(stream: dict, city: str) -> pd.Series:
    """Check a city's preprocessed data before models receive any observations.

    Verifies required keys, equal lengths, numeric/non-missing features,
    binary labels, chronological timestamps, and that Representation B
    preserves all Representation A values.  Rejects known current-hour
    weather columns to prevent target leakage.

    Returns a 
    -------
    pandas.Series
        Parsed, reset-index timestamps for cross-city alignment checks.
    """
    required = {'timestamp', 'X_A', 'X_B', 'y'}
    if not required.issubset(stream):
        raise KeyError(f'{city}: missing stream keys {sorted(required - set(stream))}')
        
    a, b, y = stream['X_A'], stream['X_B'], stream['y']
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        raise TypeError(f'{city}: X_A and X_B must be DataFrames')
        
    if not isinstance(y, pd.Series):
        raise TypeError(f'{city}: y must be a Series')
        
    if not (len(a) == len(b) == len(y) == len(stream['timestamp'])) or len(y) < 2:
        raise ValueError(f'{city}: X_A, X_B, y and timestamp must have equal length >= 2')
        
    if a.columns.has_duplicates or b.columns.has_duplicates:
        raise ValueError(f'{city}: duplicate feature names')
        
    if not a.columns.isin(b.columns).all():
        raise ValueError(f'{city}: Representation B must contain all A columns')
        
    # A and B must differ only by extra temporal columns, not by data values.
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b[list(a.columns)].reset_index(drop=True),
        check_dtype=False,
    )
    
    for rep, x in [('A', a), ('B', b)]:
        if x.shape[1] == 0 or not all(pd.api.types.is_numeric_dtype(dt) for dt in x.dtypes):
            raise ValueError(f'{city}: Representation {rep} must be non-empty and numeric')
            
        if set(x.columns) & _FORBIDDEN_FEATURES:
            raise ValueError(f'{city}: Representation {rep} contains current-hour predictors')
            
        if x.isna().any().any():
            raise ValueError(f'{city}: Representation {rep} has missing features')
            
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f'{city}: labels must be only 0 or 1 and never missing')
        
    # The timestamp is evaluation metadata, not a model input feature.
    ts = pd.to_datetime(pd.Series(stream['timestamp']).reset_index(drop=True), errors='raise')
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f'{city}: timestamp must be unique, complete and chronological')
        
    return ts


def evaluate_model(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    timestamps: pd.Series, 
    warmup_rows: int, 
    window_size: int = 200,
    checkpoint_every: int = 1000,
    record_predictions: bool = False
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Evaluate one fresh online model on a single chronological city stream.

    Parameters
    ----------
    model : River classifier or pipeline
        Supports ``predict_one(x)`` and ``learn_one(x, y)``.
        
    X : pandas.DataFrame
        Causal features only; one row per timestamp.
        
    y : pandas.Series
        Actual rain-event labels; revealed to the learner AFTER each prediction.
        
    timestamps : pandas.Series
        Observation times for checkpoints and reporting, never fed to model.
        
    warmup_rows : int
        Number of earliest rows used for learning but excluded from metrics.
        
    window_size : int
        Number of most recent scored events included in window metrics.
        
    checkpoint_every : int
        Write one metric-history record every this many evaluated events.

    record_predictions : bool
        Whether to retain one timestamp, true label and prediction per scored
        event. Warm-up observations are never retained.

    Returns
    -------
    (dict, pandas.DataFrame)
       Final summary, checkpointed metric history and optional predictions.
    """
    if not (1 <= warmup_rows < len(y)):
        raise ValueError('warmup_rows must be >=1 and leave evaluation observations')
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be >= 1')

    metrics = StreamingBinaryMetrics(window_size)
    names = X.columns.tolist()
    pred_ns = update_ns = 0
    history = []
    predictions = []
    last_ts = None

    for i, (values, label, ts) in enumerate(zip(
        X.itertuples(index=False, name=None),
        y.to_numpy(),
        timestamps,
    )):
        x = {name: float(v) for name, v in zip(names, values)}
        label = int(label)
        if i < warmup_rows:
            # Give every model the same initial labelled history, without scoring.
            model.learn_one(x, label)
            continue

        # STEP 1: Predict using only the information available before hour t.
        before = perf_counter_ns()
        pred = model.predict_one(x)
        pred_ns += perf_counter_ns() - before
        
        if pred is None or pred not in (0, 1, False, True):
            raise RuntimeError(f'Non-binary prediction at {ts}: {pred!r}')
            
        # STEP 2: Evaluate the prediction before exposing this label to model.
        metrics.update(label, int(pred))
        if record_predictions:
            predictions.append({"timestamp": ts, "y_true": label, "y_pred": int(pred)})

        # STEP 3: Incrementally update the model with the true label.
        # Timers cover only model calls, not metrics, data conversion, or I/O.
        before = perf_counter_ns()
        model.learn_one(x, label)
        update_ns += perf_counter_ns() - before

        last_ts = ts
        count = metrics.cumulative.total
        if count % checkpoint_every == 0:
            # Save intermittent snapshots rather than one row for every event.
            history.append({'timestamp': ts, **metrics.snapshot()})

    # Average model-call latency across scored events (warm-up excluded).
    result = metrics.snapshot()
    n = result['n_evaluated']
    result.update({
        'warmup_rows': warmup_rows,
        'evaluation_start': str(timestamps.iloc[warmup_rows]),
        'evaluation_end': str(last_ts),
        'prediction_ms_per_event': pred_ns / n / 1e6,
        'update_ms_per_event': update_ns / n / 1e6,
    })
    
    if not history or history[-1]['n_evaluated'] != n:
        # Guarantee that the final evaluation timestamp appears in history.
        history.append({'timestamp': last_ts, **metrics.snapshot()})
        
    return result, pd.DataFrame(history), predictions
