"""Chronological, prequential classification evaluation.

Input for each city is the dictionary returned by prepare_city_stream():
    {"timestamp": pd.Series, "X_A": pd.DataFrame,
     "X_B": pd.DataFrame, "y": pd.Series}

No model is fitted in batches. Warm-up: learn only. Evaluation:
PREDICT -> EVALUATE -> LEARN, once per observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
import json

import numpy as np
import pandas as pd

from .models import build_model
from .data import create_random_suffix

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
        return {
            **full_scores,
            "window_accuracy": window_scores["accuracy"],
            "window_macro_f1": window_scores["macro_f1"],
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
    *,
    warmup_rows: int, 
    window_size: int = 200,
    checkpoint_every: int = 1000
) -> tuple[dict, pd.DataFrame]:
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

    Returns
    -------
    (dict, pandas.DataFrame)
        Final predictive/timing summary and sparse metric-history snapshots.
    """
    if not (1 <= warmup_rows < len(y)):
        raise ValueError('warmup_rows must be >=1 and leave evaluation observations')
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be >= 1')

    metrics = StreamingBinaryMetrics(window_size)
    names = X.columns.tolist()
    pred_ns = update_ns = 0
    history = []
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
            # Unlike the lab's cold-start counter, we require an actual 0/1
            # prediction after warm-up so all models score the same events.
            raise RuntimeError(f'Non-binary prediction at {ts}: {pred!r}')
            
        # STEP 2: Evaluate the prediction before exposing this label to model.
        metrics.update(label, int(pred))

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

    # Average model-call latency across *scored* events (warm-up excluded).
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
        
    return result, pd.DataFrame(history)


def run_experiments(config: dict, city_streams: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every configured River model for both representations in each city.

    Parameters
    ----------
    city_streams : dict
        ``{city_name: prepare_city_stream(...)}`` mappings.  Each stream has
        ``timestamp``, ``X_A``, ``X_B`` and ``y`` fields.
    config : dict
        Complete YAML settings; model definitions in ``config['models']`` and
        warm-up, sliding-window and checkpoint options in ``config['evaluation']``.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        One row per city/representation/model in summary; checkpointed model
        histories in the second DataFrame.  Each run starts with a fresh model.
    """
    models_config = config['models']
    model_names = tuple(models_config)
    
    options = config.get('evaluation', {})
    warmup_fraction = float(options.get('warmup_fraction', 0.25))
    window_size = int(options.get('window_size', 200))
    checkpoint_every = int(options.get('checkpoint_every', 1000))
    
    if not 0 < warmup_fraction < 1: raise ValueError('evaluation.warmup_fraction must be between 0 and 1')
    if not city_streams: raise ValueError('No city streams provided')
    if not model_names: raise ValueError('config.models must contain at least one model')

    # Make every city/representation/model start and finish on identical times.
    expected_ts = None
    for city, stream in city_streams.items():
        ts = _validate_stream(stream, city)
        
        if expected_ts is not None and not ts.equals(expected_ts): 
            raise ValueError('City streams must have the same eligible timestamps')
        expected_ts = ts

    # Split by chronological row position, never randomly or with future data.
    warmup_rows = int(len(expected_ts) * warmup_fraction)
    if not 1 <= warmup_rows < len(expected_ts):
        raise ValueError('Not enough rows to allocate warm-up and evaluation periods')

    summaries, histories = [], []
    for city, stream in city_streams.items():
        for representation in ('A', 'B'):
            X = stream[f'X_{representation}']
            
            for model_name in model_names:
                # Model factory reads hyperparameters from YAML and starts from
                # scratch; no model state leaks across experiments.
                model = build_model(models_config, model_name)
                summary, history = evaluate_model(
                    model, X, stream['y'], stream['timestamp'],
                    warmup_rows=warmup_rows,
                    window_size=window_size,
                    checkpoint_every=checkpoint_every,
                )
                
                summaries.append({
                    'city': city, 'representation': representation,
                    'model': model_name, **summary,
                })
                
                history.insert(0, 'city', city)
                history.insert(1, 'representation', representation)
                history.insert(2, 'model', model_name)
                histories.append(history)
                print(f'{city} / {representation} / {model_name}: '
                      f'Accuracy={summary["accuracy"]:.3f}, '
                      f'Macro-F1={summary["macro_f1"]:.3f}, '
                      f'Recall={summary["recall"]:.3f}')

    return pd.DataFrame(summaries), pd.concat(histories, ignore_index=True)


def _ascii_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render selected DataFrame columns as a bordered, plain-ASCII table."""
    def _format_report_cell(value, column: str) -> str:
        """Format a value for a plain-text table without changing the data itself."""
        if pd.isna(value):
            return "NA"
        if isinstance(value, float):
            # Model-call times can be very small, so display more decimal places.
            places = 6 if column.endswith("_ms_per_event") else 4
            return f"{value:.{places}f}"
        return str(value)
        
    rows = [
        [_format_report_cell(value, name) for name, value in zip(columns, record)]
        for record in frame[columns].itertuples(index=False, name=None)
    ]
    widths = [
        max(len(col), *(len(row[i]) for row in rows))
        for i, col in enumerate(columns)
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    return "\n".join([border, line(columns), border,
                      *(line(row) for row in rows), border])    


def save_results(
    config: dict,
    summary: pd.DataFrame, 
    history: pd.DataFrame, 
    out_dir: str | Path = 'results'
) -> None:
    """Write a self-contained text report and optionally a CSV metric history.

    ``report.txt`` includes the complete experiment audit information plus 
     all columns from the summary DataFrame, displayed in several readable ASCII tables.

    ``metric_history.csv`` is created only when
    ``config['evaluation']['write_history_to_csv']`` is ``True`` (default False).
    """
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise ValueError('summary must be a nonempty pandas DataFrame')

    options = config.get('evaluation', {})
    write_history = options.get('write_history_to_csv', False)
    if not isinstance(write_history, bool):
        raise TypeError('evaluation.write_history_to_csv must be true or false in YAML')
    if write_history and not isinstance(history, pd.DataFrame):
        raise TypeError('history must be a pandas DataFrame when CSV export is enabled')

    try:
        river_version = version('river')
    except PackageNotFoundError:
        river_version = 'not installed'

    audit = {
        'cities': summary['city'].unique().tolist(),
        'representations': summary['representation'].unique().tolist(),
        'models': list(config['models']),
        'warmup_rows': int(summary['warmup_rows'].iloc[0]),
        'config': config,
        'versions': {'river': river_version, 'pandas': pd.__version__},
    }

    # Splitting the 20-run summary avoids an unmanageably wide terminal table.
    # All columns remain included, and identifiers repeat in every table.
    identifiers = ['city', 'representation', 'model']
    categories = [
        ('PREDICTIVE METRICS', [
            'n_evaluated', 'accuracy', 'macro_f1',
            'precision', 'recall',
            'window_accuracy', 'window_macro_f1',
        ]),
        ('CONFUSION MATRIX AND COMPUTATIONAL COST', [
            'TP', 'TN', 'FP', 'FN',
            'prediction_ms_per_event', 'update_ms_per_event',
        ]),
        ('EVALUATION PERIOD', [
            'warmup_rows', 'evaluation_start', 'evaluation_end',
        ]),
    ]

    report_sections = [
        'PA01 - STREAMING CLASSIFIER EVALUATION REPORT',
        '=' * 47,
        '',
        'EXPERIMENT CONFIGURATION AND AUDIT',
        '-' * 34,
        # This JSON block is for readability/reproducibility in report.txt
        json.dumps(audit, indent=2, default=str, ensure_ascii=True),
        '',
        'EXPERIMENT RESULTS',
        '-' * 18,
    ]

    included = set()
    for title, requested in categories:
        columns = identifiers + [name for name in requested if name in summary.columns]
        if len(columns) == len(identifiers):
            continue
        report_sections.extend(['', title, _ascii_table(summary, columns)])
        included.update(columns)

    # Preserve extra columns too if the evaluator is extended in the future.
    remaining = [name for name in summary.columns if name not in included]
    if remaining:
        report_sections.extend([
            '', 'ADDITIONAL EXPERIMENT FIELDS',
            _ascii_table(summary, identifiers + remaining),
        ])

    random_suffix = create_random_suffix()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'report_{random_suffix}.txt').write_text(
        '\n'.join(report_sections) + '\n', encoding='utf-8'
    )

    history_path = output / f'metric_history_{random_suffix}.csv'
    if write_history:
        history.to_csv(history_path, index=False)
