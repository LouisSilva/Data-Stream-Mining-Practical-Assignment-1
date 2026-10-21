"""Collect, inspect and save results from the streaming experiments.

ExperimentResults is the only object the notebook needs for experiment outputs:
    results = ExperimentResults(config)
    results.run(streams)
    results.save()
    results.summary      # final metrics / confusion counts / timings
    results.history      # checkpointed cumulative + window metrics
    results.predictions  # selected prediction-level traces
    results.history["window_macro_f1"]  # last however many (determined in the config) evaluated events at checkpoints
"""
from __future__ import annotations
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pandas as pd

from .data import create_random_suffix
from .evaluation import _validate_stream, evaluate_model


class ExperimentResults:
    """Owns the results and output files for one complete experiment batch."""

    def __init__(self, config: dict):
        self.config = config
        self._summaries: list[dict] = []
        self._histories: list[pd.DataFrame] = []
        self._prediction_runs: list[pd.DataFrame] = []

    @property
    def summary(self) -> pd.DataFrame:
        """One final predictive/timing/confusion row per experiment."""
        return pd.DataFrame(self._summaries)

    @property
    def history(self) -> pd.DataFrame:
        """Cumulative and recent-window metric snapshots at checkpoints."""
        if not self._histories:
            return pd.DataFrame()
        return pd.concat(self._histories, ignore_index=True)

    @property
    def predictions(self) -> pd.DataFrame:
        """Scored, timestamped predictions retained for selected runs."""
        columns = ['city', 'representation', 'model', 'timestamp', 'y_true', 'y_pred']
        if not self._prediction_runs:
            return pd.DataFrame(columns=columns)
        return pd.concat(self._prediction_runs, ignore_index=True)

    def run(self, city_streams: dict) -> None:
        """Run all city/representation/model combinations once.

        The earliest observations train each fresh model without scoring.
        Remaining observations are processed in strict chronological order.
        """
        if self._summaries:
            raise RuntimeError('This ExperimentResults instance has already run; create a new one')
            
        # Import River only when training models, not while inspecting results.
        from .models import build_model
        if not city_streams:
            raise ValueError('No city streams provided')

        models_config = self.config['models']
        model_names = tuple(models_config)
        if not model_names:
            raise ValueError('config.models must contain at least one model')

        options = self.config.get('evaluation', {})
        warmup_fraction = float(options.get('warmup_fraction', 0.25))
        window_size = int(options.get('window_size', 720))
        checkpoint_every = int(options.get('checkpoint_every', 24))
        if not 0 < warmup_fraction < 1:
            raise ValueError('evaluation.warmup_fraction must be between 0 and 1')
        if window_size < 1:
            raise ValueError('evaluation.window_size must be >= 1')
        if checkpoint_every < 1:
            raise ValueError('evaluation.checkpoint_every must be >= 1')

        trace_models = options.get('prediction_trace_models', ['online_bagging', 'hoeffding_tree'])
        trace_representation = options.get('prediction_trace_representation', 'A')
        if not isinstance(trace_models, list) or not all(isinstance(name, str) for name in trace_models):
            raise TypeError('evaluation.prediction_trace_models must be a list of names')
        if trace_representation not in ('A', 'B'):
            raise ValueError('evaluation.prediction_trace_representation must be A or B')
            
        unknown_models = set(trace_models) - set(model_names)
        if unknown_models:
            raise ValueError(f'Unknown prediction-trace models: {sorted(unknown_models)}')

        timestamps = None
        for city, stream in city_streams.items():
            city_timestamps = _validate_stream(stream, city)
            if timestamps is not None and not city_timestamps.equals(timestamps):
                raise ValueError('City streams must have the same eligible timestamps')
            timestamps = city_timestamps

        warmup_rows = int(len(timestamps) * warmup_fraction)
        if not 1 <= warmup_rows < len(timestamps):
            raise ValueError('Not enough rows to allocate warm-up and evaluation periods')

        for city, stream in city_streams.items():
            for representation in ('A', 'B'):
                features = stream[f'X_{representation}']
                
                for model_name in model_names:
                    model = build_model(models_config, model_name)
                    trace = representation == trace_representation and model_name in trace_models
                    
                    summary, history, predictions = evaluate_model(
                        model, features, stream['y'], stream['timestamp'],
                        warmup_rows=warmup_rows,
                        window_size=window_size,
                        checkpoint_every=checkpoint_every,
                        record_predictions=trace,
                    )
                    
                    key = {'city': city, 'representation': representation, 'model': model_name}
                    self._summaries.append({**key, **summary})
                    self._histories.append(history.assign(**key)[
                        ['city', 'representation', 'model',
                         *[col for col in history.columns if col not in key]]
                    ])
                    
                    if predictions:
                        self._prediction_runs.append(pd.DataFrame(predictions).assign(**key)[
                            ['city', 'representation', 'model', 'timestamp', 'y_true', 'y_pred']
                        ])
                        
                    print(f'{city} / {representation} / {model_name}: '
                          f'Accuracy={summary["accuracy"]:.3f}, '
                          f'Macro-F1={summary["macro_f1"]:.3f}, '
                          f'Precision={summary["precision"]:.3f}, '
                          f'Recall={summary["recall"]:.3f}')

    @staticmethod
    def _ascii_table(frame: pd.DataFrame, columns: list[str]) -> str:
        """Format selected summary columns as an aligned plain-text table."""
        def cell(value, column: str) -> str:
            if pd.isna(value):
                return 'NA'
            if isinstance(value, float):
                return f'{value:.{6 if column.endswith("_ms_per_event") else 4}f}'
            return str(value)

        rows = [
            [cell(value, name) for name, value in zip(columns, record)]
            for record in frame[columns].itertuples(index=False, name=None)
        ]
        widths = [max(len(col), *(len(row[i]) for row in rows)) for i, col in enumerate(columns)]
        border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

        def line(values: list[str]) -> str:
            return '| ' + ' | '.join(value.ljust(width) for value, width in zip(values, widths)) + ' |'

        return '\n'.join([border, line(columns), border, *(line(row) for row in rows), border])

    def save(self, out_dir: str | Path = 'results') -> dict[str, Path]:
        """Save the text report and configured history/prediction CSVs.

        Files for a given run share one random suffix, as in the original code.
        The saved report contains configuration and software version details.
        """
        summary = self.summary
        if summary.empty:
            raise ValueError('Run the experiments before saving results')
            
        options = self.config.get('evaluation', {})
        write_history = options.get('write_history_to_csv', False)
        if not isinstance(write_history, bool):
            raise TypeError('evaluation.write_history_to_csv must be true or false in YAML')

        try:
            river_version = version('river')
        except PackageNotFoundError:
            river_version = 'not installed'
            
        audit = {
            'cities': summary['city'].unique().tolist(),
            'representations': summary['representation'].unique().tolist(),
            'models': list(self.config['models']),
            'warmup_rows': int(summary['warmup_rows'].iloc[0]),
            'config': self.config,
            'versions': {'river': river_version, 'pandas': pd.__version__},
        }
            
        keys = ['city', 'representation', 'model']
        categories = [
            ('PREDICTIVE METRICS', [
                'n_evaluated', 'accuracy', 'macro_f1',
                'precision', 'recall', 'window_accuracy', 'window_macro_f1',
            ]),
            ('CONFUSION MATRIX AND COMPUTATIONAL COST', [
                'TP', 'TN', 'FP', 'FN', 'prediction_ms_per_event', 'update_ms_per_event',
            ]),
            ('EVALUATION PERIOD', [
                'warmup_rows', 'evaluation_start', 'evaluation_end',
            ]),
        ]
        
        sections = [
            'PA01 - STREAMING CLASSIFIER EVALUATION REPORT', '=' * 47, '',
            'EXPERIMENT CONFIGURATION AND AUDIT', '-' * 34,
            json.dumps(audit, indent=2, default=str, ensure_ascii=True),
            '', 'EXPERIMENT RESULTS', '-' * 18,
        ]
        
        included = set(keys)
        for title, requested in categories:
            columns = keys + [name for name in requested if name in summary.columns]
            if len(columns) == len(keys):
                continue
            sections.extend(['', title, self._ascii_table(summary, columns)])
            included.update(columns)

        remaining = [name for name in summary.columns if name not in included]
        if remaining:
            sections.extend(['', 'ADDITIONAL EXPERIMENT FIELDS',
                             self._ascii_table(summary, keys + remaining)])

        suffix = create_random_suffix()
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f'report_{suffix}.txt'
        report_path.write_text('\n'.join(sections) + '\n', encoding='utf-8')
        paths = {'report': report_path}

        if write_history:
            path = directory / f'metric_history_{suffix}.csv'
            self.history.to_csv(path, index=False)
            paths['history'] = path
            
        if not self.predictions.empty:
            path = directory / f'predictions_{suffix}.csv'
            self.predictions.to_csv(path, index=False)
            paths['predictions'] = path
            print(f'Saved {path}')
            
        return paths
