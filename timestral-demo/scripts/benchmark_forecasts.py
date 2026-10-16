"""Rolling-origin comparison. Run: uv run python scripts/benchmark_forecasts.py.

No future covariates enter the operational comparison. --include-oracle adds a
separate retrospective experiment with observed future covariates.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from timestral import TimestralForecaster

if __package__:
    from .forecast_metrics import summarize
else:
    from forecast_metrics import summarize

CORE_COLUMNS = [
    'shortwave_radiation_backwards_sfc', 'total_cloud_cover_sfc',
    'temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
    'zenith', 'azimuth',
]
EXPANDED_COLUMNS = CORE_COLUMNS + [
    'angle_of_incidence', 'high_cloud_cover_high_cld_lay',
    'medium_cloud_cover_mid_cld_lay', 'low_cloud_cover_low_cld_lay',
]


def lag_features(y, x, cutoffs):
    """Every feature for cutoff c comes from indices strictly below c."""
    power = y[cutoffs[:, None] - np.arange(1, 25)]
    weather = x[cutoffs[:, None] - np.array([1, 2, 3, 9])].reshape(len(cutoffs), -1)
    return np.column_stack([power, weather, power[:, :9].mean(1), power.mean(1)])


def tree_forecast(y, x, origin, horizon=30):
    # Training windows and all of their labels end strictly before the cutoff.
    cutoffs = np.arange(24, origin - horizon + 1, 3)
    steps = np.arange(horizon)
    labels = y[cutoffs[:, None] + steps]
    assert int(cutoffs[-1] + horizon - 1) < origin
    features = lag_features(y, x, cutoffs)
    train_x = np.column_stack([
        np.repeat(features, horizon, axis=0),
        np.tile(steps + 1, len(cutoffs)),
    ])
    query = np.column_stack([
        np.repeat(lag_features(y, x, np.array([origin])), horizon, axis=0), steps + 1,
    ])
    # One direct model with lead time as a feature; no recursive use of actuals.
    estimator = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=100, max_leaf_nodes=15,
        learning_rate=0.08, early_stopping=False, random_state=42,
    )
    estimator.fit(train_x, labels.ravel())
    return np.maximum(estimator.predict(query), 0)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-oracle', action='store_true')
    parser.add_argument('--origins', type=int, default=12)
    args = parser.parse_args()
    if args.origins < 2:
        parser.error('--origins must be at least 2')
    torch.set_num_threads(4)
    data = pd.read_csv('data/spg.csv')
    y = data.generated_power_kw.to_numpy(dtype=np.float32)
    x = data[EXPANDED_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError('Benchmark requires finite target and covariate values.')
    origins = np.linspace(len(y) // 2, len(y) - 30, args.origins, dtype=int)
    if len(np.unique(origins)) != len(origins) or np.min(np.diff(origins)) < 30:
        raise ValueError('Use fewer origins so 30-row evaluation windows do not overlap.')
    model = TimestralForecaster.from_pretrained('mistralai/Timestral-209M', device='cpu')
    records = []
    for origin in origins:
        forecasts = {}
        for label, columns in [('Timestral target only', []), ('Timestral core past', CORE_COLUMNS), ('Timestral expanded past', EXPANDED_COLUMNS)]:
            kwargs = {}
            if columns:
                kwargs['past_covariates'] = data.iloc[:origin][columns].to_numpy(dtype=np.float32).T[None]
            f = model.forecast(y[None, None, :origin], prediction_length=30, **kwargs)
            forecasts[('historical only', label)] = [v[0, 0].cpu().numpy() for v in (f.median, f.quantile(.1), f.quantile(.9))]
        with threadpool_limits(limits=4):
            tree = tree_forecast(y, x, origin)
        empty = np.full(30, np.nan)
        forecasts[('historical only', 'Gradient Boosting expanded past')] = (tree, empty, empty)
        forecasts[('historical only', 'Last value')] = (np.repeat(y[origin-1], 30), empty, empty)
        forecasts[('historical only', 'Repeat last 9 rows')] = (np.resize(y[origin-9:origin], 30), empty, empty)
        if args.include_oracle:
            f = model.forecast(y[None, None, :origin], prediction_length=30,
                               known_covariates=x[:origin+30].T[None])
            forecasts[('idealized future inputs', 'Timestral expanded known')] = [v[0, 0].cpu().numpy() for v in (f.median, f.quantile(.1), f.quantile(.9))]
        for (scenario, label), (median, lower, upper) in forecasts.items():
            assert median.shape == (30,) and np.isfinite(median).all()
            for k in range(30):
                records.append(dict(origin=int(origin), lead=k+1, observation=int(origin+k),
                                    scenario=scenario, model=label, actual=float(y[origin+k]),
                                    predicted=float(median[k]), p10=float(lower[k]), p90=float(upper[k])))
        print(f'Finished origin {origin} ({len(records)} predictions)', flush=True)
    output = Path('reports/rolling'); output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(records)
    scores = summarize(predictions)
    predictions.to_csv(output / 'predictions.csv', index=False)
    scores.to_csv(output / 'metrics.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps(dict(
        data_sha256=hashlib.sha256(Path("data/spg.csv").read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ["timestral", "torch", "scikit-learn", "numpy", "pandas"]},
        origins=origins.tolist(), horizons=[1, 9, 30], core_columns=CORE_COLUMNS,
        expanded_columns=EXPANDED_COLUMNS, oracle=args.include_oracle,
        tree=dict(lags=24, weather_lags=[1,2,3,9], training_stride=3,
                  max_iter=100, max_leaf_nodes=15, loss='absolute_error'),
        note='Exploratory evaluation. Assumes row chronology and historical feature availability. No timestamps or weather issue times supplied.',
    ), indent=2)+'\n')
    print(scores.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
