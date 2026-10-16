# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["timesfm[torch]==3.0.2", "numpy>=1.26.4", "pandas>=2.2"]
# ///
"""Run with uv run --script scripts/evaluate_timesfm.py.

Separate uv-managed environment; does not change the locked Timestral project.
Uses the existing rolling protocol and never consumes future covariates.
Downloaded TimesFM-3 weights are for non-commercial, non-production use.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from forecast_metrics import summarize
from timesfm_adapter import forecast_at_cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path('reports/rolling')
    protocol = json.loads((source / 'protocol.json').read_text())
    data_path = Path('data/spg.csv')
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if protocol['data_sha256'] != digest:
        raise ValueError('Dataset changed: rerun the Timestral benchmark first.')
    data = pd.read_csv(data_path)
    target = data.generated_power_kw.to_numpy(dtype=np.float32)
    origins = protocol['origins']
    if not origins or any(c < 1 or c + 30 > len(target) for c in origins):
        raise ValueError('Invalid rolling origins.')
    output = Path('reports/timesfm')
    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path='google/timesfm-3.0-pytorch', device=args.device,
        per_core_batch_size=1, local_files_only=args.local_files_only,
    ))
    variants = [
        ('TimesFM-3 target only', None),
        ('TimesFM-3 core past', data[protocol['core_columns']].to_numpy(dtype=np.float32)),
        ('TimesFM-3 expanded past', data[protocol['expanded_columns']].to_numpy(dtype=np.float32)),
    ]
    records = []
    for origin in origins:
        for label, covariates in variants:
            median, lower, upper = forecast_at_cutoff(model, target, covariates, origin)
            for k in range(30):
                records.append(dict(origin=origin, lead=k+1, observation=origin+k,
                                    scenario='historical only', model=label,
                                    actual=float(target[origin+k]), predicted=float(median[k]),
                                    p10=float(lower[k]), p90=float(upper[k])))
        print(f'Finished TimesFM-3 origin {origin}', flush=True)
    predictions = pd.DataFrame(records)
    metrics = summarize(predictions)
    # Forecast beyond the end using the same six historical covariates as the notebook.
    median, lower, upper = forecast_at_cutoff(model, target, variants[1][1], len(target))
    future = pd.DataFrame(dict(observation=np.arange(len(target), len(target)+30),
                               median=median, p10=lower, p90=upper))
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    future.to_csv(output / 'forecast.csv', index=False)
    protocol.update(oracle=False, model_id='google/timesfm-3.0-pytorch', device=args.device,
                    backend='pytorch', symmetric_averaging=True, make_positive=True,
                    sort_quantiles=True, use_znorm=False, padding_mode='none',
                    quantile_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    packages={p: importlib.metadata.version(p) for p in ['timesfm','torch','numpy','pandas']})
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
