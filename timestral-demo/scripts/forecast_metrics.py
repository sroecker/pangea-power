"""Shared pooled metrics for identical rolling forecast horizons."""
import numpy as np
import pandas as pd


def summarize(predictions):
    frames = []
    for horizon in (1, 9, 30):
        p = predictions.loc[predictions.lead <= horizon].copy()
        p['absolute_error'] = (p.actual - p.predicted).abs()
        p['squared_error'] = (p.actual - p.predicted) ** 2
        p['covered'] = np.where(p.p10.notna(), ((p.actual >= p.p10) & (p.actual <= p.p90)).astype(float), np.nan)
        scores = p.groupby(['scenario', 'model']).agg(
            mae_kw=('absolute_error', 'mean'), mse=('squared_error', 'mean'),
            coverage_80=('covered', 'mean'), n_predictions=('actual', 'size'),
        ).reset_index()
        scores['rmse_kw'] = np.sqrt(scores.pop('mse'))
        scores['horizon'] = horizon
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)

