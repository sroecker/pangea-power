"""TimesFM-3 input alignment and decile extraction (no model download on import)."""
import numpy as np


def forecast_at_cutoff(model, target, covariates, cutoff, horizon=30):
    """Use only history before cutoff. Covariates have shape (rows, channels)."""
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 1 or not 1 <= cutoff <= len(target) or horizon < 1:
        raise ValueError('Expected a 1D target, valid cutoff, and positive horizon.')
    context = target[:cutoff]
    if not np.isfinite(context).all():
        raise ValueError('This comparison requires finite historical targets.')
    kwargs = {}
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float32)
        if covariates.ndim != 2 or len(covariates) < cutoff:
            raise ValueError('Covariates must be (rows, channels) covering the context.')
        past = covariates[:cutoff].T
        if not np.isfinite(past).all():
            raise ValueError('This comparison requires finite historical covariates.')
        kwargs['past_only_covariates'] = [past]
    outputs = list(model.predict_batch(
        contexts=[context], horizon=horizon, return_quantiles=True,
        use_symmetric_averaging=True, make_positive=True,
        sort_quantiles=True, use_znorm=False, padding_mode='none', **kwargs,
    ))
    if len(outputs) != 1:
        raise ValueError('Expected exactly one TimesFM forecast.')
    # The official univariate API returns (H, 9), with deciles 0.1 ... 0.9.
    quantiles = np.asarray(outputs[0].quantiles)
    if quantiles.shape != (horizon, 9) or not np.isfinite(quantiles).all():
        raise ValueError(f'Invalid TimesFM quantile shape/values: {quantiles.shape}')
    if (np.diff(quantiles, axis=-1) < 0).any():
        raise ValueError('TimesFM returned crossing quantiles.')
    return quantiles[:, 4], quantiles[:, 0], quantiles[:, 8]
