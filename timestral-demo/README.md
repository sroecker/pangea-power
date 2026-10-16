# Timestral forecasting notebook

Forecast `generated_power_kw` from `data/spg.csv` with Timestral-209M, using historical power and six weather/solar-angle covariates. The notebook predicts the next 30 rows, plots the median and 80% interval, and exports `forecast.csv`.

## Run with uv

```sh
uv sync --locked
uv run jupyter lab timestral_forecast.ipynb
```

Use the project's Python 3.12 `.venv` as the notebook kernel (in VS Code select `.venv/bin/python`). Dependencies and resolved versions are in `pyproject.toml` and `uv.lock`. The first model load downloads about 837 MB of weights. Inference uses CUDA when available, otherwise CPU. Use a current uv release; older versions can crash under macOS sandboxing.

The CSV has 4,213 rows and no timestamps. Row order is assumed chronological, and horizons are in observations, not hours. Variable inferred solar-cycle lengths suggest a filtered or incomplete time grid. Original timestamps are required to establish calendar timing. Historical covariates are assumed available at the forecast cutoff; the CSV does not provide weather forecast issue times.

## Reproduce the comparison

```sh
uv run python scripts/benchmark_forecasts.py --include-oracle
uv run python -m unittest discover -s tests -v
```

The benchmark evaluates 12 non-overlapping 30-row windows, with origins equally spaced across the latter half of the dataset. Horizons 1 and 9 use prefixes of each 30-row prediction. Errors are pooled across origins; RMSE is computed from pooled squared errors. No model tuning uses these windows. This is exploratory evaluation; some recent rows were inspected in earlier experiments, so it is not an untouched final test set.

All operational models use only information before each cutoff:

- Timestral with power history alone, six core historical covariates, or ten expanded historical covariates.
- Histogram Gradient Boosting with 24 power lags, weather lags 1/2/3/9, power averages, and forecast lead. A single direct model predicts all leads; every training label precedes the cutoff. Features summarize history rather than providing Timestral's entire context. Fixed settings: absolute-error loss, 100 iterations, 15 leaves, no early stopping.
- Last-value and repeating-nine-row baselines. Nine rows is a heuristic and not a verified daily period.

`--include-oracle` adds an explicitly separate experiment using actual future weather and solar angles. It is conditional prediction with hindsight, not deployable forecasting. It does not affect the operational comparisons.

Results (MAE in kW; lower is better):

| Model | 1 row | 9 rows | 30 rows |
|---|---:|---:|---:|
| Timestral power only | 209.1 | 340.5 | 412.7 |
| Timestral core past covariates | 225.5 | 250.7 | 349.9 |
| Timestral expanded past covariates | 222.9 | 276.3 | 357.5 |
| Gradient Boosting expanded past | 265.9 | 515.0 | 663.2 |
| Last value | 478.4 | 933.2 | 871.4 |
| Repeat last 9 rows | 744.4 | 1041.9 | 1048.5 |

The notebook retains the six core covariates: radiation, total cloud cover, temperature, humidity, zenith, and azimuth. Adding incidence angle and three cloud layers did not improve 9- or 30-row MAE here. Power-only Timestral performed best on one-row MAE. More origins and a fresh evaluation period are needed before generalizing these findings.

Detailed predictions, pooled metrics, dataset hash, package versions, and settings are in `reports/rolling/`. The notebook displays these results and can regenerate them by setting `RUN_BENCHMARK = True`. The earlier three-window experiment remains available in `scripts/evaluate_covariates.py` and the other report CSVs; its scores cover different origins.

Sources: [Timestral model card](https://huggingface.co/mistralai/Timestral-209M), [covariate API](https://github.com/mistralai/timestral#covariates), [dataset](https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset), [example regression notebook](https://github.com/MariaPinto85/Solar-Energy-Power-Generation/blob/main/Solar%20Energy%20Forcasting.ipynb). Random-split scores from regression examples are not comparable to these rolling forecasts.

## TimesFM-3 comparison

```sh
uv run --script scripts/evaluate_timesfm.py
```

The script declares its own dependencies (including `timesfm[torch]==3.0.2`) using inline uv metadata. uv creates an isolated environment; the main project and its lockfile remain usable without TimesFM. First use requires network access for dependencies and model weights. To pin its full dependency resolution after downloading, run `uv lock --script scripts/evaluate_timesfm.py`.

It evaluates target-only, six-core-past, and expanded-past variants using the saved rolling protocol and verifies the dataset hash. It uses the official PyTorch evaluator with symmetric averaging, non-negative clipping, and sorted deciles, matching Timestral's corresponding defaults. Results go to `reports/timesfm/`, including the next 30-row forecast. CPU is the default; `--device cuda` selects a CUDA GPU. The notebook automatically includes available TimesFM results and checks that dataset, cutoffs, horizons, and feature sets match.

**Execution status:** TimesFM integration and input/output adapter tests are implemented. Actual TimesFM inference has not run here: package downloads are blocked and the weights are not cached. No TimesFM scores are claimed.

[TimesFM-3 downloaded weights](https://huggingface.co/google/timesfm-3.0-pytorch) are licensed for non-commercial, non-production use. See [the official API](https://github.com/google-research/timesfm) for details.

The notebook now also plots all actual generated power, a recent-history close-up, and actual-versus-predicted power on three held-out windows. There are no actual observations beyond the end of the CSV.
