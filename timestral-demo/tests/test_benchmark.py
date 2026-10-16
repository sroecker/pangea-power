"""Check cutoff isolation and metric aggregation, without downloading a model."""
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.benchmark_forecasts import lag_features, tree_forecast, summarize


class BenchmarkTests(unittest.TestCase):
    def test_features_end_before_cutoff(self):
        y = np.arange(100, dtype=float)
        x = np.column_stack([y * 2, y * 3])
        features = lag_features(y, x, np.array([40]))[0]
        np.testing.assert_array_equal(features[:24], np.arange(39, 15, -1))
        np.testing.assert_array_equal(features[24:32], x[[39, 38, 37, 31]].ravel())

    def test_future_values_cannot_change_tree_forecast(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(0, 10, 200)
        x = rng.normal(size=(200, 2))
        altered_y, altered_x = y.copy(), x.copy()
        altered_y[150:] = 1e6
        altered_x[150:] = -1e6
        with threadpool_limits(limits=2):
            expected = tree_forecast(y, x, 150, horizon=9)
            actual = tree_forecast(altered_y, altered_x, 150, horizon=9)
        np.testing.assert_array_equal(expected, actual)

    def test_metrics_use_pooled_squared_errors(self):
        data = pd.DataFrame(dict(scenario=['test']*2, model=['m']*2,
                                 lead=[1, 1], actual=[0., 0.], predicted=[3., 4.],
                                 p10=[-1., -1.], p90=[1., 1.]))
        result = summarize(data).iloc[0]
        self.assertEqual(result.mae_kw, 3.5)
        self.assertAlmostEqual(result.rmse_kw, np.sqrt(12.5))
        self.assertEqual(result.coverage_80, 1.)


if __name__ == '__main__':
    unittest.main()
