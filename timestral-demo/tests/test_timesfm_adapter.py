import unittest
from types import SimpleNamespace
import numpy as np
from scripts.timesfm_adapter import forecast_at_cutoff


class RecordingModel:
    def predict_batch(self, **kwargs):
        self.arguments = kwargs
        horizon = kwargs['horizon']
        return iter([SimpleNamespace(quantiles=np.tile(np.arange(9.), (horizon, 1)))])


class TimesFMAdapterTests(unittest.TestCase):
    def test_only_history_is_passed_and_deciles_are_selected(self):
        model = RecordingModel()
        target = np.arange(50, dtype=float)
        covariates = np.arange(150, dtype=float).reshape(50, 3)
        # Future values are invalid and must never reach the model or validation.
        target[25:] = np.nan
        covariates[25:] = np.nan
        median, lower, upper = forecast_at_cutoff(model, target, covariates, 25, 9)
        np.testing.assert_array_equal(model.arguments['contexts'][0], target[:25])
        np.testing.assert_array_equal(model.arguments['past_only_covariates'][0], covariates[:25].T)
        self.assertNotIn('past_future_covariates', model.arguments)
        np.testing.assert_array_equal(median, np.full(9, 4.))
        np.testing.assert_array_equal(lower, np.zeros(9))
        np.testing.assert_array_equal(upper, np.full(9, 8.))

    def test_power_only_does_not_supply_covariates(self):
        model = RecordingModel()
        forecast_at_cutoff(model, np.arange(30.), None, 25)
        self.assertNotIn('past_only_covariates', model.arguments)

    def test_bad_quantile_shape_is_rejected(self):
        class BadModel:
            def predict_batch(self, **kwargs):
                return [SimpleNamespace(quantiles=np.zeros((1, 30, 9)))]
        with self.assertRaises(ValueError):
            forecast_at_cutoff(BadModel(), np.arange(40.), None, 30)


if __name__ == '__main__':
    unittest.main()
