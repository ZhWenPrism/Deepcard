import unittest
import numpy as np

from DeepCard.feature_engineering import ZScoreScaler


class ScalerContractTest(unittest.TestCase):
    def test_transform_requires_a_fitted_scaler(self):
        with self.assertRaises(ValueError):
            ZScoreScaler().transform([[1., 2.]])

    def test_fit_rejects_empty_or_unusable_columns(self):
        for X in ([], np.zeros((0, 2)), [[np.nan, 1.], [np.nan, 2.]], [[np.inf, 1.]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                ZScoreScaler().fit(X)

    def test_transform_rejects_broadcasted_feature_width(self):
        scaler = ZScoreScaler().fit([[1., 2.], [3., 4.]])
        for X in ([[2.]], [2., 3.], [[np.inf, 1.]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                scaler.transform(X)

    def test_partial_missing_and_constant_columns_keep_fitted_statistics(self):
        X = np.array([[1., 5.], [3., 5.], [np.nan, 5.]])
        scaler = ZScoreScaler().fit(X)
        np.testing.assert_allclose(scaler.mean_, [2., 5.])
        np.testing.assert_allclose(scaler.std_, [1., 1.])
        transformed = scaler.transform(X)
        np.testing.assert_allclose(transformed[:2], [[-1., 0.], [1., 0.]])
        self.assertTrue(np.isnan(transformed[2, 0]))
        self.assertEqual(scaler.transform(np.empty((0, 2))).shape, (0, 2))
