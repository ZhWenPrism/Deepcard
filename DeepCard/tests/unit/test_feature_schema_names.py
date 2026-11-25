import unittest
from DeepCard.datasets.schema import FeatureSchema


class FeatureSchemaNamesTest(unittest.TestCase):
    def test_invalid_columns_cannot_form_a_39_feature_schema(self):
        for name in ("", " ", None, 0, " feature-0"):
            names = [f"feature-{i}" for i in range(39)]
            names[-1] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                FeatureSchema(names)

    def test_valid_feature_order_and_units_are_preserved(self):
        names = tuple(f"feature-{i}" for i in range(39))
        schema = FeatureSchema(names, {"feature-0": "mm"})
        self.assertEqual(schema.names, names)
        self.assertEqual(schema.units["feature-0"], "mm")
