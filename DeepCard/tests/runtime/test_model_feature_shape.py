import unittest
import torch
from DeepCard.config import MODEL
from DeepCard.model import DeepCard


class ModelFeatureShapeTest(unittest.TestCase):
    def setUp(self):
        self.model = DeepCard({**MODEL, "hidden_channels": 4, "attn_heads": 1,
                               "num_res_blocks": 0, "dropout": 0}).eval()

    def test_incorrect_feature_width_is_rejected_before_prediction(self):
        for shape in ((2, 38), (2, 40), (2, 1, 38), (2, 2, 39), (2, 1, 1, 39)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "39-feature"):
                self.model(torch.zeros(shape))

    def test_flat_and_channel_inputs_produce_identical_logits(self):
        values = torch.zeros(2, 39)
        with torch.no_grad():
            flat, _ = self.model(values)
            channel, _ = self.model(values.unsqueeze(1))
        for task in flat:
            torch.testing.assert_close(flat[task], channel[task])
