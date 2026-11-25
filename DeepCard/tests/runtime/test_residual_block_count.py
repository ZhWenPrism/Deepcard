import unittest
import numpy as np
from DeepCard.config import MODEL
from DeepCard.model import DeepCard


class ResidualBlockCountTest(unittest.TestCase):
    def test_invalid_counts_are_rejected(self):
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "non-negative integer"):
                DeepCard({**MODEL, "num_res_blocks": count})

    def test_zero_and_integer_counts_preserve_the_requested_architecture(self):
        for count in (0, 1, np.int64(2)):
            with self.subTest(count=count):
                model = DeepCard({**MODEL, "num_res_blocks": count})
                self.assertEqual(len(model.res_blocks), count)
