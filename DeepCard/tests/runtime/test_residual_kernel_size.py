import unittest
import torch
from DeepCard.config import MODEL
from DeepCard.model import DeepCard, ResidualBlock1D


class ResidualKernelSizeTest(unittest.TestCase):
    def test_invalid_kernels_fail_at_construction(self):
        for value in (0, 2, 4, True):
            for factory in (lambda: ResidualBlock1D(8, value),
                            lambda: DeepCard({**MODEL, "kernel_size": value})):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive odd integer"):
                    factory()

    def test_supported_kernels_preserve_the_skip_shape(self):
        torch.manual_seed(17)
        for value in (1, 3, 5):
            block = ResidualBlock1D(8, value, dropout=0).eval()
            self.assertEqual(block(torch.zeros(2, 8, 39)).shape, (2, 8, 39))
