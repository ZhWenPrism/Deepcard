import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from DeepCard import run_train
from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS


class ArtifactSerializationFailureTest(unittest.TestCase):
    def test_serializer_failures_preserve_all_previous_artifacts(self):
        def fail_after_writing(value, destination, *args, **kwargs):
            if hasattr(destination, "write"):
                destination.write(b"partial")
            else:
                Path(destination).write_bytes(b"partial")
            raise OSError("serialization failed")

        x = np.zeros((2, 39))
        labels = {task: np.zeros(2, dtype=int) for task in MULTICLASS_TASKS + BINARY_TASKS}
        for serializer in ("torch.save", "pickle.dump"):
            with self.subTest(serializer=serializer), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                originals = {"deepcard_fold0.pt": b"previous checkpoint", "scaler.pkl": b"previous scaler",
                             "ensemble.json": json.dumps({"checkpoints": ["deepcard_fold0.pt"]}).encode()}
                for name, content in originals.items():
                    (root / name).write_bytes(content)
                with patch.object(sys, "argv", ["train", "--train_csv", "synthetic.csv", "--out", tmp]), \
                        patch.object(run_train, "load_xy", return_value=(x, labels)), \
                        patch.object(run_train, "cross_validate", return_value=[torch.nn.Linear(1, 1)]), \
                        patch(serializer, side_effect=fail_after_writing):
                    with self.assertRaisesRegex(OSError, "serialization failed"):
                        run_train.main()
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, originals)
