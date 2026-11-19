import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from DeepCard import run_evaluate, run_train
from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS


class EnsembleManifestTest(unittest.TestCase):
    def test_retraining_with_fewer_folds_excludes_prior_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(torch.nn.Linear(1, 1).state_dict(), root / "deepcard_fold4.pt")
            previous = (root / "deepcard_fold4.pt").read_bytes()
            x = np.zeros((2, 39))
            labels = {t: np.zeros(2, dtype=int) for t in MULTICLASS_TASKS + BINARY_TASKS}
            with patch.object(sys, "argv", ["train", "--train_csv", "synthetic.csv", "--out", tmp]), \
                    patch.object(run_train, "load_xy", return_value=(x, labels)), \
                    patch.object(run_train, "cross_validate", return_value=[torch.nn.Linear(1, 1)]):
                run_train.main()
            model = torch.nn.Linear(1, 1)
            with patch.object(sys, "argv", ["evaluate", "--test_csv", "synthetic.csv", "--artifacts", tmp]), \
                    patch.object(run_evaluate, "load_xy", return_value=(x, labels)), \
                    patch.object(run_evaluate, "DeepCard", return_value=model), \
                    patch.object(run_evaluate, "ensemble_predict", return_value={}) as predict, \
                    patch.object(run_evaluate, "evaluate_all", return_value={}):
                run_evaluate.main()
                self.assertEqual(len(predict.call_args.args[0]), 1)
            self.assertEqual(json.loads((root / "ensemble.json").read_text())["checkpoints"], ["deepcard_fold0.pt"])
            self.assertEqual((root / "deepcard_fold4.pt").read_bytes(), previous)
