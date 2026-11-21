from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from DeepCard import feature_engineering, run_evaluate
from DeepCard.config import BINARY_TASKS, MULTICLASS_TASKS
from DeepCard.feature_engineering import ZScoreScaler


class ScalerEntrypointCompatibilityTest(unittest.TestCase):
    def test_package_evaluation_loads_a_legacy_script_scaler(self):
        scaler = ZScoreScaler().fit(np.zeros((2, 39)))
        with patch.object(ZScoreScaler, "__module__", "feature_engineering"), \
                patch.dict(sys.modules, {"feature_engineering": feature_engineering}):
            legacy = pickle.dumps(scaler)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            sys.modules.pop("feature_engineering", None)
            (Path(tmp) / "scaler.pkl").write_bytes(legacy)
            labels = {task: np.zeros(2, dtype=int) for task in MULTICLASS_TASKS + BINARY_TASKS}
            with patch.object(sys, "argv", ["evaluate", "--test_csv", "synthetic.csv", "--artifacts", tmp]), \
                    patch.object(run_evaluate, "load_xy", return_value=(np.zeros((2, 39)), labels)), \
                    patch.object(run_evaluate, "checkpoint_paths", return_value=[]):
                with self.assertRaisesRegex(SystemExit, "No fold checkpoints"):
                    run_evaluate.main()

    def test_script_loader_reads_a_package_scaler(self):
        scaler = ZScoreScaler().fit(np.zeros((2, 39)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scaler.pkl"
            path.write_bytes(pickle.dumps(scaler))
            script = ("import sys; from artifacts import load_scaler; "
                      "scaler=load_scaler(sys.argv[1]); "
                      "assert scaler.mean_.shape == (39,); assert (scaler.std_ == 1).all()")
            completed = subprocess.run([sys.executable, "-B", "-c", script, str(path)],
                                       cwd=Path(__file__).resolve().parents[2],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
