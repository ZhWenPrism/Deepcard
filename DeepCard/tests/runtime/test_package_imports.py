import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PackageImportsTest(unittest.TestCase):
    def test_runtime_modules_import_through_the_package(self):
        for name in ("model", "losses", "data", "train", "evaluate", "interpret", "run_train", "run_evaluate"):
            with self.subTest(module=name):
                importlib.import_module("DeepCard." + name)

    def test_documented_script_entry_points_still_offer_help(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            for name in ("run_train.py", "run_evaluate.py"):
                result = subprocess.run([sys.executable, str(root / name), "--help"],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

