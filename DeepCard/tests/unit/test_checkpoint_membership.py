import json
from pathlib import Path
import tempfile
import unittest

from DeepCard.artifacts import checkpoint_paths, write_checkpoint_manifest


class CheckpointMembershipTest(unittest.TestCase):
    def test_legacy_discovery_ignores_non_fold_names_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("deepcard_fold0.pt", "deepcard_fold10.pt", "deepcard_fold_notes.pt"):
                (root / name).write_bytes(b"fixture")
            (root / "deepcard_fold2.pt").mkdir()
            self.assertEqual([Path(p).name for p in checkpoint_paths(root)],
                             ["deepcard_fold0.pt", "deepcard_fold10.pt"])

    def test_invalid_membership_cannot_replace_an_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "ensemble.json"
            original = json.dumps({"checkpoints": ["deepcard_fold0.pt"]}).encode()
            for names in ([], ["deepcard_fold0.pt"] * 2, ["../deepcard_fold0.pt"], ["deepcard_fold_notes.pt"]):
                manifest.write_bytes(original)
                with self.subTest(names=names):
                    with self.assertRaises(ValueError):
                        write_checkpoint_manifest(root, names)
                    self.assertEqual(manifest.read_bytes(), original)

    def test_manifest_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ["deepcard_fold10.pt", "deepcard_fold2.pt"]
            write_checkpoint_manifest(tmp, tuple(names))
            self.assertEqual([Path(p).name for p in checkpoint_paths(tmp)], names)
