"""Checkpoint membership for saved DeepCard ensembles."""
import glob
import json
import os
import re
import tempfile


def checkpoint_paths(directory):
    manifest = os.path.join(directory, "ensemble.json")
    if not os.path.isfile(manifest):
        return sorted(glob.glob(os.path.join(directory, "deepcard_fold*.pt")))
    with open(manifest, encoding="utf-8") as handle:
        data = json.load(handle)
    names = data.get("checkpoints") if isinstance(data, dict) else None
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or re.fullmatch(r"deepcard_fold[0-9]+\.pt", name) is None for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("ensemble.json must list unique fold checkpoint filenames")
    return [os.path.join(directory, name) for name in names]


def write_checkpoint_manifest(directory, names):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".ensemble.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump({"checkpoints": names}, handle)
            handle.write("\n")
        os.replace(temporary, os.path.join(directory, "ensemble.json"))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
