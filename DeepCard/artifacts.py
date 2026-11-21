"""Checkpoint membership for saved DeepCard ensembles."""
from contextlib import ExitStack, contextmanager
import glob
import json
import os
import pickle
import re
import tempfile


def _checkpoint_names(names):
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or re.fullmatch(r"deepcard_fold[0-9]+\.pt", name) is None for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("ensemble.json must list unique fold checkpoint filenames")
    return list(names)


@contextmanager
def _staged_artifact(path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=os.path.dirname(os.fspath(path)),
                                         prefix=".artifact.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            yield handle
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_scaler(path):
    if __package__:
        from .feature_engineering import ZScoreScaler
    else:
        from feature_engineering import ZScoreScaler

    class ScalerUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in {"feature_engineering", "DeepCard.feature_engineering"} and name == "ZScoreScaler":
                return ZScoreScaler
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return ScalerUnpickler(handle).load()


def save_ensemble(directory, models, scaler):
    import torch

    models = tuple(models)
    names = _checkpoint_names([f"deepcard_fold{i}.pt" for i in range(len(models))])
    with ExitStack() as staging:
        manifest = staging.enter_context(_staged_artifact(os.path.join(directory, "ensemble.json")))
        manifest.write((json.dumps({"checkpoints": names}) + "\n").encode("utf-8"))
        scaler_file = staging.enter_context(_staged_artifact(os.path.join(directory, "scaler.pkl")))
        pickle.dump(scaler, scaler_file)
        for model, name in zip(models, names):
            checkpoint = staging.enter_context(_staged_artifact(os.path.join(directory, name)))
            torch.save(model.state_dict(), checkpoint)


def checkpoint_paths(directory):
    manifest = os.path.join(directory, "ensemble.json")
    if not os.path.isfile(manifest):
        return sorted(path for path in glob.glob(os.path.join(directory, "deepcard_fold*.pt"))
                      if os.path.isfile(path) and re.fullmatch(r"deepcard_fold[0-9]+\.pt", os.path.basename(path)))
    with open(manifest, encoding="utf-8") as handle:
        data = json.load(handle)
    names = data.get("checkpoints") if isinstance(data, dict) else None
    return [os.path.join(directory, name) for name in _checkpoint_names(names)]


def write_checkpoint_manifest(directory, names):
    names = _checkpoint_names(names)
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
