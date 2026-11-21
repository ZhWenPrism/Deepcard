"""Evaluate a trained DeepCard ensemble on a held-out / external CSV.

Usage:
    python run_evaluate.py --test_csv data/internal_test.csv --artifacts artifacts
"""
import argparse
import os

import torch

if __package__:
    from .artifacts import checkpoint_paths, load_scaler
    from .config import MODEL
    from .data import load_xy
    from .model import DeepCard
    from .train import ensemble_predict
    from .evaluate import evaluate_all
else:
    from artifacts import checkpoint_paths, load_scaler
    from config import MODEL
    from data import load_xy
    from model import DeepCard
    from train import ensemble_predict
    from evaluate import evaluate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    X, y_true = load_xy(args.test_csv)
    scaler = load_scaler(os.path.join(args.artifacts, "scaler.pkl"))
    Xs = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in checkpoint_paths(args.artifacts):
        m = DeepCard(MODEL).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        models.append(m)
    if not models:
        raise SystemExit("No fold checkpoints found in artifacts directory.")

    preds = ensemble_predict(models, Xs, device=device)
    results = evaluate_all(y_true, preds)

    for task, metrics in results.items():
        pretty = {k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in metrics.items()}
        print(f"{task}: {pretty}")


if __name__ == "__main__":
    main()
