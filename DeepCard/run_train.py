"""Train DeepCard with 5-fold CV and save the ensemble + scaler.

Usage:
    python run_train.py --train_csv data/train.csv --out artifacts
"""
import argparse
import os
import pickle

import torch

if __package__:
    from .feature_engineering import ZScoreScaler
    from .data import load_xy
    from .train import cross_validate
    from .artifacts import write_checkpoint_manifest
else:
    from feature_engineering import ZScoreScaler
    from data import load_xy
    from train import cross_validate
    from artifacts import write_checkpoint_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, labels = load_xy(args.train_csv)
    scaler = ZScoreScaler().fit(X)
    Xs = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = cross_validate(Xs, labels, device=device)

    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(args.out, f"deepcard_fold{i}.pt"))
    with open(os.path.join(args.out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    write_checkpoint_manifest(args.out, [f"deepcard_fold{i}.pt" for i in range(len(models))])
    print(f"Saved {len(models)} fold models + scaler to '{args.out}'")


if __name__ == "__main__":
    main()
