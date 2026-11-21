"""Train DeepCard with 5-fold CV and save the ensemble + scaler.

Usage:
    python run_train.py --train_csv data/train.csv --out artifacts
"""
import argparse
import os

import torch

if __package__:
    from .feature_engineering import ZScoreScaler
    from .data import load_xy
    from .train import cross_validate
    from .artifacts import save_ensemble
else:
    from feature_engineering import ZScoreScaler
    from data import load_xy
    from train import cross_validate
    from artifacts import save_ensemble


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

    save_ensemble(args.out, models, scaler)
    print(f"Saved {len(models)} fold models + scaler to '{args.out}'")


if __name__ == "__main__":
    main()
