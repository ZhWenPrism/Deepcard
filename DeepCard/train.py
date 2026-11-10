"""Training: stratified 5-fold cross-validation, Adam + cosine annealing + L2,
ensemble by averaging fold probabilities. See manuscript "Cross-validation and
evaluation logic".
"""
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

if __package__:
    from .config import (MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY,
                        MODEL, TRAIN, SEED)
    from .model import DeepCard
    from .losses import MultiTaskLoss, compute_class_weights
    from .data import EchoDataset, add_gaussian_noise
    from .validation import feature_matrix, training_labels
else:
    from config import (MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY,
                        MODEL, TRAIN, SEED)
    from model import DeepCard
    from losses import MultiTaskLoss, compute_class_weights
    from data import EchoDataset, add_gaussian_noise
    from validation import feature_matrix, training_labels


def _make_loss(labels):
    """Build a MultiTaskLoss with inverse-frequency weights from the labels."""
    class_weights, pos_weights = {}, {}
    for t in MULTICLASS_TASKS:
        class_weights[t] = compute_class_weights(labels[t], NUM_SEVERITY)
    for t in BINARY_TASKS:
        y = np.asarray(labels[t]).astype(int)
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - int(y.sum()), 1)
        pos_weights[t] = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    return MultiTaskLoss(class_weights, pos_weights)


def _train_epoch(model, loader, criterion, optimizer, device, noise_std):
    model.train()
    for X, y in loader:
        X = add_gaussian_noise(X.to(device), noise_std)      # augmentation
        y = {t: v.to(device) for t, v in y.items()}
        optimizer.zero_grad()
        out, _ = model(X)
        loss, _ = criterion(out, y)
        loss.backward()
        optimizer.step()


def cross_validate(X, labels, device="cpu"):
    """Stratified 5-fold CV. Returns the list of trained fold models (ensemble)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = feature_matrix(X)
    labels = training_labels(labels, len(X))
    strat = labels[MULTICLASS_TASKS[0]]   # stratify on a primary task
    skf = StratifiedKFold(n_splits=TRAIN["n_folds"], shuffle=True, random_state=SEED)

    models = []
    for fold, (tr, _) in enumerate(skf.split(X, strat)):
        model = DeepCard(MODEL).to(device)
        fold_labels = {t: np.asarray(v)[tr] for t, v in labels.items()}
        criterion = _make_loss(fold_labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"],
                                     weight_decay=TRAIN["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN["cosine_T_max"])

        ds = EchoDataset(X[tr], fold_labels)
        loader = DataLoader(ds, batch_size=TRAIN["batch_size"], shuffle=True)
        for _ in range(TRAIN["epochs"]):
            _train_epoch(model, loader, criterion, optimizer, device,
                         TRAIN["gaussian_noise_std"])
            scheduler.step()
        models.append(model)
        print(f"[fold {fold + 1}/{TRAIN['n_folds']}] training complete")
    return models


@torch.no_grad()
def ensemble_predict(models, X, device="cpu", batch_size=512):
    """Average predicted probabilities across the fold models."""
    models = tuple(models)
    if not models:
        raise ValueError("the ensemble must contain at least one model")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    X = feature_matrix(X)
    for model in models:
        model.to(device).eval()
    predictions = {task: np.empty((len(X), NUM_SEVERITY), dtype=np.float32) for task in MULTICLASS_TASKS}
    predictions.update({task: np.empty(len(X), dtype=np.float32) for task in BINARY_TASKS})
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
        probabilities = {task: [] for task in MULTICLASS_TASKS + BINARY_TASKS}
        for model in models:
            outputs, _ = model(batch)
            for task in MULTICLASS_TASKS:
                probabilities[task].append(torch.softmax(outputs[task], dim=1).cpu().numpy())
            for task in BINARY_TASKS:
                probabilities[task].append(torch.sigmoid(outputs[task]).cpu().numpy())
        for task, values in probabilities.items():
            predictions[task][start:end] = np.mean(values, axis=0)
    return predictions
