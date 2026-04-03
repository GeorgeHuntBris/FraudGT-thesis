"""
Random Forest baseline on Elliptic, DGraph, and Bitcoin-M datasets.
Uses node features only (no graph structure).
Run: python run_rf.py
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import os

DATA_DIR = os.path.expanduser("~/FraudGT-thesis/data")
N_SEEDS = 3


# Takes a trained classifier, feature matrix X and true labels y
def evaluate(clf, X, y):
    # Hard predictions for each node
    pred = clf.predict(X)

    # Returns col 0 = prob normal and col 1 = prob fraud
    prob = clf.predict_proba(X)[:, 1]

    f1   = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.0
    ap   = average_precision_score(y, prob) if len(np.unique(y)) > 1 else 0.0
    return f1, prec, rec, auc, ap


def run_dataset(name, data_path):
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST — {name}")
    print(f"{'='*60}")

    # Convert binary file back into python object
    data_dict = torch.load(data_path, weights_only=False)

    data = data_dict['train']  # use train split to get full masks (could use any i.e train/val/test)
    x = data['node'].x.numpy()
    y = data['node'].y.numpy()

    train_mask = data['node'].train_mask.numpy()
    val_mask   = data['node'].val_mask.numpy()
    test_mask  = data['node'].test_mask.numpy()

    # Filter out unlabeled nodes (label == -1) (we can include unlabeled for GNNs but not for the RF)
    train_idx = np.where(train_mask & (y != -1))[0]
    val_idx   = np.where(val_mask   & (y != -1))[0]
    test_idx  = np.where(test_mask  & (y != -1))[0]

    X_train, y_train = x[train_idx], y[train_idx]
    X_val,   y_val   = x[val_idx],   y[val_idx]
    X_test,  y_test  = x[test_idx],  y[test_idx]

    print(f"Train: {len(X_train)} nodes | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train fraud rate: {y_train.mean():.4f}")

    val_results  = []
    test_results = []

    for seed in range(N_SEEDS):
        clf = RandomForestClassifier(
            n_estimators=50,
            max_features=50,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        vf, vp, vr, va, vap = evaluate(clf, X_val, y_val)
        tf, tp, tr, ta, tap = evaluate(clf, X_test, y_test)

        val_results.append((vf, vp, vr, va, vap))
        test_results.append((tf, tp, tr, ta, tap))

        print(f"  Seed {seed} | Val  F1={vf:.4f} Prec={vp:.4f} Rec={vr:.4f} AUC={va:.4f} AP={vap:.4f}")
        print(f"           | Test F1={tf:.4f} Prec={tp:.4f} Rec={tr:.4f} AUC={ta:.4f} AP={tap:.4f}")

    val_arr  = np.array(val_results)
    test_arr = np.array(test_results)

    print(f"\n  MEAN Val  | F1={val_arr[:,0].mean():.4f}±{val_arr[:,0].std():.4f} "
          f"Prec={val_arr[:,1].mean():.4f} Rec={val_arr[:,2].mean():.4f} "
          f"AUC={val_arr[:,3].mean():.4f} AP={val_arr[:,4].mean():.4f}")
    print(f"  MEAN Test | F1={test_arr[:,0].mean():.4f}±{test_arr[:,0].std():.4f} "
          f"Prec={test_arr[:,1].mean():.4f} Rec={test_arr[:,2].mean():.4f} "
          f"AUC={test_arr[:,3].mean():.4f} AP={test_arr[:,4].mean():.4f}")


if __name__ == "__main__":

    datasets = [
        ("Elliptic",  os.path.join(DATA_DIR, "Elliptic/processed/data.pt")),
        ("DGraph",    os.path.join(DATA_DIR, "DGraph/processed/data.pt")),
        ("Bitcoin-M", os.path.join(DATA_DIR, "BitcoinM/processed/data.pt")),
    ]

    for name, path in datasets:
        if os.path.exists(path):
            run_dataset(name, path)
        else:
            print(f"\nSkipping {name} — data not found at {path}")
