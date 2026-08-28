"""
KNN evaluation of DINOv2 embeddings on CRC tissue classification.
Standard benchmark protocol: train on NCT-CRC-HE-100K, test on CRC-VAL-HE-7K.
Also runs linear probe (LogisticRegression) for comparison.

Usage: python eval_knn.py
"""
import os, sys, time
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("DINOV2_OUT", Path(__file__).resolve().parent.parent / "outputs"))

def load_pt(path):
    d = torch.load(path, map_location="cpu")
    X = d["embeddings"].numpy().astype(np.float32)
    y = np.array(d["labels"])
    classes = d["classes"]
    return X, y, classes


def evaluate(model_key, train_path, test_path):
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")

    X_train, y_train, classes = load_pt(train_path)
    X_test,  y_test,  _       = load_pt(test_path)
    print(f"Train: {X_train.shape}  classes={classes}")
    print(f"Test : {X_test.shape}")

    # L2-normalize for cosine similarity
    X_train_n = normalize(X_train, norm="l2")
    X_test_n  = normalize(X_test,  norm="l2")

    results = {}

    # ── KNN (k=1, 5, 20) ─────────────────────────────────────────────────────
    for k in [1, 5, 20]:
        print(f"\n  KNN k={k} ...")
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                                   n_jobs=-1, algorithm="brute")
        # Note: on L2-normalized vectors, euclidean ≈ cosine ranking
        knn.fit(X_train_n, y_train)
        y_pred = knn.predict(X_test_n)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - t0
        print(f"    Accuracy : {acc*100:.2f}%  ({elapsed:.1f}s)")
        results[f"knn_k{k}"] = acc

    # Best KNN detailed report
    print(f"\n  KNN k=20 per-class report:")
    knn20 = KNeighborsClassifier(n_neighbors=20, metric="euclidean",
                                 n_jobs=-1, algorithm="brute")
    knn20.fit(X_train_n, y_train)
    y_pred20 = knn20.predict(X_test_n)
    print(classification_report(y_test, y_pred20, target_names=classes))

    # ── Linear probe ─────────────────────────────────────────────────────────
    print(f"\n  Linear probe (LogisticRegression, C=1.0) ...")
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=42)
    lr.fit(X_train_n, y_train)
    y_pred_lr = lr.predict(X_test_n)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    elapsed = time.time() - t0
    print(f"    Accuracy : {acc_lr*100:.2f}%  ({elapsed:.1f}s)")
    results["linear_probe"] = acc_lr
    print(f"\n  Linear probe per-class report:")
    print(classification_report(y_test, y_pred_lr, target_names=classes))

    # ── Confusion matrix (linear probe) ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_lr, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(f"Confusion Matrix — Linear Probe on {model_key}\n"
                 f"Train: NCT-CRC-100K  Test: CRC-VAL-7K  Acc: {acc_lr*100:.2f}%",
                 fontsize=10)
    plt.tight_layout()
    cm_path = OUT / f"confusion_{model_key}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {cm_path}")

    return results


# ── Run for all 3 models ──────────────────────────────────────────────────────
all_results = {}
for model_key in ["vits14", "vitb14", "vitl14", "vitg14", "hoptimus0", "uni"]:
    train_path = OUT / f"nct_crc_100k_{model_key}.pt"
    test_path  = OUT / f"crc_val_7k_{model_key}.pt"
    if not train_path.exists() or not test_path.exists():
        print(f"SKIP {model_key} — files not found")
        continue
    all_results[model_key] = evaluate(model_key, train_path, test_path)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY — Accuracy on CRC-VAL-HE-7K")
print(f"{'Model':<10} {'KNN k=1':>10} {'KNN k=5':>10} {'KNN k=20':>10} {'LinearProbe':>12}")
print("-" * 55)
for model, res in all_results.items():
    print(f"{model:<10} "
          f"{res.get('knn_k1',0)*100:>9.2f}% "
          f"{res.get('knn_k5',0)*100:>9.2f}% "
          f"{res.get('knn_k20',0)*100:>9.2f}% "
          f"{res.get('linear_probe',0)*100:>11.2f}%")

print("\nDONE")
