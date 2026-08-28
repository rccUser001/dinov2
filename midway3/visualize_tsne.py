"""
t-SNE visualization of DINOv2 embeddings.
Generates PNGs per dataset/model combo and a cross-dataset plot.

Usage: python visualize_tsne.py
Outputs: outputs/tsne_<dataset>_<model>.png
         outputs/tsne_cross_dataset_<model>.png
"""
import os, sys, time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE

OUT = Path(os.environ.get("DINOV2_OUT", Path(__file__).resolve().parent.parent / "outputs"))

# 9-class CRC palette (colorblind-friendly)
CRC_PALETTE = {
    "ADI": "#e6194b", "BACK": "#3cb44b", "DEB": "#ffe119",
    "LYM": "#4363d8", "MUC": "#f58231", "MUS": "#911eb4",
    "NORM": "#42d4f4", "STR": "#f032e6", "TUM": "#bfef45",
}
PCAM_PALETTE = {"normal": "#4363d8", "tumor": "#e6194b"}

MAX_PER_CLASS = 500  # subsample per class for t-SNE speed


def load_pt(path):
    d = torch.load(path, map_location="cpu")
    emb = d["embeddings"].numpy().astype(np.float32)
    labels = d["labels"]
    classes = d["classes"]
    return emb, labels, classes


def subsample(emb, labels, classes, n_per_class):
    idx = []
    for c in classes:
        ci = [i for i, l in enumerate(labels) if l == c]
        idx.extend(ci[:n_per_class])
    idx = sorted(idx)
    return emb[idx], [labels[i] for i in idx]


def run_tsne(emb, perplexity=30, n_iter=1000):
    t0 = time.time()
    proj = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=42, n_jobs=-1, verbose=0).fit_transform(emb)
    print(f"    t-SNE done in {time.time()-t0:.1f}s  ({len(emb)} points)")
    return proj


def plot_tsne(proj, labels, classes, palette, title, out_path, alpha=0.4, s=6):
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in classes:
        mask = np.array([l == c for l in labels])
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=palette.get(c, "#888888"), label=c,
                   alpha=alpha, s=s, linewidths=0)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ── Per-dataset plots ─────────────────────────────────────────────────────────
TASKS = [
    ("crc_val_7k",   "vitl14",    CRC_PALETTE,  "CRC-VAL-HE-7K — ViT-L/14 (7180 patches, 9 classes)"),
    ("crc_val_7k",   "vits14",    CRC_PALETTE,  "CRC-VAL-HE-7K — ViT-S/14 (7180 patches, 9 classes)"),
    ("crc_val_7k",   "vitg14",    CRC_PALETTE,  "CRC-VAL-HE-7K — ViT-G/14 (7180 patches, 9 classes)"),
    ("crc_val_7k",   "hoptimus0", CRC_PALETTE,  "CRC-VAL-HE-7K — H-optimus-0 (7180 patches, 9 classes)"),
    ("crc_val_7k",   "uni",       CRC_PALETTE,  "CRC-VAL-HE-7K — UNI (7180 patches, 9 classes)"),
    ("nct_crc_100k", "vitl14",    CRC_PALETTE,  f"NCT-CRC-100K — ViT-L/14 ({MAX_PER_CLASS}×9 subsample)"),
    ("nct_crc_100k", "vitg14",    CRC_PALETTE,  f"NCT-CRC-100K — ViT-G/14 ({MAX_PER_CLASS}×9 subsample)"),
    ("nct_crc_100k", "hoptimus0", CRC_PALETTE,  f"NCT-CRC-100K — H-optimus-0 ({MAX_PER_CLASS}×9 subsample)"),
    ("nct_crc_100k", "uni",       CRC_PALETTE,  f"NCT-CRC-100K — UNI ({MAX_PER_CLASS}×9 subsample)"),
    ("pcam",         "vitl14",    PCAM_PALETTE, f"PCam — ViT-L/14 ({MAX_PER_CLASS}×2 subsample)"),
    ("pcam",         "vitg14",    PCAM_PALETTE, f"PCam — ViT-G/14 ({MAX_PER_CLASS}×2 subsample)"),
    ("pcam",         "hoptimus0", PCAM_PALETTE, f"PCam — H-optimus-0 ({MAX_PER_CLASS}×2 subsample)"),
    ("pcam",         "uni",       PCAM_PALETTE, f"PCam — UNI ({MAX_PER_CLASS}×2 subsample)"),
]

for dataset, model, palette, title in TASKS:
    pt_path = OUT / f"{dataset}_{model}.pt"
    if not pt_path.exists():
        print(f"  SKIP (not found): {pt_path.name}")
        continue
    print(f"\n  [{dataset} / {model}]")
    emb, labels, classes = load_pt(pt_path)
    emb_sub, labels_sub = subsample(emb, labels, classes, MAX_PER_CLASS)
    proj = run_tsne(emb_sub)
    out_path = OUT / f"tsne_{dataset}_{model}.png"
    plot_tsne(proj, labels_sub, classes, palette, title, out_path)

# ── Cross-dataset plot (CRC + PCam in same space via ViT-L/14) ───────────────
print("\n  [cross-dataset: CRC-VAL-7K + PCam / vitl14]")
crc_emb, crc_labels, crc_classes = load_pt(OUT / "crc_val_7k_vitl14.pt")
pcam_emb, pcam_labels, _ = load_pt(OUT / "pcam_vitl14.pt")

# subsample both
crc_sub, crc_lbl = subsample(crc_emb, crc_labels, crc_classes, 200)
pcam_idx = np.random.RandomState(42).choice(len(pcam_labels), 400, replace=False)
pcam_sub = pcam_emb[pcam_idx]
pcam_lbl = [pcam_labels[i] for i in pcam_idx]

# tag labels with dataset prefix so they're distinct
combined_emb = np.concatenate([crc_sub, pcam_sub], axis=0)
combined_labels = [f"CRC:{l}" for l in crc_lbl] + [f"PCam:{l}" for l in pcam_lbl]
combined_classes = sorted(set(combined_labels))

cross_palette = {f"CRC:{c}": v for c, v in CRC_PALETTE.items()}
cross_palette.update({"PCam:normal": "#00aaff", "PCam:tumor": "#ff4444"})

proj_cross = run_tsne(combined_emb)

fig, ax = plt.subplots(figsize=(11, 8))
for c in combined_classes:
    mask = np.array([l == c for l in combined_labels])
    marker = "o" if c.startswith("CRC") else "^"
    ax.scatter(proj_cross[mask, 0], proj_cross[mask, 1],
               color=cross_palette.get(c, "#888"), label=c,
               alpha=0.5, s=8, marker=marker, linewidths=0)
ax.legend(markerscale=3, framealpha=0.8, fontsize=7, ncol=2)
ax.set_title("Cross-dataset: CRC-VAL-7K (circles) vs PCam (triangles) — ViT-L/14", fontsize=11)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
cross_path = OUT / "tsne_cross_dataset_vitl14.png"
fig.savefig(cross_path, dpi=150)
plt.close(fig)
print(f"    Saved: {cross_path}")

print("\nAll t-SNE plots done.")
