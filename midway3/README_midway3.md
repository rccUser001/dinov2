# DINOv2 + Pathology Foundation Model Inference on Midway3 (RCC / UChicago)

> **Note:** These are exploratory single runs on shared HPC resources — not formal benchmarks,
> not tuned for optimal performance, and not intended for publication. Results and throughput
> numbers may vary across GPU nodes and queue conditions. The primary purpose is to document
> a working replication pipeline on Midway3 so others can reproduce or build on these runs.

This directory contains scripts for running embedding extraction and evaluation
on the RCC Midway3 HPC cluster. Models covered:

- **DINOv2** — ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14 (ImageNet pretrained, fbaipublicfiles)
- **H-optimus-0** — ViT-g/14 + 4 registers, 1134.8M params (Bioptimus, HF-gated, pathology pretrained)
- **UNI** — ViT-L/16, 303.4M params (MahmoodLab, HF-gated, pathology pretrained)

DINOv2 and H-optimus-0 full inference ran on the `gpu` partition.
UNI full inference ran on the `gpu` partition (V100).
Evaluation (t-SNE, KNN) ran on the `caslake` CPU partition.

---

## Environment Setup

```bash
# Use full path — module load doesn't work in non-interactive shells
/software/python-miniforge-25.3.0-el8-x86_64/bin/conda env create \
    --prefix /scratch/beagle3/<user>/dinov2_env \
    --file midway3/env_inference.yaml \
    --yes

# Fix numpy: conda-forge pulls 2.x but torchvision 0.15.0 requires 1.x
conda activate /scratch/beagle3/<user>/dinov2_env
pip install "numpy<2"
pip install scikit-learn matplotlib seaborn
```

**Key env details** (`env_inference.yaml`):
- python=3.9, pytorch=2.0.0+cu117, torchvision=0.15.0
- xformers NOT installed — use `XFORMERS_DISABLED=1` in job scripts
- h5py, scikit-learn 1.6.1, matplotlib, seaborn added via pip after env create

**SLURM scripts use** `set -eo pipefail` (NOT `-u` — conda's MKL activation has unbound vars that trip `-u`).

---

## Pretrained Weights

Compute nodes have no internet. Download weights to a cache dir on the login node first:

```bash
mkdir -p /scratch/beagle3/<user>/torch_hub_cache/hub/checkpoints
cd /scratch/beagle3/<user>/torch_hub_cache/hub/checkpoints

wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth  # 85 MB
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth  # 333 MB
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth  # ~1.1 GB
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth  # ~4.3 GB
```

Set `TORCH_HOME` in every job script:
```bash
export TORCH_HOME=/scratch/beagle3/<user>/torch_hub_cache
```

| Model | Params | Embed dim | Size | License | Source |
|-------|--------|-----------|------|---------|--------|
| ViT-S/14 | 22.1M | 384 | 85 MB | Apache-2.0 | fbaipublicfiles |
| ViT-B/14 | 86.6M | 768 | 333 MB | Apache-2.0 | fbaipublicfiles |
| ViT-L/14 | 304.4M | 1024 | ~1.1 GB | Apache-2.0 | fbaipublicfiles |
| ViT-G/14 | 1136.5M | 1536 | ~4.3 GB | Apache-2.0 | fbaipublicfiles |
| H-optimus-0 | 1134.8M | 1536 | ~4.3 GB | Apache-2.0 | `hf-hub:bioptimus/H-optimus-0` (gated) |
| UNI | 303.4M | 1024 | ~1.2 GB | CC-BY-NC-ND 4.0 | `hf-hub:MahmoodLab/UNI` (gated) |

**License notes:**
- H-optimus-0: Apache-2.0 — permissive; see [model card](https://huggingface.co/bioptimus/H-optimus-0)
- UNI: CC-BY-NC-ND 4.0 — **non-commercial research use only**; do not redistribute weights; attribution required. See [model card](https://huggingface.co/MahmoodLab/UNI). Cite: Chen et al., *Nature Medicine*, 2024.

**HuggingFace gated models** require accepting the user agreement on the HF model page before downloading.
Download on the login node (compute nodes have no internet):
```bash
export HF_HOME=/scratch/beagle3/<user>/torch_hub_cache/hf
python -c "
from huggingface_hub import login, snapshot_download
login(token='<hf-token>')
snapshot_download('bioptimus/H-optimus-0')  # ~4.3 GB
snapshot_download('MahmoodLab/UNI')         # ~1.2 GB
"
```
timm's `hf-hub:` loader requires the standard HF cache layout — do **not** pass `local_dir` to `snapshot_download`.

---

## Datasets

### CRC-VAL-HE-7K and NCT-CRC-HE-100K

Colorectal cancer H&E patches, 9 tissue classes (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM).
224×224 px JPEGs in class subdirectories. Source: [Zenodo 1214456](https://zenodo.org/record/1214456).

```bash
# CRC-VAL-HE-7K (800 MB)
wget https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip
unzip CRC-VAL-HE-7K.zip

# NCT-CRC-HE-100K (11.7 GB)
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip
```

### PCam (PatchCamelyon)

Lymph node H&E patches from Camelyon16. 96×96 px, 2 classes (normal/tumor).
HDF5 format (.h5.gz). Source: [Zenodo 2546921](https://zenodo.org/record/2546921).

```bash
# Test split (~32,768 patches, ~800 MB each)
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz

# Valid split (~32,768 patches)
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz

# Rename for convenience
mv camelyonpatch_level_2_split_test_x.h5.gz  pcam_test_x.h5.gz
mv camelyonpatch_level_2_split_test_y.h5.gz  pcam_test_y.h5.gz
mv camelyonpatch_level_2_split_valid_x.h5.gz pcam_valid_x.h5.gz
mv camelyonpatch_level_2_split_valid_y.h5.gz pcam_valid_y.h5.gz
```

PCam images are resized from 96×96 to 224×224 at load time in `infer_dataset.py`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `env_inference.yaml` | Conda env spec |
| `infer_basic.py` | Smoke test: 4 synthetic images, ViT-S/14 only |
| `infer_dataset.py` | Main inference: lazy DataLoader, file-based (CRC/NCT) + HDF5 (PCam) |
| `visualize_tsne.py` | t-SNE PNGs per dataset/model + cross-dataset plot |
| `eval_knn.py` | KNN (k=1/5/20) + LogisticRegression linear probe; CRC benchmark protocol |
| `run_infer.sh` | SLURM: smoke test (gpu, 1 GPU, 16G, 15 min) |
| `run_crc7k_all.sh` | SLURM: CRC-VAL-HE-7K × 3 models (gpu, 1 GPU, 32G, 30 min) |
| `run_nct100k.sh` | SLURM: NCT-CRC-HE-100K × 3 models (gpu, 1 GPU, 64G, 1 hr) |
| `run_pcam.sh` | SLURM: PCam × 3 models (gpu, 1 GPU, 48G, 1 hr) |
| `run_vitg14_test.sh` | SLURM: ViT-G/14 smoke test — 500-patch CRC subset (gpu, RTX 6000, 1 GPU, 32G, 30 min) |
| `run_vitg14_full.sh` | SLURM: ViT-G/14 full inference — all 3 datasets (gpu, RTX 6000, 1 GPU, 64G, 2 hr) |
| `run_tsne_knn.sh` | SLURM: t-SNE + KNN eval (caslake CPU, 16 cores, 64G, 2 hr) |

### OOM fix

v1 of `infer_dataset.py` preloaded all images into RAM — 100K × 224×224×3 float32 = ~57 GB → OOM at 64G.
v2 uses a lazy `torch.utils.data.DataLoader` — only ~38 MB/batch in RAM at any time. 5–10× throughput improvement for I/O-bound models (ViT-S).

### Output format

Each `.pt` file saved by `infer_dataset.py`:
```python
{
  "embeddings": torch.Tensor(N, embed_dim),  # float32
  "labels":     list[str],                   # per-patch class name
  "classes":    list[str],                   # all unique classes
  "model":      str,                         # e.g. "vitl14"
  "dataset":    str,                         # e.g. "crc_val_7k"
  "n_patches":  int
}
```

---

## Results

### Throughput

| Model | GPU | CRC-VAL-7K | NCT-CRC-100K | PCam (65K) | VRAM |
|-------|-----|-----------|--------------|-----------|------|
| ViT-S/14 | RTX 6000 | 59.0/sec | 332.7/sec | 688.3/sec | 0.45 GB |
| ViT-B/14 | RTX 6000 | 21.1/sec | 221.2/sec | 221.4/sec | 0.99 GB |
| ViT-L/14 | RTX 6000 | 6.6/sec | 66.9/sec | 67.0/sec | 2.05 GB |
| ViT-G/14 | RTX 6000 | 18.0/sec | 18.1/sec | 17.9/sec | 5.33 GB |
| H-optimus-0 | A100 | ~25/sec | 25.4/sec | 25.2/sec | 5.38 GB |
| UNI | V100 | — | ~89/sec | ~89/sec | 1.57 GB |

### KNN + Linear Probe — CRC-VAL-HE-7K Benchmark

Train: NCT-CRC-HE-100K (100K patches, 9 classes) → Test: CRC-VAL-HE-7K (7,180 patches).
Protocol: L2-normalize embeddings → euclidean KNN (≡ cosine on unit vectors) + LogisticRegression linear probe.

| Model | KNN k=1 | KNN k=5 | KNN k=20 | Linear Probe |
|-------|---------|---------|----------|--------------|
| ViT-S/14 | 86.14% | 88.57% | 89.22% | 92.52% |
| ViT-B/14 | 88.79% | 90.39% | 90.95% | 93.38% |
| ViT-L/14 | 87.70% | 90.00% | 90.42% | 93.11% |
| ViT-G/14 | 91.06% | 92.81% | 93.06% | 94.21% |
| H-optimus-0 | 94.50% | 94.97% | 95.28% | **96.21%** |
| UNI | **95.75%** | **96.07%** | **96.24%** | 96.16% |

**Key observations:**
- SOTA on CRC-VAL-HE-7K: ~97%. The ImageNet DINOv2 gap (~3%) is almost entirely closed by pathology pretraining.
- UNI (ViT-L/16, 1.2 GB) outperforms H-optimus-0 (ViT-g/14, 4.3 GB) on KNN — smaller model, better pathology representation.
- Both pathology models exceed DINOv2 ViT-G/14 by ~2% linear probe, using the same or smaller architecture.
- Hardest class across all models: STR (stroma), F1 = 0.57–0.78. Easiest: ADI and BACK, F1 = 0.99–1.00.

**Attribution:**
- UNI: Chen et al., "Towards a General-Purpose Foundation Model for Computational Pathology", *Nature Medicine*, 2024. Model weights used under CC-BY-NC-ND 4.0 for non-commercial research only.
- H-optimus-0: Bioptimus, 2024. Model weights used under Apache-2.0.
- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", *TMLR*, 2024.

### SLURM Job History

| Script | Status | Notes |
|--------|--------|-------|
| run_infer.sh | FAILED | `set -u` tripped MKL unbound var — fixed to `set -eo pipefail` |
| run_infer.sh | PASS | Smoke test; ViT-S/14; output (4,384) |
| run_infer_real.sh | PASS | ⚠️ all 1000 patches from ADI class (sorted+cap bug, superseded) |
| run_crc7k_all.sh | PASS | CRC-VAL-7K × 3 models |
| run_pcam.sh | OOM | 48G — preload bug (v1) |
| run_nct100k.sh | OOM | 64G — preload bug (v1) |
| run_nct100k.sh (v2) | PASS | 100K × 3 models; lazy DataLoader |
| run_pcam.sh (v2) | PASS | 65,536 × 3 models; lazy DataLoader |
| run_tsne_knn.sh | PASS | t-SNE (5 PNGs) + KNN/linear probe; caslake; 3m03s |
| run_vitg14_test.sh | PASS | ViT-G/14 smoke test; 500 patches CRC-VAL-7K; RTX 6000 |
| run_vitg14_full.sh | PASS | ViT-G/14 full inference; all 3 datasets (172,716 patches); 2:41:40; RTX 6000; 5.33 GB VRAM |
| run_tsne_knn.sh | PASS | t-SNE + KNN/linear probe incl. vitg14; caslake; ~5 min |
| run_hoptimus0_full.sh | PASS | H-optimus-0 full inference; all 3 datasets (172,716 patches); A100; 25.2–25.4 patches/sec; 5.38 GB VRAM |
| run_uni_full.sh | PASS | UNI full inference; NCT-CRC-100K + PCam; V100; ~89 patches/sec; 1.57 GB VRAM; 33 min |
| run_tsne_knn.sh | PASS | t-SNE + KNN/linear probe all 6 models; caslake; ~14 min |
