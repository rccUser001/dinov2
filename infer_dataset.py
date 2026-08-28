"""
General DINOv2 inference script — lazy loading, no full-dataset preload.
Usage:
  python infer_dataset.py --dataset <name> --data-dir <path> \
    [--format file|pcam] [--models vits14 vitb14 vitl14] \
    [--batch-size 64] [--max-patches N]

Formats:
  file : directory tree of images (jpg/png/tif) with class subdirs
  pcam : PatchCamelyon HDF5 (.h5 or .h5.gz), shape (N,96,96,3)

Outputs: outputs/<dataset>_<model>.pt
  keys: embeddings (N, dim), labels, classes, model, dataset, n_patches
"""
import os, sys, argparse, time, warnings, gzip, shutil
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"
warnings.filterwarnings("ignore")
# TORCH_HOME and HF_HOME should be set in the job script or environment

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    required=True)
parser.add_argument("--data-dir",   required=True)
parser.add_argument("--format",     default="file", choices=["file", "pcam"])
parser.add_argument("--models",     nargs="+", default=["vits14", "vitb14"],
                    choices=["vits14", "vitb14", "vitl14", "vitg14", "hoptimus0", "uni"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--max-patches",type=int, default=0)
args = parser.parse_args()

DATASET  = args.dataset
DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path(os.environ.get("DINOV2_OUT", Path(__file__).parent / "outputs"))
OUT_DIR.mkdir(exist_ok=True)
BATCH    = args.batch_size
MAX      = args.max_patches or None
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Dataset : {DATASET}  format={args.format}  models={args.models}")
print()

def make_transform(mean, std):
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

# ImageNet normalization — used for all DINOv2 models
TRANSFORM_IMAGENET = make_transform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Pathology normalization — H-optimus-0 trained on H&E histology
TRANSFORM_PATHOLOGY = make_transform([0.707223, 0.578729, 0.703421],
                                     [0.211883, 0.230117, 0.177517])

transform = TRANSFORM_IMAGENET  # default; overridden per model in main loop

# ── Datasets (lazy) ──────────────────────────────────────────────────────────
class FileDataset(Dataset):
    def __init__(self, root, transform, max_n=None):
        exts = {".tif", ".png", ".jpg", ".jpeg"}
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)
        if max_n:
            paths = paths[:max_n]
        self.paths   = paths
        self.labels  = [p.parent.name for p in paths]
        self.classes = sorted(set(self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class PcamDataset(Dataset):
    """Reads PCam HDF5 lazily (decompresses .gz on first access)."""
    def __init__(self, data_dir, transform, max_n=None):
        import h5py
        data_dir = Path(data_dir)
        # decompress any .gz files
        for gz in sorted(data_dir.glob("*.h5.gz")):
            h5 = gz.with_suffix("")
            if not h5.exists():
                print(f"  Decompressing {gz.name} ...")
                with gzip.open(gz, "rb") as fi, open(h5, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        # collect all x/y pairs
        x_files = sorted(data_dir.glob("*_x.h5"))
        y_files = {f.stem.replace("_x", "_y"): data_dir / f"{f.stem.replace('_x','_y')}.h5"
                   for f in x_files}
        self.x_files = x_files
        self.y_files = y_files
        self.transform = transform
        # build index: list of (file_idx, row_idx)
        self._index = []
        self._handles = []
        for fi, xf in enumerate(x_files):
            with h5py.File(xf, "r") as f:
                n = len(f["x"])
            self._index.extend((fi, r) for r in range(n))
        if max_n:
            self._index = self._index[:max_n]
        self.classes = ["normal", "tumor"]
        # cache open file handles
        import h5py as _h5py
        self._h5py = _h5py
        self._xh = [_h5py.File(xf, "r") for xf in x_files]
        yf_paths = [data_dir / f"{xf.stem.replace('_x','_y')}.h5" for xf in x_files]
        self._yh = [_h5py.File(yf, "r") if yf.exists() else None for yf in yf_paths]
        self.labels = None  # built lazily

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        fi, ri = self._index[i]
        img = self._xh[fi]["x"][ri]          # uint8, (96,96,3)
        label_val = int(self._yh[fi]["y"][ri].flat[0]) if self._yh[fi] else 0
        label = "tumor" if label_val == 1 else "normal"
        return self.transform(Image.fromarray(img)), label

    def all_labels(self):
        """Collect labels for all patches (iterates HDF5 y files)."""
        labels = []
        for fi, ri in self._index:
            val = int(self._yh[fi]["y"][ri].flat[0]) if self._yh[fi] else 0
            labels.append("tumor" if val == 1 else "normal")
        return labels


# ── Model registry ───────────────────────────────────────────────────────────
MODEL_FNS = {
    "vits14": ("dinov2.hub.backbones", "dinov2_vits14",  384),
    "vitb14": ("dinov2.hub.backbones", "dinov2_vitb14",  768),
    "vitl14": ("dinov2.hub.backbones", "dinov2_vitl14", 1024),
    "vitg14": ("dinov2.hub.backbones", "dinov2_vitg14", 1536),
}

# timm/HuggingFace models: (hf_repo, embed_dim, transform)
TIMM_MODELS = {
    "hoptimus0": ("hf-hub:bioptimus/H-optimus-0", 1536, TRANSFORM_PATHOLOGY),
    "uni":       ("hf-hub:MahmoodLab/UNI",         1024, TRANSFORM_IMAGENET),
}

# Which transform does each model key need?
MODEL_TRANSFORMS = {k: TRANSFORM_IMAGENET for k in MODEL_FNS}
MODEL_TRANSFORMS.update({k: v[2] for k, v in TIMM_MODELS.items()})


def load_model(model_key):
    if model_key in TIMM_MODELS:
        import timm as _timm
        hf_path, expected_dim, _ = TIMM_MODELS[model_key]
        print(f"  Loading {model_key} from {hf_path} ...")
        model = _timm.create_model(hf_path, pretrained=True,
                                   init_values=1e-5, dynamic_img_size=True)
    else:
        import importlib
        mod_path, fn_name, expected_dim = MODEL_FNS[model_key]
        fn = getattr(importlib.import_module(mod_path), fn_name)
        print(f"  Loading {model_key} ...")
        model = fn(pretrained=True)
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params : {params:.1f}M  embed_dim : {expected_dim}")
    return model


def run_model(model_key, loader, n_total):
    model = load_model(model_key)

    # warm-up
    dummy = torch.randn(BATCH, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(2):
            model(dummy)
    torch.cuda.synchronize()

    embeddings, all_labels = [], []
    t0 = time.time()
    for imgs, lbls in loader:
        with torch.no_grad():
            embeddings.append(model(imgs.to(device)).cpu())
        all_labels.extend(lbls)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    embeddings = torch.cat(embeddings, 0)
    n = embeddings.shape[0]
    throughput = n / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Processed : {n}  |  {throughput:.1f} patches/sec  |  {elapsed:.1f}s")
    print(f"  Output    : {tuple(embeddings.shape)}  |  peak VRAM : {peak_mem:.2f} GB")
    print(f"  mean/std  : {embeddings.mean():.4f} / {embeddings.std():.4f}")

    out_path = OUT_DIR / f"{DATASET}_{model_key}.pt"
    classes = sorted(set(all_labels))
    torch.save({"embeddings": embeddings, "labels": all_labels,
                "classes": classes, "model": model_key,
                "dataset": DATASET, "n_patches": n}, out_path)
    print(f"  Saved     : {out_path}")
    torch.cuda.reset_peak_memory_stats()
    del model
    return throughput

# ── Main ─────────────────────────────────────────────────────────────────────
def build_loader(tfm):
    if args.format == "file":
        ds = FileDataset(DATA_DIR, tfm, MAX)
        print(f"  {len(ds)} patches  |  {len(ds.classes)} classes: {ds.classes}")
    elif args.format == "pcam":
        ds = PcamDataset(DATA_DIR, tfm, MAX)
        print(f"  {len(ds)} patches  |  classes: {ds.classes}")
    return ds, DataLoader(ds, batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

print("=== Loading dataset (ImageNet transform) ===")
ds, loader = build_loader(TRANSFORM_IMAGENET)

print(f"\n=== Running models ===")
results = {}
_last_tfm = TRANSFORM_IMAGENET
for model_key in args.models:
    print(f"\n--- {model_key} ---")
    tfm = MODEL_TRANSFORMS[model_key]
    if tfm is not _last_tfm:
        print(f"  Rebuilding loader with {model_key} transform ...")
        ds, loader = build_loader(tfm)
        _last_tfm = tfm
    results[model_key] = run_model(model_key, loader, len(ds))

print("\n=== Throughput Summary ===")
for k, v in results.items():
    print(f"  {k:8s}: {v:.1f} patches/sec")
print(f"\nAll embeddings saved to {OUT_DIR}/")
print("DONE")
