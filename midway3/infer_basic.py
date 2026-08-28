"""
Basic DINOv2 inference smoke test.
Loads ViT-S/14 pretrained on LVD-142M, runs 4 random images,
prints embedding shape and GPU utilization.
"""
import os
import sys
import time
import warnings

# Disable xformers (not installed in lean env) and suppress warnings
os.environ["XFORMERS_DISABLED"] = "1"
warnings.filterwarnings("ignore")

# Point torch.hub cache to local scratch copy
TORCH_HOME = "/scratch/beagle3/<user>/<workdir>/torch_hub_cache"
os.environ["TORCH_HOME"] = TORCH_HOME

# Add repo to path so we can import dinov2 package directly
REPO = "/scratch/beagle3/<user>/<workdir>"
sys.path.insert(0, REPO)

import torch
import torchvision.transforms as T

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on      : {device}\n")

# Build ViT-S/14 and load pretrained weights from local cache
print("Loading DINOv2 ViT-S/14 (pretrained, LVD-142M) ...")
t0 = time.time()

from dinov2.hub.backbones import dinov2_vits14
model = dinov2_vits14(pretrained=True)
model = model.to(device).eval()

print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"Parameters      : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

# Standard DINOv2 preprocessing (ImageNet stats, 224x224 at patch_size=14 boundary)
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Generate 4 synthetic RGB images (random noise — sufficient to verify the forward pass)
from PIL import Image
import numpy as np

imgs = []
for i in range(4):
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    imgs.append(transform(Image.fromarray(arr)))

batch = torch.stack(imgs).to(device)
print(f"Input batch     : {tuple(batch.shape)}  (B x C x H x W)")

# Forward pass
print("Running forward pass ...")
t1 = time.time()
with torch.no_grad():
    features = model(batch)
elapsed = time.time() - t1

print(f"\n--- Results ---")
print(f"Output shape    : {tuple(features.shape)}  (B x embed_dim)")
print(f"Embed dim       : {features.shape[1]}  (expected 384 for ViT-S/14)")
print(f"Forward pass    : {elapsed*1000:.1f} ms  ({elapsed*1000/batch.shape[0]:.1f} ms/image)")
print(f"Feature mean    : {features.mean().item():.4f}")
print(f"Feature std     : {features.std().item():.4f}")
print(f"Feature min/max : {features.min().item():.4f} / {features.max().item():.4f}")

if torch.cuda.is_available():
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU mem    : {mem:.2f} GB")

print("\nPASS — DINOv2 ViT-S/14 inference complete.")
