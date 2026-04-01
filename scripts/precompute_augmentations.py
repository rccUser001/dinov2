#!/usr/bin/env python3
"""Pre-compute DINOv2 augmentations and write to new WebDataset shards.

Reads raw image shards, applies DataAugmentationDINO N times per tile,
and writes the augmented crops as tensors to new tar shards. This moves
the CPU-heavy augmentation work offline so training just reads tensors.

Usage:
    python scripts/precompute_augmentations.py \
        --input-dir /scratch/wds_shards \
        --output-dir /scratch/wds_augmented \
        --copies 5 \
        --global-crops-size 224 \
        --local-crops-size 96 \
        --local-crops-number 8 \
        --max-samples-per-shard 5000 \
        --num-workers 16
"""

import argparse
import io
import logging
import os
import sys
import time

import torch
import webdataset as wds
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dinov2.data.augmentations import DataAugmentationDINO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def serialize_augmented(output_dict):
    """Serialize the augmented crops dict to a single bytes buffer."""
    buf = io.BytesIO()
    # Save as a flat dict of tensors
    save_dict = {}
    for i, gc in enumerate(output_dict["global_crops"]):
        save_dict[f"global_crop_{i}"] = gc
    for i, lc in enumerate(output_dict["local_crops"]):
        save_dict[f"local_crop_{i}"] = lc
    torch.save(save_dict, buf)
    return buf.getvalue()


def process_shards(args):
    augmentation = DataAugmentationDINO(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        global_crops_size=args.global_crops_size,
        local_crops_size=args.local_crops_size,
    )

    input_shards = sorted(
        [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
         if f.endswith(".tar")]
    )
    logger.info(f"Found {len(input_shards)} input shards in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create a simple pipeline to read raw images
    dataset = (
        wds.WebDataset(input_shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "__key__")
    )

    shard_idx = 0
    sample_in_shard = 0
    sink = None
    total_written = 0
    t0 = time.time()

    def open_new_shard():
        nonlocal shard_idx, sample_in_shard, sink
        if sink is not None:
            sink.close()
        shard_path = os.path.join(
            args.output_dir, f"augmented-{shard_idx:06d}.tar"
        )
        sink = wds.TarWriter(shard_path)
        sample_in_shard = 0
        shard_idx += 1

    open_new_shard()

    for image, key in dataset:
        for copy_idx in range(args.copies):
            # Each copy gets different random augmentations
            augmented = augmentation(image)
            aug_bytes = serialize_augmented(augmented)

            sample_key = f"{key}_aug{copy_idx:03d}"
            sink.write({
                "__key__": sample_key,
                "crops.pth": aug_bytes,
            })
            sample_in_shard += 1
            total_written += 1

            if sample_in_shard >= args.max_samples_per_shard:
                open_new_shard()

        if total_written % 10000 == 0:
            elapsed = time.time() - t0
            rate = total_written / elapsed
            logger.info(
                f"Written {total_written} augmented samples "
                f"({rate:.0f} samples/s, {shard_idx} shards)"
            )

    if sink is not None:
        sink.close()

    elapsed = time.time() - t0
    logger.info(
        f"Done: {total_written} samples in {shard_idx} shards, "
        f"{elapsed:.1f}s ({total_written/elapsed:.0f} samples/s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 augmentations")
    parser.add_argument("--input-dir", required=True, help="Directory with raw image tar shards")
    parser.add_argument("--output-dir", required=True, help="Directory for augmented tar shards")
    parser.add_argument("--copies", type=int, default=5,
                        help="Number of augmented copies per tile (default: 5)")
    parser.add_argument("--max-samples-per-shard", type=int, default=5000,
                        help="Max samples per output shard (default: 5000)")
    parser.add_argument("--global-crops-size", type=int, default=224)
    parser.add_argument("--local-crops-size", type=int, default=96)
    parser.add_argument("--local-crops-number", type=int, default=8)
    parser.add_argument("--global-crops-scale", nargs=2, type=float, default=[0.32, 1.0])
    parser.add_argument("--local-crops-scale", nargs=2, type=float, default=[0.05, 0.32])
    args = parser.parse_args()
    process_shards(args)


if __name__ == "__main__":
    main()
