import glob
import logging
import os
import random

import torch
import webdataset as wds

logger = logging.getLogger("dinov2")


def _skip_and_log(exn):
    """Handler for WebDataset that logs warnings and skips bad samples."""
    logger.warning(f"WebDataset: skipping sample due to error: {exn}")
    return True


class WebDatasetWrapper(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset that streams images from WebDataset tar shards.

    Streams from tar shards in a directory, applies DINOv2 augmentations,
    and provides infinite iteration with pseudorandom shard-level and
    sample-level shuffling. Uses wds.split_by_node and wds.split_by_worker
    for distributed sharding.

    Args:
        shards_path: Directory containing tar shard files.
        image_transform: Callable (e.g. DataAugmentationDINO) applied to each PIL image.
        shard_pattern: Glob pattern for shard files relative to shards_path.
            Defaults to "tiles-*.tar".
        shuffle_buffer: Number of samples for in-memory shuffle buffer.
        seed: Base seed for pseudorandom shard shuffling. Each epoch uses
            seed + epoch_number for reproducible but varying order.
    """

    def __init__(
        self,
        shards_path,
        image_transform,
        shard_pattern="tiles-*.tar",
        shuffle_buffer=10000,
        seed=0,
    ):
        super().__init__()
        self.shards_path = shards_path
        self.image_transform = image_transform
        self.shard_pattern = shard_pattern
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0

        self.shard_urls = sorted(
            glob.glob(os.path.join(shards_path, shard_pattern))
        )
        if len(self.shard_urls) == 0:
            raise FileNotFoundError(
                f"No shards matching '{shard_pattern}' found in {shards_path}"
            )
        logger.info(
            f"WebDataset: found {len(self.shard_urls)} shards "
            f"matching '{shard_pattern}' in {shards_path}"
        )

    def _get_shuffled_shards(self):
        """Return shard URLs in pseudorandom order for the current epoch."""
        rng = random.Random(self.seed + self._epoch)
        shards = list(self.shard_urls)
        rng.shuffle(shards)
        return shards

    def _make_pipeline(self):
        shards = self._get_shuffled_shards()
        pipeline = (
            wds.WebDataset(
                shards,
                shardshuffle=False,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                handler=_skip_and_log,
            )
            .shuffle(1000)
            .decode("pil")
            .to_tuple("jpg;jpeg;png", "__key__")
            .map_tuple(self.image_transform, lambda key: ())
            .shuffle(self.shuffle_buffer)
        )
        return pipeline

    def __iter__(self):
        while True:
            logger.info(f"WebDataset: starting epoch {self._epoch} (seed={self.seed + self._epoch})")
            pipeline = self._make_pipeline()
            yield from pipeline
            self._epoch += 1


def make_webdataset(cfg_train, image_transform):
    """Factory function to create a WebDatasetWrapper from config.

    Config keys read from cfg_train.slideflow:
        webdataset_path (str): Directory containing shard tar files.
        shard_pattern (str, optional): Glob pattern for shards. Default "tiles-*.tar".
        shuffle_buffer (int, optional): Sample shuffle buffer size. Default 10000.
        seed (int, optional): Base seed for shard shuffling. Default 0.

    Args:
        cfg_train: Training config with cfg_train.slideflow.webdataset_path.
        image_transform: DataAugmentationDINO transform instance.

    Returns:
        WebDatasetWrapper instance.
    """
    sf_cfg = cfg_train.slideflow
    shards_path = sf_cfg.webdataset_path

    kwargs = {}
    if hasattr(sf_cfg, "shard_pattern") and sf_cfg.shard_pattern:
        kwargs["shard_pattern"] = sf_cfg.shard_pattern
    if hasattr(sf_cfg, "shuffle_buffer") and sf_cfg.shuffle_buffer is not None:
        kwargs["shuffle_buffer"] = sf_cfg.shuffle_buffer
    if hasattr(sf_cfg, "seed") and sf_cfg.seed is not None:
        kwargs["seed"] = sf_cfg.seed

    logger.info(f"Creating WebDataset from {shards_path}")
    return WebDatasetWrapper(
        shards_path=shards_path,
        image_transform=image_transform,
        **kwargs,
    )
