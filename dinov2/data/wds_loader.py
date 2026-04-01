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
        shuffle_buffer: Number of samples for in-memory shuffle buffer (0 to disable).
        seed: Base seed for pseudorandom shard shuffling. Each epoch uses
            seed + epoch_number for reproducible but varying order.
    """

    def __init__(
        self,
        shards_path,
        image_transform,
        shard_pattern="tiles-*.tar",
        shuffle_buffer=0,
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

    @staticmethod
    def _warn_and_continue(exn):
        """Log a warning and skip the sample instead of crashing."""
        logger.warning(f"WebDataset: skipping sample due to error: {exn}")
        return True

    def _make_pipeline(self):
        shards = self._get_shuffled_shards()
        pipeline = wds.WebDataset(
            shards,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=self._warn_and_continue,
        ).decode("pil").to_tuple("jpg;jpeg;png", "__key__").map_tuple(self.image_transform, lambda key: ())
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        return pipeline

    def __iter__(self):
        while True:
            logger.info(f"WebDataset: starting epoch {self._epoch} (seed={self.seed + self._epoch})")
            pipeline = self._make_pipeline()
            yield from pipeline
            self._epoch += 1


class PreaugmentedWebDataset(torch.utils.data.IterableDataset):
    """WebDataset loader for pre-augmented shards (output of precompute_augmentations.py).

    Reads shards containing pre-computed crop tensors (crops.pth) instead of
    raw images. Skips all augmentation at training time.

    Args:
        shards_path: Directory containing augmented tar shard files.
        n_global_crops: Number of global crops stored per sample (default: 2).
        n_local_crops: Number of local crops stored per sample (default: 8).
        shard_pattern: Glob pattern for shard files. Default "augmented-*.tar".
        shuffle_buffer: Sample shuffle buffer size. Default 0.
        seed: Base seed for shard shuffling.
    """

    def __init__(
        self,
        shards_path,
        n_global_crops=2,
        n_local_crops=8,
        shard_pattern="augmented-*.tar",
        shuffle_buffer=0,
        seed=0,
    ):
        super().__init__()
        self.shards_path = shards_path
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
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
            f"PreaugmentedWebDataset: found {len(self.shard_urls)} shards "
            f"in {shards_path}"
        )

    def _decode_crops(self, sample):
        """Decode a crops.pth sample into the dict format expected by collate."""
        import io
        crops_bytes = sample["crops.pth"]
        buf = io.BytesIO(crops_bytes)
        tensors = torch.load(buf, weights_only=True)

        output = {
            "global_crops": [tensors[f"global_crop_{i}"] for i in range(self.n_global_crops)],
            "local_crops": [tensors[f"local_crop_{i}"] for i in range(self.n_local_crops)],
            "global_crops_teacher": [tensors[f"global_crop_{i}"] for i in range(self.n_global_crops)],
            "offsets": (),
        }
        return output, ()

    def _get_shuffled_shards(self):
        rng = random.Random(self.seed + self._epoch)
        shards = list(self.shard_urls)
        rng.shuffle(shards)
        return shards

    @staticmethod
    def _warn_and_continue(exn):
        logger.warning(f"PreaugmentedWebDataset: skipping sample: {exn}")
        return True

    def _make_pipeline(self):
        shards = self._get_shuffled_shards()
        pipeline = wds.WebDataset(
            shards,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=self._warn_and_continue,
        ).map(self._decode_crops)
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        return pipeline

    def __iter__(self):
        while True:
            logger.info(f"PreaugmentedWebDataset: starting epoch {self._epoch}")
            pipeline = self._make_pipeline()
            yield from pipeline
            self._epoch += 1


def make_webdataset(cfg_train, image_transform):
    """Factory function to create a WebDatasetWrapper from config.

    Config keys read from cfg_train.slideflow:
        webdataset_path (str): Directory containing shard tar files.
        shard_pattern (str, optional): Glob pattern for shards. Default "tiles-*.tar".
        shuffle_buffer (int, optional): Sample shuffle buffer size. Default 0 (disabled).
        seed (int, optional): Base seed for shard shuffling. Default 0.
        preaugmented (bool, optional): If True, use PreaugmentedWebDataset
            (shards contain pre-computed crops, no augmentation at train time).

    Args:
        cfg_train: Training config with cfg_train.slideflow.webdataset_path.
        image_transform: DataAugmentationDINO transform instance.

    Returns:
        WebDatasetWrapper or PreaugmentedWebDataset instance.
    """
    sf_cfg = cfg_train.slideflow
    shards_path = sf_cfg.webdataset_path

    preaugmented = getattr(sf_cfg, "preaugmented", False)

    kwargs = {}
    if hasattr(sf_cfg, "shard_pattern") and sf_cfg.shard_pattern:
        kwargs["shard_pattern"] = sf_cfg.shard_pattern
    if hasattr(sf_cfg, "shuffle_buffer") and sf_cfg.shuffle_buffer is not None:
        kwargs["shuffle_buffer"] = sf_cfg.shuffle_buffer
    if hasattr(sf_cfg, "seed") and sf_cfg.seed is not None:
        kwargs["seed"] = sf_cfg.seed

    if preaugmented:
        # For preaugmented shards, use default pattern unless overridden
        if "shard_pattern" not in kwargs:
            kwargs["shard_pattern"] = "augmented-*.tar"
        n_local = getattr(cfg_train.crops, "local_crops_number", 8) if hasattr(cfg_train, "crops") else 8
        logger.info(f"Creating PreaugmentedWebDataset from {shards_path}")
        return PreaugmentedWebDataset(
            shards_path=shards_path,
            n_local_crops=n_local,
            **kwargs,
        )

    logger.info(f"Creating WebDataset from {shards_path}")
    return WebDatasetWrapper(
        shards_path=shards_path,
        image_transform=image_transform,
        **kwargs,
    )
