import glob
import logging
import os

import torch
import webdataset as wds

logger = logging.getLogger("dinov2")


class WebDatasetWrapper(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset that streams images from WebDataset tar shards.

    Finds all tiles-*.tar shards in a directory, applies DINOv2 augmentations,
    and provides infinite iteration with shard-level and sample-level shuffling.
    Uses wds.split_by_node and wds.split_by_worker for distributed sharding.
    """

    def __init__(self, shards_path, image_transform, shuffle_buffer=10000):
        super().__init__()
        self.shards_path = shards_path
        self.image_transform = image_transform
        self.shuffle_buffer = shuffle_buffer

        self.shard_urls = sorted(glob.glob(os.path.join(shards_path, "tiles-*.tar")))
        if len(self.shard_urls) == 0:
            raise FileNotFoundError(
                f"No tiles-*.tar shards found in {shards_path}"
            )
        logger.info(f"WebDataset: found {len(self.shard_urls)} shards in {shards_path}")

    def _make_pipeline(self):
        pipeline = (
            wds.WebDataset(self.shard_urls, shardshuffle=True)
            .shuffle(1000)
            .pipe(wds.split_by_node)
            .pipe(wds.split_by_worker)
            .decode("pil")
            .to_tuple("jpg;jpeg;png", "__key__")
            .map_tuple(self.image_transform, lambda key: ())
            .shuffle(self.shuffle_buffer)
        )
        return pipeline

    def __iter__(self):
        while True:
            pipeline = self._make_pipeline()
            yield from pipeline


def make_webdataset(cfg_train, image_transform):
    """Factory function to create a WebDatasetWrapper from config.

    Args:
        cfg_train: Training config with cfg_train.slideflow.webdataset_path.
        image_transform: DataAugmentationDINO transform instance.

    Returns:
        WebDatasetWrapper instance.
    """
    shards_path = cfg_train.slideflow.webdataset_path
    logger.info(f"Creating WebDataset from {shards_path}")
    return WebDatasetWrapper(
        shards_path=shards_path,
        image_transform=image_transform,
    )
