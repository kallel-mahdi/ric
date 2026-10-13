"""Deterministic Grain pipelines and device-staged evaluation splits."""

from contextlib import closing
from dataclasses import dataclass
from functools import partial
from math import ceil

import grain
import jax
import numpy as np
from flax import struct
from torchvision.transforms import functional as transforms

from ric.data.datasets import DATASETS, MEDMNIST, NORM_STATS


@dataclass(frozen=True)
class Data:
    train: grain.IterDataset
    val: grain.MapDataset
    test: grain.MapDataset
    train_size: int
    num_classes: int
    norm_stats: tuple


@struct.dataclass
class EvalSplit:
    images: jax.Array
    labels: jax.Array
    mask: jax.Array


def _resized_crop(image, rng, image_size):
    width, height = image.size
    for _ in range(10):
        area = width * height * rng.uniform(0.08, 1.0)
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        crop_width, crop_height = round(np.sqrt(area * ratio)), round(np.sqrt(area / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(rng.integers(height - crop_height + 1))
            left = int(rng.integers(width - crop_width + 1))
            return transforms.resized_crop(
                image, top, left, crop_height, crop_width, [image_size, image_size]
            )
    ratio = width / height
    if ratio < 3 / 4:
        crop_width, crop_height = width, round(width / (3 / 4))
    elif ratio > 4 / 3:
        crop_width, crop_height = round(height * (4 / 3)), height
    else:
        crop_width, crop_height = width, height
    return transforms.resized_crop(
        image,
        (height - crop_height) // 2,
        (width - crop_width) // 2,
        crop_height,
        crop_width,
        [image_size, image_size],
    )


def _augment(record, rng, *, image_size, medical):
    image = record["image"]
    if medical or image_size <= 64:
        image = transforms.pad(image, 4)
        width, height = image.size
        top = int(rng.integers(height - image_size + 1))
        left = int(rng.integers(width - image_size + 1))
        image = transforms.crop(image, top, left, image_size, image_size)
    else:
        image = _resized_crop(image, rng, image_size)
    # OrganSMNIST distinguishes left/right organs; all MedMNIST uses crop only.
    if not medical and rng.random() < 0.5:
        image = transforms.hflip(image)
    return record | {"image": np.asarray(image, dtype=np.uint8)}


def _prepare_eval(record, *, image_size, medical):
    image = record["image"]
    if not medical and image_size > 64:
        image = transforms.center_crop(transforms.resize(image, round(image_size * 256 / 224)), image_size)
    return record | {"image": np.asarray(image, dtype=np.uint8)}


def build_data(cfg, *, shuffle_seed):
    train, val, test = DATASETS[cfg.dataset.name](cfg)
    rng = np.random.RandomState(shuffle_seed)
    indices = np.arange(len(train))
    if cfg.dataset.dataset_fraction < 1:
        count = max(1, int(len(train) * cfg.dataset.dataset_fraction))
        indices = rng.choice(indices, size=count, replace=False)
    indices = rng.permutation(indices)
    # MapDataset.slice only accepts slices in Grain 0.2.18.
    selected = grain.MapDataset.source(indices).map(train.__getitem__)
    options = {"image_size": cfg.dataset.image_size, "medical": cfg.dataset.name in MEDMNIST}
    train_pipeline = selected.seed(shuffle_seed).shuffle().repeat()
    if cfg.dataset.augmentation:
        train_pipeline = train_pipeline.random_map(partial(_augment, **options))
    else:
        train_pipeline = train_pipeline.map(partial(_prepare_eval, **options))
    train_pipeline = (
        train_pipeline.batch(cfg.algorithm.train_batch_size, drop_remainder=True)
        # Buffer one scan chunk across readers, rather than a full chunk in every worker.
        .to_iter_dataset(
            grain.ReadOptions(
                num_threads=1,
                prefetch_buffer_size=ceil(
                    cfg.algorithm.train_scan_chunk_batches / max(cfg.data_loader.train_worker_count, 1)
                ),
            )
        )
        .mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=cfg.data_loader.train_worker_count,
                per_worker_buffer_size=cfg.data_loader.train_worker_buffer_size,
            )
        )
    )
    return Data(
        train=train_pipeline,
        val=val.map(partial(_prepare_eval, **options)),
        test=test.map(partial(_prepare_eval, **options)),
        train_size=len(indices),
        num_classes=cfg.dataset.num_classes,
        norm_stats=NORM_STATS[cfg.dataset.normalization],
    )


def stage(dataset, batch_size):
    """Decode once into padded uint8 arrays; the mask excludes padding from every metric."""
    count = len(dataset)
    batch_size = min(batch_size, count)
    num_batches = -(-count // batch_size)
    labels = np.zeros((num_batches, batch_size), dtype=np.int32)
    batches = dataset.batch(batch_size).to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    )
    with closing(iter(batches)) as iterator:
        for index, batch in enumerate(iterator):
            if index == 0:
                images = np.zeros((num_batches, batch_size, *batch["image"].shape[1:]), dtype=np.uint8)
            images[index, : len(batch["label"])] = batch["image"]
            labels[index, : len(batch["label"])] = batch["label"]
    mask = np.arange(num_batches * batch_size).reshape(labels.shape) < count
    return jax.device_put(EvalSplit(images, labels, mask))
