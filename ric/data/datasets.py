"""Paper datasets, fixed splits, and device-side image normalization."""

from pathlib import Path

import grain
import jax.numpy as jnp
import medmnist
import numpy as np
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive

from ric.data import galaxy10

NORM_STATS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "cifar": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "medmnist": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # RGB moments of the fixed Galaxy10 DECaLS training manifest.
    "galaxy10": (
        [0.167564122821, 0.162647174765, 0.158951437903],
        [0.128820360482, 0.118095699697, 0.111642278880],
    ),
}


def normalize_images(images, norm_stats):
    mean, std = norm_stats
    # Preserve ToTensor + Normalize's float32 order, including with bf16 encoders.
    return (images.astype(jnp.float32) / 255.0 - jnp.asarray(mean)) / jnp.asarray(std)


def split_indices(num_samples, train_fraction, seed):
    train_size = int(train_fraction * num_samples)
    train, val = torch.utils.data.random_split(
        range(num_samples),
        [train_size, num_samples - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.asarray(train.indices), np.asarray(val.indices)


def _records(dataset, clean_labels):
    def record(index, example):
        image, label = example
        label = np.asarray(label, dtype=np.int32).reshape(())
        return {"image": image, "label": label, "is_noisy": np.asarray(label != clean_labels[index])}

    return grain.MapDataset.source(dataset).map_with_index(record)


def _official_splits(cfg, train, test, clean_train_labels, test_labels):
    train_indices, val_indices = split_indices(len(train), cfg.dataset.train_test_split, cfg.split_seed)
    return (
        _records(torch.utils.data.Subset(train, train_indices), clean_train_labels[train_indices]),
        _records(torch.utils.data.Subset(train, val_indices), clean_train_labels[val_indices]),
        _records(test, test_labels),
    )


def _cifar10(cfg):
    root, download = str(cfg.dataset.data_path), cfg.dataset.download
    train = datasets.CIFAR10(root=root, train=True, download=download)
    test = datasets.CIFAR10(root=root, train=False, download=download)
    clean = np.asarray(train.targets, dtype=np.int32)
    if cfg.dataset.name == "cifar10n":
        labels = torch.load(
            Path(cfg.dataset.noise_labels_path).expanduser(), map_location="cpu", weights_only=False
        )
        train.targets = np.asarray(labels[cfg.dataset.noise_label_key], dtype=np.int32).tolist()
        clean = np.asarray(labels["clean_label"], dtype=np.int32)
    return _official_splits(cfg, train, test, clean, np.asarray(test.targets, dtype=np.int32))


def _svhn(cfg):
    root, download = str(cfg.dataset.data_path), cfg.dataset.download
    train = datasets.SVHN(root=root, split="train", download=download)
    test = datasets.SVHN(root=root, split="test", download=download)
    return _official_splits(cfg, train, test, np.asarray(train.labels), np.asarray(test.labels))


def _imagewoof(cfg):
    folder = {"full": "imagewoof2", "320px": "imagewoof2-320", "160px": "imagewoof2-160"}[
        cfg.dataset.fastai_size
    ]
    root = Path(cfg.dataset.data_path).expanduser()
    if cfg.dataset.download and not (root / folder).exists():
        download_and_extract_archive(
            url=f"https://s3.amazonaws.com/fast-ai-imageclas/{folder}.tgz", download_root=str(root)
        )
    train = datasets.ImageFolder(str(root / folder / "train"))
    test = datasets.ImageFolder(str(root / folder / "val"))
    return _official_splits(cfg, train, test, np.asarray(train.targets), np.asarray(test.targets))


MEDMNIST = {
    "dermamnist": medmnist.DermaMNIST,
    "octmnist": medmnist.OCTMNIST,
    "organsmnist": medmnist.OrganSMNIST,
    "pneumoniamnist": medmnist.PneumoniaMNIST,
    "tissuemnist": medmnist.TissueMNIST,
}


def _medmnist(cfg):
    splits = []
    for split in ("train", "val", "test"):
        dataset = MEDMNIST[cfg.dataset.name](
            split=split,
            root=str(cfg.dataset.data_path),
            download=cfg.dataset.download,
            as_rgb=True,
            size=cfg.dataset.image_size,
        )
        splits.append(_records(dataset, dataset.labels.reshape(-1)))
    return tuple(splits)


DATASETS = {
    "cifar10": _cifar10,
    "cifar10n": _cifar10,
    "svhn": _svhn,
    "imagewoof": _imagewoof,
    "galaxy10": galaxy10.load,
    **dict.fromkeys(MEDMNIST, _medmnist),
}
