"""Convert the official deduplicated Galaxy10 DECaLS HDF5 into PNGs and fixed splits.

Source: https://zenodo.org/records/10845026/files/Galaxy10_DECals_NoDuplicated.h5
Each g<index>.png retains its one-based source row number.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grain
import h5py
import numpy as np
from PIL import Image

GALAXY10_DEDUPLICATED_MD5 = "a920094d5e470e3705f4691c0d6dff54"


def load_image_split_manifest(root, *, split_seed, train_test_split):
    manifest = json.loads((root / "preprocessing_manifest.json").read_text())
    assert (manifest["schema_version"], manifest["dataset"], manifest["split_seed"], manifest["config"]) == (
        2,
        "galaxy10",
        split_seed,
        {"train_test_split": train_test_split},
    ), "Galaxy10 manifest does not match this run."
    result = {name: [] for name in ("train", "val", "test")}
    sample_ids = [record["sample_id"] for record in manifest["records"]]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate Galaxy10 sample IDs."
    for record in manifest["records"]:
        path = root / record["path"]
        assert path.is_file(), path
        result[record["split"]].append((path, int(record["label"])))
    assert {name: len(records) for name, records in result.items()} == manifest["splits"]
    return result


def _read_image(record):
    path, label = record
    with Image.open(path) as image:
        image = image.convert("RGB")
    return {"image": image, "label": np.asarray(label, dtype=np.int32), "is_noisy": np.asarray(False)}


def load(cfg):
    splits = load_image_split_manifest(
        Path(cfg.dataset.data_path).expanduser() / "galaxy10",
        split_seed=cfg.split_seed,
        train_test_split=cfg.dataset.train_test_split,
    )
    return tuple(grain.MapDataset.source(splits[name]).map(_read_image) for name in ("train", "val", "test"))


def stratified_holdout(labels, *, holdout_fraction, seed):
    rng = np.random.RandomState(seed)
    holdout = np.concatenate(
        [
            rng.permutation(np.flatnonzero(labels == class_id))[
                : round(holdout_fraction * np.sum(labels == class_id))
            ]
            for class_id in np.unique(labels)
        ]
    )
    holdout = np.sort(holdout.astype(np.int64))
    keep = np.setdiff1d(np.arange(len(labels), dtype=np.int64), holdout)
    return keep, holdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="./other/data/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--out", default="./other/data/galaxy10")
    parser.add_argument("--train-test-split", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.h5).expanduser()
    output = Path(args.out).expanduser()
    with source.open("rb") as file:
        checksum = hashlib.file_digest(file, "md5").hexdigest()
    assert checksum == GALAXY10_DEDUPLICATED_MD5, "Expected the official deduplicated Galaxy10 file."
    if output.exists() and not args.overwrite:
        load_image_split_manifest(output, split_seed=args.split_seed, train_test_split=args.train_test_split)
        print(f"[galaxy10] matching artifact already exists at {output}")
        return

    with h5py.File(source, "r") as file:
        labels = np.asarray(file["ans"], dtype=np.int64)
        images = np.asarray(file["images"], dtype=np.uint8)
        coordinates = np.column_stack((np.asarray(file["ra"]), np.asarray(file["dec"])))
    assert len(np.unique(coordinates, axis=0)) == len(coordinates), "Duplicate sky coordinates."
    if images.shape[-1] not in (1, 3):
        images = np.moveaxis(images, 1, -1)

    fraction = 1.0 - args.train_test_split
    trainval, test = stratified_holdout(labels, holdout_fraction=fraction, seed=args.split_seed)
    train_local, val_local = stratified_holdout(
        labels[trainval], holdout_fraction=fraction, seed=args.split_seed
    )
    split_of = np.empty(len(labels), dtype=object)
    split_of[test] = "test"
    split_of[trainval[val_local]] = "val"
    split_of[trainval[train_local]] = "train"

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".galaxy10-", dir=output.parent) as temporary:
        staging = Path(temporary) / "converted"
        paths = [
            Path("images") / f"class_{int(label)}" / f"g{index + 1:05d}.png"
            for index, label in enumerate(labels)
        ]
        for directory in {path.parent for path in paths}:
            (staging / directory).mkdir(parents=True, exist_ok=True)

        def encode(index):
            Image.fromarray(images[index]).save(staging / paths[index])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(encode, range(len(labels))), start=1):
                if count % 2000 == 0 or count == len(labels):
                    print(f"[galaxy10] {count}/{len(labels)}", flush=True)
        records = [
            {
                "sample_id": path.stem,
                "path": str(path),
                "label": int(labels[index]),
                "split": str(split_of[index]),
            }
            for index, path in enumerate(paths)
        ]
        manifest = {
            "schema_version": 2,
            "dataset": "galaxy10",
            "split_seed": args.split_seed,
            "config": {"train_test_split": args.train_test_split},
            "splits": {name: int(np.sum(split_of == name)) for name in ("train", "val", "test")},
            "source": {"filename": source.name, "md5": checksum},
            "records": records,
        }
        (staging / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2))
        if output.exists():
            previous = Path(tempfile.mkdtemp(prefix=".galaxy10-previous-", dir=output.parent))
            output.replace(previous / "galaxy10")
            staging.replace(output)
            shutil.rmtree(previous)
        else:
            staging.replace(output)
    print(f"[galaxy10] wrote {len(labels)} samples to {output}", flush=True)
