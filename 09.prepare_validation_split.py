from figure_assets import split_chart
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import numpy as np


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "results" / "validation_split_v1"
SPLIT_SEED = 20261120
VALIDATION_SIZE = 400


def make_split(targets):
    known_ids = np.flatnonzero(np.isin(targets, np.arange(8))).astype(np.int64)
    if len(known_ids) != 40000:
        raise ValueError(f"Expected 40,000 known training images, found {len(known_ids)}.")

    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    shuffled = rng.permutation(known_ids)
    validation = np.sort(shuffled[:VALIDATION_SIZE])
    pool = np.sort(shuffled[VALIDATION_SIZE:])

    if np.intersect1d(validation, pool).size:
        raise ValueError("Validation and acquisition pool IDs overlap.")
    if not np.array_equal(np.sort(np.concatenate([validation, pool])), known_ids):
        raise ValueError("The split does not cover the original known training IDs.")
    return validation, pool


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_split(targets, class_names, output):
    targets = np.asarray(targets)
    validation, pool = make_split(targets)
    arrays = {"validation_ids.npy": validation, "pool_ids.npy": pool}
    counts = np.bincount(targets[validation], minlength=8)
    details = {
        "protocol_version": "1.0",
        "dataset": "CIFAR-10",
        "source_split": "train",
        "id_type": "zero-based row index in the original training split",
        "seed": SPLIT_SEED,
        "generator": "PCG64",
        "validation_size": len(validation),
        "pool_size": len(pool),
        "validation_class_counts": dict(zip(class_names[:8], counts.tolist())),
    }

    if output.exists():
        saved = json.loads((output / "split_summary.json").read_text(encoding="utf-8"))
        for name, expected in arrays.items():
            path = output / name
            if not np.array_equal(np.load(path, allow_pickle=False), expected):
                raise ValueError(f"Saved IDs differ from the protocol: {path}")
        details["sha256"] = {name: file_hash(output / name) for name in arrays}
        if saved != details:
            raise ValueError(f"Saved metadata or hashes differ: {output}")
        print("Verified the existing split.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish both arrays and their summary together.
        with TemporaryDirectory(dir=output.parent) as temp:
            folder = Path(temp) / "split"
            folder.mkdir()
            for name, ids in arrays.items():
                np.save(folder / name, ids)
            details["sha256"] = {name: file_hash(folder / name) for name in arrays}
            (folder / "split_summary.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8"
            )
            folder.rename(output)
        print("Saved the validation split.")

    print(f"Validation: {len(validation):,} | Acquisition pool: {len(pool):,}")
    print("Validation counts:", details["validation_class_counts"])
    print("Output:", output)


def main():
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=str(PROJECT / "data"), train=True, download=False)
    prepare_split(train.targets, train.classes, OUTPUT)
    split_chart(__file__, OUTPUT)


if __name__ == "__main__":
    main()
