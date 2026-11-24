"""Download CIFAKE from Hugging Face and create the ImageFolder layout."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "dragonintelligence/CIFAKE-image-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap for quick experiments; omit to export the full dataset.",
    )
    return parser.parse_args()


def export_examples(dataset, indices, split_name: str, output_dir: Path, limit_per_class: int | None) -> None:
    counts = {"fake": 0, "real": 0}
    selected = []

    for index in indices:
        label = "fake" if int(dataset[index]["label"]) == 0 else "real"
        if limit_per_class is None or counts[label] < limit_per_class:
            selected.append((index, label))
            counts[label] += 1
        if limit_per_class is not None and all(value >= limit_per_class for value in counts.values()):
            break

    for index, label in tqdm(selected, desc=f"Exporting {split_name}"):
        image = dataset[index]["image"].convert("RGB")
        target = output_dir / split_name / label / f"{split_name}_{index:06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            image.save(target, format="PNG")

    print(f"{split_name}: {counts}")


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    print(f"Downloading {DATASET_NAME}...")
    train_dataset = load_dataset(DATASET_NAME, split="train", cache_dir=str(args.cache_dir))
    test_dataset = load_dataset(DATASET_NAME, split="test", cache_dir=str(args.cache_dir))

    rng = random.Random(args.seed)
    train_indices = list(range(len(train_dataset)))
    rng.shuffle(train_indices)
    val_size = int(len(train_indices) * args.val_fraction)
    val_indices = train_indices[:val_size]
    actual_train_indices = train_indices[val_size:]

    export_examples(train_dataset, actual_train_indices, "train", args.output_dir, args.limit_per_class)
    export_examples(train_dataset, val_indices, "val", args.output_dir, args.limit_per_class)
    export_examples(test_dataset, list(range(len(test_dataset))), "test", args.output_dir, args.limit_per_class)


if __name__ == "__main__":
    main()

