"""Train a real-vs-fake image classifier with transfer learning."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform


def run_epoch(model, loader, criterion, device, optimizer=None, fake_index=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_seen = []
    predictions_seen = []
    fake_probs_seen = []

    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        labels_seen.extend(labels.detach().cpu().tolist())
        predictions_seen.extend(predictions.detach().cpu().tolist())
        fake_probs_seen.extend(torch.softmax(logits, dim=1)[:, fake_index].detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(labels_seen, predictions_seen),
        "precision": precision_score(labels_seen, predictions_seen, zero_division=0),
        "recall": recall_score(labels_seen, predictions_seen, zero_division=0),
        "f1": f1_score(labels_seen, predictions_seen, zero_division=0),
    }
    if len(set(labels_seen)) == 2:
        # ImageFolder assigns classes alphabetically: fake=0 and real=1.
        # ROC-AUC expects the class encoded as 1 to be the positive class, so
        # use the real probability here to keep the metric label-consistent.
        real_probs_seen = [1.0 - probability for probability in fake_probs_seen]
        metrics["auc"] = roc_auc_score(labels_seen, real_probs_seen)
    else:
        metrics["auc"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(args.data_dir / "val", transform=eval_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(f"Train/validation classes differ: {train_dataset.classes} vs {val_dataset.classes}")
    if set(train_dataset.classes) != {"fake", "real"}:
        raise ValueError("Expected exactly two folders named 'fake' and 'real'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    fake_index = train_dataset.class_to_idx["fake"]
    best_auc = -float("inf")

    print(f"Device: {device}")
    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, fake_index)
        val_metrics = run_epoch(model, val_loader, criterion, device, fake_index=fake_index)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val F1 {val_metrics['f1']:.4f} | "
            f"val AUC {val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "image_size": args.image_size,
                    "best_val_auc": best_auc,
                },
                args.output,
            )
            print(f"Saved best model to {args.output}")

    test_path = args.data_dir / "test"
    if test_path.exists():
        test_dataset = datasets.ImageFolder(test_path, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        checkpoint = torch.load(args.output, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(model, test_loader, criterion, device, fake_index=fake_index)
        print(
            "Test results | "
            f"accuracy {test_metrics['accuracy']:.4f} | "
            f"precision {test_metrics['precision']:.4f} | "
            f"recall {test_metrics['recall']:.4f} | "
            f"F1 {test_metrics['f1']:.4f} | "
            f"ROC-AUC {test_metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()

