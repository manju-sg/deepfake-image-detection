"""Predict whether one image is probably real or AI-generated."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from provenance import human_summary, inspect_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--fake-threshold", type=float, default=0.80)
    parser.add_argument("--real-threshold", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    file_bytes = args.image.read_bytes()
    image = Image.open(args.image).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    fake_probability = probabilities[class_names.index("fake")].item()
    provenance = inspect_provenance(file_bytes)
    high_resolution_upload = max(image.size) > 256

    if provenance["provider_hits"]:
        result = "AI-generated — provider provenance hint"
    elif high_resolution_upload:
        result = "Uncertain — image is outside the 32x32 training domain"
    elif provenance["camera_metadata"] and not provenance["provider_hits"] and fake_probability >= args.fake_threshold:
        result = "Conflict — camera metadata vs visual model"
    elif fake_probability >= args.fake_threshold:
        result = "AI-generated"
    elif fake_probability <= args.real_threshold:
        result = "Probably real"
    else:
        result = "Uncertain"

    print(f"Image: {args.image}")
    print(f"Result: {result}")
    print(f"AI-generated probability: {fake_probability:.2%}")
    print(f"Provenance: {human_summary(provenance)}")


if __name__ == "__main__":
    main()

