"""Streamlit upload interface for the trained detector."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from provenance import human_summary, inspect_provenance


MODEL_PATH = Path("best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_detector():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    image_size = checkpoint["image_size"]
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, class_names, image_transform


st.set_page_config(page_title="Deep-Fake Detector", page_icon="🔍", layout="centered")
st.title("🔍 Deep-Fake Image Detector")
st.write("Upload an image to estimate whether it is real or AI-generated.")
st.caption(f"Running on: {DEVICE}")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model, class_names, image_transform = load_detector()
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    fake_probability = probabilities[class_names.index("fake")].item()
    real_probability = probabilities[class_names.index("real")].item()
    provenance = inspect_provenance(file_bytes)
    high_resolution_upload = max(image.size) > 256

    if provenance["provider_hits"]:
        providers = ", ".join(provenance["provider_hits"])
        st.error(f"Final assessment: AI-generated — provider provenance hint detected ({providers}).")
    elif high_resolution_upload:
        st.warning("Final assessment: Uncertain — this model was trained on 32×32 CIFAKE images, so this high-resolution upload is outside its training domain.")
    elif provenance["camera_metadata"] and not provenance["provider_hits"] and fake_probability >= 0.80:
        st.warning("Final assessment: Conflict — camera metadata suggests a captured image, but the visual model predicts AI. Do not label this image AI-generated without further review.")
    elif fake_probability >= 0.80:
        st.error("Final assessment: AI-generated")
    elif fake_probability <= 0.20:
        st.success("Final assessment: Probably real")
    else:
        st.warning("Prediction: Uncertain")

    left, right = st.columns(2)
    left.metric("AI-generated probability", f"{fake_probability:.2%}")
    right.metric("Real probability", f"{real_probability:.2%}")
    st.progress(fake_probability, text="AI-generated probability")

    st.subheader("Provider and provenance hints")
    if provenance["provider_hits"]:
        st.warning(human_summary(provenance))
        for provider, hits in provenance["provider_hits"].items():
            st.write(f"**{provider}:** {', '.join(hits)}")
    elif provenance["c2pa_present"]:
        st.info(human_summary(provenance))
    elif provenance["generic_ai_terms"]:
        st.info(human_summary(provenance))
    else:
        st.caption(human_summary(provenance))

    with st.expander("Readable file metadata"):
        if provenance["metadata"]:
            st.json(provenance["metadata"])
        else:
            st.write("No readable metadata was found.")

    st.markdown(
        "For stronger provenance checks, verify the original file with "
        "[OpenAI Verify](https://openai.com/verify/) or review "
        "[Google SynthID](https://deepmind.google/models/synthid/) support."
    )
    st.info("This is a research model, not proof of authenticity. Performance depends on how similar the uploaded image is to the training data.")

