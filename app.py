"""Professional Streamlit interface for the deepfake image detector."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from provenance import human_summary, inspect_provenance


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink:#10233f; --muted:#63748b; --line:#dbe5f0; --teal:#0b8f9c; --teal-dark:#08717d; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        .stApp { background:linear-gradient(180deg,#f7faff 0%,#fff 38%,#f8fbfd 100%); color:var(--ink); }
        .block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#10233f; border-right:0; }
        [data-testid="stSidebar"] * { color:#e8f1fb; }
        [data-testid="stSidebar"] .stCaption { color:#a9bad0; }
        [data-testid="stFileUploader"] { background:#fff; border:1px dashed #9bb4cc; border-radius:18px; padding:.45rem; }
        [data-testid="stFileUploader"] section { border:0; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.75rem 1rem; box-shadow:0 6px 22px rgba(16,35,63,.04); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; }
        .brand { display:flex; align-items:center; gap:12px; margin-bottom:1.75rem; }
        .brand-mark { width:42px; height:42px; border-radius:13px; display:grid; place-items:center; color:#fff; background:linear-gradient(135deg,#0b8f9c,#163e6c); box-shadow:0 8px 20px rgba(11,143,156,.23); font-size:21px; font-weight:800; }
        .brand-name { font:800 1.18rem 'Manrope',sans-serif; letter-spacing:-.03em; color:var(--ink); }
        .brand-sub { font-size:.72rem; color:var(--muted); margin-top:1px; letter-spacing:.04em; text-transform:uppercase; }
        .hero { border-radius:26px; padding:2.25rem 2.4rem; margin-bottom:1.4rem; background:radial-gradient(circle at 90% 20%,rgba(71,190,193,.24),transparent 28%),linear-gradient(120deg,#10233f 0%,#163e6c 63%,#0b8f9c 140%); color:#fff; overflow:hidden; position:relative; }
        .hero:after { content:''; position:absolute; width:230px; height:230px; border-radius:50%; border:1px solid rgba(255,255,255,.12); right:7%; top:-80px; box-shadow:0 0 0 22px rgba(255,255,255,.035),0 0 0 44px rgba(255,255,255,.025); }
        .eyebrow { font-size:.72rem; letter-spacing:.17em; font-weight:700; opacity:.74; text-transform:uppercase; }
        .hero h1 { color:#fff; font:800 clamp(2rem,4vw,3.25rem) 'Manrope',sans-serif; letter-spacing:-.055em; line-height:1.04; margin:.55rem 0 .9rem; max-width:680px; }
        .hero p { color:#d7e6f7; font-size:1.03rem; line-height:1.65; max-width:650px; margin:0; }
        .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:1.35rem; }
        .pill { display:inline-flex; align-items:center; gap:6px; padding:6px 11px; border:1px solid rgba(255,255,255,.2); border-radius:99px; color:#e5f4fb; font-size:.78rem; background:rgba(255,255,255,.08); }
        .section-title { color:var(--ink); font:700 1.15rem 'Manrope',sans-serif; letter-spacing:-.025em; margin:.2rem 0 .75rem; }
        .panel { background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.25rem; box-shadow:0 9px 30px rgba(16,35,63,.045); }
        .panel-title { color:var(--ink); font:700 1rem 'Manrope',sans-serif; margin-bottom:.25rem; }
        .panel-copy { color:var(--muted); line-height:1.55; font-size:.9rem; }
        .step { display:flex; gap:11px; margin:.85rem 0; }
        .step-number { flex:0 0 28px; height:28px; border-radius:9px; display:grid; place-items:center; background:#e8f6f7; color:var(--teal-dark); font-weight:700; font-size:.8rem; }
        .step-text { color:var(--ink); font-size:.86rem; line-height:1.45; }
        .step-text span { color:var(--muted); display:block; font-size:.78rem; margin-top:2px; }
        .result-card { border-radius:22px; padding:1.4rem 1.5rem; border:1px solid var(--line); background:#fff; box-shadow:0 12px 32px rgba(16,35,63,.06); margin:1.15rem 0; }
        .result-card.danger { border-color:#f1caca; background:linear-gradient(135deg,#fff 0%,#fff7f7 100%); }
        .result-card.success { border-color:#bde7d7; background:linear-gradient(135deg,#fff 0%,#f5fffb 100%); }
        .result-card.warning { border-color:#f1dcad; background:linear-gradient(135deg,#fff 0%,#fffaf0 100%); }
        .result-kicker { color:var(--muted); font-size:.76rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; }
        .result-title { font:800 1.55rem 'Manrope',sans-serif; letter-spacing:-.04em; color:var(--ink); margin:.25rem 0 .35rem; }
        .result-copy { color:var(--muted); line-height:1.5; font-size:.9rem; margin:0; }
        .score-label { display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem; margin:.8rem 0 .35rem; }
        .score-track { height:10px; border-radius:999px; background:#eaf0f6; overflow:hidden; }
        .score-fill { height:100%; border-radius:inherit; background:linear-gradient(90deg,#0b8f9c,#c64545); }
        .signal { display:inline-flex; align-items:center; gap:7px; border-radius:99px; padding:5px 10px; background:#f2f6fa; color:var(--muted); font-size:.77rem; margin-top:.8rem; }
        .signal-dot { width:7px; height:7px; border-radius:50%; background:var(--teal); }
        .info-box { background:#f5f8fc; border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem; color:var(--muted); font-size:.84rem; line-height:1.55; }
        .file-chip { display:inline-block; color:#35516f; background:#edf4fb; border-radius:8px; padding:5px 9px; margin:3px 4px 0 0; font-size:.76rem; }
        .footer { color:#8797aa; text-align:center; font-size:.75rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:2.5rem; }
        div[data-testid="stTabs"] button { color:var(--muted); }
        div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--teal-dark); }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    return model, class_names, image_transform, image_size


def result_details(fake_probability: float, provenance: dict[str, Any], high_resolution: bool) -> tuple[str, str, str]:
    if provenance["provider_hits"]:
        return "AI-generated signal", "danger", "A provider-related marker was found in the file. Treat it as a provenance hint and verify the original source separately."
    if high_resolution:
        return "Uncertain", "warning", "This image is outside the visual resolution range used to train the current checkpoint."
    if provenance["camera_metadata"] and fake_probability >= 0.80:
        return "Review recommended", "warning", "Camera or gallery metadata conflicts with the visual model score. Do not make a final authenticity claim from this result alone."
    if fake_probability >= 0.80:
        return "Likely AI-generated", "danger", "The visual model detected patterns associated with its AI-generated training examples."
    if fake_probability <= 0.20:
        return "Likely real", "success", "The visual model found stronger similarity to its real-image training examples."
    return "Uncertain", "warning", "The model probabilities are close enough that a confident decision is not recommended."


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div style='font:800 1.2rem Manrope,sans-serif; margin-bottom:1.5rem;'>TrueLens</div>", unsafe_allow_html=True)
        st.markdown("### Analysis profile")
        st.caption("A visual classifier with a lightweight provenance scan.")
        st.markdown("<div class='file-chip'>ResNet-18</div><div class='file-chip'>CIFAKE-trained</div><div class='file-chip'>Local inference</div>", unsafe_allow_html=True)
        st.markdown("### Runtime")
        st.caption(f"Accelerator: {DEVICE}")
        st.caption("Model input: 128 × 128 pixels")
        st.caption("Supported: JPG, PNG, WebP")
        st.markdown("### Important note")
        st.caption("Predictions are estimates, not proof of origin. Results are most reliable when the image resembles the training data.")
        st.markdown("---")
        st.caption("TrueLens · Image authenticity intelligence")


def render_provenance(provenance: dict[str, Any]) -> None:
    st.markdown("<div class='section-title'>Provenance signals</div>", unsafe_allow_html=True)
    if provenance["provider_hits"]:
        st.error(human_summary(provenance))
        for provider, hits in provenance["provider_hits"].items():
            st.markdown(f"**{provider}**  ")
            st.markdown(" ".join(f"`{hit}`" for hit in hits))
    elif provenance["c2pa_present"]:
        st.info(human_summary(provenance))
    elif provenance["generic_ai_terms"]:
        st.info(human_summary(provenance))
    elif provenance["camera_metadata"]:
        st.success(human_summary(provenance))
    else:
        st.caption(human_summary(provenance))


st.set_page_config(page_title="TrueLens | Image Authenticity", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
inject_styles()
render_sidebar()

st.markdown(
    """
    <div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">TrueLens</div><div class="brand-sub">Image authenticity intelligence</div></div></div>
    <section class="hero"><div class="eyebrow">Visual forensics · AI image analysis</div><h1>See the signals behind an image.</h1><p>Upload an image to receive a deep-learning assessment, probability scores, and a separate scan for readable provenance markers.</p><div class="pill-row"><span class="pill">✦ Deep-learning classifier</span><span class="pill">⌁ Provenance-aware</span><span class="pill">▣ Private local processing</span></div></section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The detector checkpoint is missing. Run the training command before starting the app.")
    st.stop()

model, class_names, image_transform, image_size = load_detector()
upload_col, method_col = st.columns([1.28, 0.72], gap="large")
with upload_col:
    st.markdown("<div class='section-title'>Start an analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-copy' style='margin-bottom:.7rem;'>Drop a single image here. The detector works best with a clear, unedited image.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with method_col:
    st.markdown("<div class='panel'><div class='panel-title'>How the analysis works</div><div class='step'><div class='step-number'>01</div><div class='step-text'>Visual encoding<span>ResNet-18 extracts image features.</span></div></div><div class='step'><div class='step-number'>02</div><div class='step-text'>Probability scoring<span>Two-class model estimates real vs synthetic.</span></div></div><div class='step'><div class='step-number'>03</div><div class='step-text'>Context scan<span>Readable EXIF, XMP, C2PA, and provider hints are checked separately.</span></div></div></div>", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("<div class='info-box' style='margin-top:1.35rem;'><strong>Good to know</strong><br>A probability is a model estimate, not a certificate of origin. Screenshots, recompression, metadata removal, and unfamiliar generators can reduce reliability.</div>", unsafe_allow_html=True)
else:
    try:
        file_bytes = uploaded.getvalue()
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"This file could not be read as an image: {exc}")
        st.stop()

    tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    fake_probability = probabilities[class_names.index("fake")].item()
    real_probability = probabilities[class_names.index("real")].item()
    provenance = inspect_provenance(file_bytes)
    high_resolution_upload = max(image.size) > 256
    title, tone, explanation = result_details(fake_probability, provenance, high_resolution_upload)

    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Analysis result</div>", unsafe_allow_html=True)
    preview_col, result_col = st.columns([0.9, 1.1], gap="large")
    with preview_col:
        st.image(image, use_container_width=True)
        safe_name = html.escape(uploaded.name)
        st.markdown(f"<span class='file-chip'>{safe_name}</span><span class='file-chip'>{image.size[0]} × {image.size[1]} px</span><span class='file-chip'>{len(file_bytes)/1024:.1f} KB</span>", unsafe_allow_html=True)
    with result_col:
        st.markdown(f"<div class='result-card {tone}'><div class='result-kicker'>Final assessment</div><div class='result-title'>{title}</div><p class='result-copy'>{explanation}</p><div class='signal'><span class='signal-dot'></span>Model evaluated locally on {DEVICE}</div></div>", unsafe_allow_html=True)
        metric_left, metric_right = st.columns(2)
        metric_left.metric("AI-generated probability", f"{fake_probability:.2%}")
        metric_right.metric("Real probability", f"{real_probability:.2%}")
        st.markdown(f"<div class='score-label'><span>AI-generated score</span><strong>{fake_probability:.2%}</strong></div><div class='score-track'><div class='score-fill' style='width:{fake_probability * 100:.2f}%'></div></div>", unsafe_allow_html=True)

    tab_overview, tab_provenance, tab_metadata = st.tabs(["Overview", "Provenance", "Readable metadata"])
    with tab_overview:
        st.markdown("<div class='panel' style='margin-top:1rem;'><div class='panel-title'>What this result means</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='panel-copy'>{explanation} The visual model was trained on {image_size} × {image_size} inputs, so image dimensions and editing history matter when interpreting the score.</div></div>", unsafe_allow_html=True)
    with tab_provenance:
        st.markdown("<div class='panel' style='margin-top:1rem;'>", unsafe_allow_html=True)
        render_provenance(provenance)
        st.markdown("</div>", unsafe_allow_html=True)
    with tab_metadata:
        st.markdown("<div class='panel' style='margin-top:1rem;'>", unsafe_allow_html=True)
        if provenance["metadata"]:
            st.json(provenance["metadata"])
        else:
            st.caption("No readable metadata was found in this file.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-box' style='margin-top:1rem;'>For stronger provenance checks, verify the original file with <a href='https://openai.com/verify/' target='_blank'>OpenAI Verify</a> or review <a href='https://deepmind.google/models/synthid/' target='_blank'>Google SynthID</a> support. A missing marker does not prove an image is real.</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>TrueLens · Visual authenticity analysis · Results should be reviewed with source context.</div>", unsafe_allow_html=True)
