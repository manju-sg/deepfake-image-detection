"""Lightweight local provenance and provider-hint inspection.

This module does not claim to prove authorship. It looks for surviving
metadata/text markers such as C2PA, OpenAI, DALL-E, Gemini, and Imagen.
Those markers can disappear after screenshots, recompression, or conversion.
"""

from __future__ import annotations

import io
import re
from typing import Any

from PIL import Image


PROVIDER_MARKERS = {
    "OpenAI / ChatGPT / DALL-E": [
        "openai",
        "chatgpt",
        "dall-e",
        "dalle",
        "gpt-image",
        "gpt_image",
    ],
    "Google / Gemini / Imagen": [
        "google",
        "gemini",
        "imagen",
        "synthid",
    ],
}


def _flatten_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def inspect_provenance(file_bytes: bytes) -> dict[str, Any]:
    """Return provider hints and readable metadata from an image byte string."""

    haystack = file_bytes.lower()
    metadata: dict[str, str] = {}
    try:
        image = Image.open(io.BytesIO(file_bytes))
        for key, value in image.info.items():
            text = _flatten_text(value).strip()
            if text:
                metadata[str(key)] = text[:1000]
        for key, value in image.getexif().items():
            text = _flatten_text(value).strip()
            if text:
                metadata[f"EXIF:{key}"] = text[:1000]
    except Exception as exc:  # Keep prediction usable for unusual image files.
        metadata["reader_warning"] = str(exc)

    # ICC profiles and JFIF blocks often contain unrelated vendor names such
    # as "Google Inc.". Do not use those binary/color-management fields for
    # provider attribution.
    ignored_metadata_keys = {"icc_profile", "jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi"}
    semantic_metadata = "\n".join(
        value for key, value in metadata.items() if key.lower() not in ignored_metadata_keys
    ).lower()
    searchable_text = haystack + b"\n" + semantic_metadata.encode()
    marker_hits: dict[str, list[str]] = {}
    for provider, markers in PROVIDER_MARKERS.items():
        hits = [marker for marker in markers if marker in semantic_metadata]
        if hits:
            marker_hits[provider] = sorted(set(hits))

    c2pa_present = any(token in searchable_text for token in (b"c2pa", b"content credentials", b"jumbf"))
    camera_fields = {
        "EXIF:271",  # Make
        "EXIF:272",  # Model
        "EXIF:305",  # Software
        "EXIF:306",  # DateTime
        "EXIF:36867",  # DateTimeOriginal
    }
    camera_metadata = bool(camera_fields.intersection(metadata))
    generic_ai_terms = [
        term
        for term in ("ai-generated", "ai generated", "synthetic", "generative", "stable diffusion", "midjourney")
        if term in semantic_metadata
    ]

    if marker_hits:
        status = "provider_hint"
    elif camera_metadata:
        status = "camera_metadata"
    elif c2pa_present:
        status = "c2pa_present"
    elif generic_ai_terms:
        status = "generic_ai_hint"
    else:
        status = "no_marker_found"

    return {
        "status": status,
        "provider_hits": marker_hits,
        "c2pa_present": c2pa_present,
        "camera_metadata": camera_metadata,
        "generic_ai_terms": sorted(set(generic_ai_terms)),
        "metadata": metadata,
    }


def human_summary(result: dict[str, Any]) -> str:
    if result["provider_hits"]:
        providers = ", ".join(result["provider_hits"])
        return f"Provider marker found: {providers}. This is a provenance hint, not proof of an unedited file."
    if result["c2pa_present"]:
        return "C2PA/Content Credentials marker found, but the local scanner could not identify the issuer."
    if result["camera_metadata"]:
        return "Camera/gallery metadata found (for example make, model, software, or capture time). This supports a camera-captured origin, but metadata can be edited."
    if result["generic_ai_terms"]:
        return "Generic AI-generation metadata terms found; no specific provider was identified."
    return "No readable provider marker found. This does not prove the image is real or human-created."

