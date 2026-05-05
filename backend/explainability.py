from __future__ import annotations

from pathlib import Path
from typing import Dict
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image

from gradcam import GradCAM, make_heatmap, make_overlay


EXPLANATIONS_DIR = Path(__file__).resolve().parent / "static" / "explanations"


def _ensure_output_dir() -> Path:
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPLANATIONS_DIR


def _save_rgb_image(path: Path, image_array: np.ndarray) -> None:
    Image.fromarray(image_array).save(path, format="JPEG", quality=95)


def _compute_edges(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def generate_explainability_images(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    predicted_class: int,
) -> Dict[str, str]:
    output_dir = _ensure_output_dir()
    unique_prefix = uuid4().hex

    gradcam = GradCAM(model, model.conv_head)
    try:
        cam = gradcam.generate(input_tensor=input_tensor, class_index=predicted_class)
    finally:
        gradcam.remove_hooks()
    original_height, original_width = original_rgb.shape[:2]
    cam = cv2.resize(cam, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    heatmap = make_heatmap(cam)
    overlay = make_overlay(original_rgb, heatmap)
    edges = _compute_edges(original_rgb)

    original_path = output_dir / f"{unique_prefix}_original.jpg"
    edges_path = output_dir / f"{unique_prefix}_edges.jpg"
    heatmap_path = output_dir / f"{unique_prefix}_gradcam_heatmap.jpg"
    overlay_path = output_dir / f"{unique_prefix}_gradcam_overlay.jpg"

    _save_rgb_image(original_path, original_rgb)
    _save_rgb_image(edges_path, edges)
    _save_rgb_image(heatmap_path, heatmap)
    _save_rgb_image(overlay_path, overlay)

    return {
        "original.jpg": f"/static/explanations/{original_path.name}",
        "edges.jpg": f"/static/explanations/{edges_path.name}",
        "gradcam_heatmap.jpg": f"/static/explanations/{heatmap_path.name}",
        "gradcam_overlay.jpg": f"/static/explanations/{overlay_path.name}",
    }
