from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


NUM_CLASSES = 5
MODEL_NAME = "efficientnet_b3"
DEFAULT_MODEL_FILENAME = "best_model.pth"


def _candidate_model_paths(model_filename: str) -> list[Path]:
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    return [
        backend_dir / "saved_models" / model_filename,
        project_root / model_filename,
        backend_dir / "saved_models" / "final_dr_model.pth",
        project_root / "final_dr_model.pth",
    ]


def resolve_model_path(model_filename: str = DEFAULT_MODEL_FILENAME) -> Path:
    for path in _candidate_model_paths(model_filename):
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in _candidate_model_paths(model_filename))
    raise FileNotFoundError(f"Could not locate model weights. Searched:\n{searched}")


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(
    model_filename: str = DEFAULT_MODEL_FILENAME,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, torch.device, Path]:
    resolved_device = device or torch.device("cpu")
    model_path = resolve_model_path(model_filename)

    checkpoint = torch.load(model_path, map_location=resolved_device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }

    # Support checkpoints saved either with a plain Linear classifier or with
    # a Sequential(Dropout, Linear) classifier head.
    if "classifier.weight" in cleaned_state_dict and "classifier.bias" in cleaned_state_dict:
        cleaned_state_dict["classifier.1.weight"] = cleaned_state_dict.pop("classifier.weight")
        cleaned_state_dict["classifier.1.bias"] = cleaned_state_dict.pop("classifier.bias")

    model = build_model()
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device, model_path
