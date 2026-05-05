from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (256, 256)


_preprocess_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_rgb_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = _preprocess_transform(image)
    return tensor.unsqueeze(0)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def load_and_preprocess(file_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
    image = load_rgb_image(file_bytes)
    tensor = preprocess_image(image)
    return image, tensor
