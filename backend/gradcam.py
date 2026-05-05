from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def generate(self, input_tensor: torch.Tensor, class_index: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)
        if class_index is None:
            class_index = int(torch.argmax(outputs, dim=1).item())

        score = outputs[:, class_index]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value
        return cam


def make_heatmap(cam: np.ndarray) -> np.ndarray:
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def make_overlay(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    base = rgb_image.astype(np.float32)
    cam = heatmap.astype(np.float32)
    overlay = cv2.addWeighted(base, 1.0 - alpha, cam, alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)
