from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


CLASS_LABELS: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


MEDICAL_EXPLANATIONS: Dict[int, str] = {
    0: "No diabetic retinopathy is detected. The retina appears free from the typical visible lesions associated with diabetic damage.",
    1: "Mild diabetic retinopathy is suggested. Small retinal abnormalities such as microaneurysms may be present and should be monitored clinically.",
    2: "Moderate diabetic retinopathy is suggested. Retinal damage appears more established, and ophthalmology follow-up is recommended to assess progression risk.",
    3: "Severe diabetic retinopathy is suggested. Extensive retinal changes may be present, which can significantly increase the risk of vision-threatening complications.",
    4: "Proliferative diabetic retinopathy is suggested. Advanced disease may involve abnormal blood vessel growth and requires urgent specialist evaluation.",
}


@dataclass
class PredictionResult:
    class_index: int
    class_name: str
    confidence: float
    medical_explanation: str
    probabilities: torch.Tensor


def predict(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> PredictionResult:
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    class_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[class_index].item())
    class_name = CLASS_LABELS[class_index]
    medical_explanation = MEDICAL_EXPLANATIONS[class_index]

    return PredictionResult(
        class_index=class_index,
        class_name=class_name,
        confidence=confidence,
        medical_explanation=medical_explanation,
        probabilities=probabilities,
    )
