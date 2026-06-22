"""CNN patch classification and image preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class ImagePatchClassifier(nn.Module):
    """MobileNetV2 with a binary pedestrian/background output."""

    def __init__(self) -> None:
        super().__init__()
        self.model = mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one raw logit per image patch."""

        return self.model(x).reshape(-1)

    def predict_probabilities(self, batch: torch.Tensor) -> np.ndarray:
        """Return pedestrian probabilities for a tensor batch."""

        self.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self(batch))
        return probabilities.detach().cpu().numpy()


def load_patch_classifier(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> ImagePatchClassifier:
    """Load a binary MobileNetV2 checkpoint supplied by the caller."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    classifier = ImagePatchClassifier().to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def preprocess_patch(
    patch: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
    image_is_bgr: bool = True,
) -> torch.Tensor:
    """Resize and normalize one patch for a MobileNetV2 binary classifier."""

    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"patch must have shape (H, W, 3), got {patch.shape}")

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if image_is_bgr else patch
    resized = cv2.resize(patch_rgb, output_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float()


def extract_preprocessed_patches(
    image: np.ndarray,
    bboxes: Sequence,
    min_patch_height_px: int,
    min_patch_width_px: int,
) -> tuple[list[torch.Tensor], list[int], list[np.ndarray]]:
    """Crop valid image boxes and return normalized tensors and source indices."""

    image_height, image_width = image.shape[:2]
    tensors: list[torch.Tensor] = []
    box_indices: list[int] = []
    raw_patches: list[np.ndarray] = []

    for bbox_index, bbox in enumerate(bboxes):
        v_min = max(0, min(int(bbox.v), image_height - 1))
        u_min = max(0, min(int(bbox.u), image_width - 1))
        v_max = max(0, min(int(bbox.v + bbox.h), image_height))
        u_max = max(0, min(int(bbox.u + bbox.w), image_width))

        if v_max <= v_min or u_max <= u_min:
            continue
        if (v_max - v_min) < min_patch_height_px or (u_max - u_min) < min_patch_width_px:
            continue

        patch = image[v_min:v_max, u_min:u_max].copy()
        raw_patches.append(patch)
        tensors.append(preprocess_patch(patch))
        box_indices.append(bbox_index)

    return tensors, box_indices, raw_patches


def score_patches(
    classifier: ImagePatchClassifier,
    patch_tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Run the classifier on preprocessed patches."""

    if not patch_tensors:
        return np.zeros((0,), dtype=np.float32)

    batch = torch.stack(list(patch_tensors), dim=0).to(device)
    return classifier.predict_probabilities(batch)
