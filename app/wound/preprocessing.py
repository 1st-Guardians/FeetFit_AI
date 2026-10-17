"""Photometric preprocessing shared by wound inference and color experiments."""
from __future__ import annotations

import cv2
import numpy as np

REDNESS_POLICY = "positive_lab_a_scale_preserve_L_and_b_v1"


def reduce_redness(rgb: np.ndarray, method: str, amount: float) -> np.ndarray:
    """Modify positive Lab a* while preserving requested L* and b* values."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB")
    if method == "none":
        return rgb.copy()
    if method not in {"scale", "shift"} or not np.isfinite(amount) or amount < 0:
        raise ValueError("Invalid redness adjustment")
    if method == "scale" and amount > 1:
        raise ValueError("Redness reduction must be in [0, 1]")
    if amount == 0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    original_a = lab[..., 1].copy()
    positive = original_a > 0
    if method == "scale":
        lab[..., 1][positive] *= 1 - amount
    else:
        lab[..., 1][positive] = np.maximum(0, original_a[positive] - amount)
    output = np.rint(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
    output[~positive] = rgb[~positive]
    return output
