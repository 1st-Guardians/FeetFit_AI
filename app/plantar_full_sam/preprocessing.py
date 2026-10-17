"""Prepare plantar foot detection using the upload's original RGB values."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.plantar_full_sam.foot_roi import FootRoiResult


PREPROCESSING_POLICY = "fixed_quarter_turn_original_pixels_v4"
MAX_PREPROCESSING_ATTEMPTS = 1


@dataclass(frozen=True)
class PlantarFootInputCandidate:
    """One detector input with unchanged geometry and explicit color metadata."""

    image_rgb: np.ndarray
    method: str
    brightness_gain: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)
    rotation_degrees_clockwise: int = 90


@dataclass(frozen=True)
class PreparedPlantarFootInput:
    """An RGB input and ROI expressed in that input's pixel coordinates.

    Any detector-owned cardinal view is relative to ``image_rgb`` as well.
    Consumers must use this RGB image with the ROI instead of the upload.
    """

    image_rgb: np.ndarray
    roi: FootRoiResult
    rotation_degrees_clockwise: int = 0
    brightness_gain: float = 1.0
    audit: dict[str, Any] = field(default_factory=dict)


def iter_plantar_foot_inputs(
    image_rgb: np.ndarray,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Compatibility tuple for the original-pixel input used by ROI detection."""
    for candidate in iter_plantar_foot_candidates(image_rgb):
        yield (
            candidate.rotation_degrees_clockwise,
            candidate.brightness_gain,
            candidate.image_rgb,
        )


def iter_plantar_foot_candidates(
    image_rgb: np.ndarray,
) -> Iterator[PlantarFootInputCandidate]:
    """Yield one clockwise 90-degree turn without any photometric correction."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a uint8 HxWx3 numpy array")
    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or min(image_rgb.shape[:2]) == 0
    ):
        raise ValueError("image_rgb must be a nonempty uint8 HxWx3 RGB image")

    yield PlantarFootInputCandidate(
        image_rgb=cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE),
        method="original",
        parameters={"gain": 1.0},
    )


def prepare_plantar_foot_input(
    image_rgb: np.ndarray,
    *,
    extract: Callable[[np.ndarray], FootRoiResult],
) -> PreparedPlantarFootInput:
    """Rotate clockwise 90 degrees and detect once at original brightness.

    All source uploads use the camera orientation requiring this fixed turn.
    Failed detection retains the original-pixel ROI; model errors propagate.
    """
    candidate = next(iter_plantar_foot_candidates(image_rgb))
    roi = extract(candidate.image_rgb)
    selected = {
        "candidate_index": 0,
        "source": "preprocessed",
        "rotation_degrees_clockwise": 90,
        "brightness_gain": 1.0,
        "method": candidate.method,
        "parameters": dict(candidate.parameters),
        "found": bool(roi.found),
        "confidence": float(roi.confidence),
        "reason": str(getattr(roi, "reason", "")),
        "detection_pass": str(getattr(roi, "detection_pass", "original")),
    }

    audit = {
        "policy": PREPROCESSING_POLICY,
        "original_shape": [int(value) for value in image_rgb.shape],
        "coordinate_space": "prepared_image_rgb",
        "attempt_count": 1,
        "maximum_attempt_count": MAX_PREPROCESSING_ATTEMPTS,
        "attempts": [selected],
        "selected": dict(selected),
    }
    return PreparedPlantarFootInput(
        image_rgb=candidate.image_rgb,
        roi=roi,
        rotation_degrees_clockwise=90,
        brightness_gain=1.0,
        audit=audit,
    )
