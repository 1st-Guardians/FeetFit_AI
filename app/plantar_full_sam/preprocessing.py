"""Try bounded photometric fallbacks only when plantar foot detection fails."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.plantar_full_sam.foot_roi import FootRoiResult


PREPROCESSING_POLICY = "fixed_quarter_turn_then_photometric_fallback_v3"
_BRIGHTNESS_GAINS = (1.0, 1.5, 2.0)
MAX_PREPROCESSING_ATTEMPTS = 8


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
    """Compatibility tuples for the same candidates used by ROI detection.

    Gain describes only multiplication; nonlinear methods are recorded by
    ``iter_plantar_foot_candidates`` and the prepared input's audit.
    """
    for candidate in iter_plantar_foot_candidates(image_rgb):
        yield (
            candidate.rotation_degrees_clockwise,
            candidate.brightness_gain,
            candidate.image_rgb,
        )


def _gamma_rgb(image_rgb: np.ndarray, exponent: float) -> np.ndarray:
    lut = np.rint(
        np.power(np.arange(256, dtype=np.float32) / 255.0, exponent) * 255.0
    ).astype(np.uint8)
    return cv2.LUT(image_rgb, lut)


def _clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def iter_plantar_foot_candidates(
    image_rgb: np.ndarray,
) -> Iterator[PlantarFootInputCandidate]:
    """Lazily yield up to eight color variants of one clockwise 90-degree turn.

    Every variant starts from original pixels, with gamma+CLAHE an explicit
    combined recipe. Successful detection ends iteration before later work is
    computed. No crop, resize, reflection, or further rotation is introduced.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a uint8 HxWx3 numpy array")
    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or min(image_rgb.shape[:2]) == 0
    ):
        raise ValueError("image_rgb must be a nonempty uint8 HxWx3 RGB image")

    rotated_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    for gain in _BRIGHTNESS_GAINS:
        candidate_image = (
            rotated_rgb
            if gain == 1.0
            else np.clip(rotated_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        )
        yield PlantarFootInputCandidate(
            image_rgb=candidate_image,
            method="original" if gain == 1.0 else f"gain_{gain:g}",
            brightness_gain=gain,
            parameters={"gain": gain},
        )

    for exponent in (0.65, 0.45):
        yield PlantarFootInputCandidate(
            image_rgb=_gamma_rgb(rotated_rgb, exponent),
            method=f"gamma_{exponent:g}",
            parameters={"gamma_exponent": exponent},
        )
    clahe_parameters = {"clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8]}
    yield PlantarFootInputCandidate(
        image_rgb=_clahe_rgb(rotated_rgb),
        method="clahe_luminance",
        parameters=dict(clahe_parameters),
    )
    yield PlantarFootInputCandidate(
        image_rgb=_clahe_rgb(_gamma_rgb(rotated_rgb, 0.65)),
        method="gamma_0.65_clahe_luminance",
        parameters={"gamma_exponent": 0.65, **clahe_parameters},
    )

    lab = cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2LAB)
    step = max(1, int(np.ceil(max(lab.shape[:2]) / 512)))
    low, high = np.percentile(lab[::step, ::step, 0], (2.0, 98.0))
    # Do not stretch tiny sensor noise in an almost uniform image to full range.
    if high - low >= 8.0:
        lab[..., 0] = np.rint(np.clip(
            (lab[..., 0].astype(np.float32) - float(low)) * (255.0 / float(high - low)),
            0, 255,
        )).astype(np.uint8)
        yield PlantarFootInputCandidate(
            image_rgb=cv2.cvtColor(lab, cv2.COLOR_LAB2RGB),
            method="percentile_luminance_stretch",
            parameters={
                "percentiles": [2.0, 98.0],
                "luminance_low": float(low),
                "luminance_high": float(high),
            },
        )


def prepare_plantar_foot_input(
    image_rgb: np.ndarray,
    *,
    extract: Callable[[np.ndarray], FootRoiResult],
) -> PreparedPlantarFootInput:
    """Rotate clockwise 90 degrees and prefer detection on original pixels.

    All source uploads use the camera orientation requiring this fixed turn.
    Try gain, gamma, local contrast and luminance-stretch fallbacks only after
    preceding candidates fail, with at most eight detector attempts.
    A brighter image can produce a more confident partial-foot mask, so it
    must not replace a successful original detection. Detector thresholds are
    unchanged; model errors propagate and all failures retain the first ROI.
    """
    attempts: list[dict[str, Any]] = []

    def record(roi: FootRoiResult, candidate: PlantarFootInputCandidate) -> dict[str, Any]:
        row = {
            "candidate_index": len(attempts),
            "source": "preprocessed",
            "rotation_degrees_clockwise": 90,
            "brightness_gain": float(candidate.brightness_gain),
            "method": candidate.method,
            "parameters": dict(candidate.parameters),
            "found": bool(roi.found),
            "confidence": float(roi.confidence),
            "reason": str(getattr(roi, "reason", "")),
            "detection_pass": str(getattr(roi, "detection_pass", "original")),
        }
        attempts.append(row)
        return row

    candidates = iter_plantar_foot_candidates(image_rgb)
    best_candidate = next(candidates)
    best_roi = extract(best_candidate.image_rgb)
    selected = record(best_roi, best_candidate)

    if not best_roi.found:
        for candidate in candidates:
            candidate_roi = extract(candidate.image_rgb)
            row = record(candidate_roi, candidate)
            if candidate_roi.found:
                best_candidate = candidate
                best_roi = candidate_roi
                selected = row
                break

    audit = {
        "policy": PREPROCESSING_POLICY,
        "original_shape": [int(value) for value in image_rgb.shape],
        "coordinate_space": "prepared_image_rgb",
        "attempt_count": len(attempts),
        "maximum_attempt_count": MAX_PREPROCESSING_ATTEMPTS,
        "attempts": attempts,
        "selected": dict(selected),
    }
    return PreparedPlantarFootInput(
        image_rgb=best_candidate.image_rgb,
        roi=best_roi,
        rotation_degrees_clockwise=90,
        brightness_gain=best_candidate.brightness_gain,
        audit=audit,
    )
