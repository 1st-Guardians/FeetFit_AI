"""Local plantar-contact segmentation with YOLO and fine-tuned SegFormer-B0.

The production path is intentionally isolated from every GPT/OpenAI and tinea
analysis path. Hardware photos are rotated 90 degrees clockwise before YOLO
foot detection. Detection, contact inputs, and displays preserve the original
sole RGB values on both feet without brightness or contrast correction.
SegFormer-B0 predicts the original and horizontally flipped crops; only their
restored binary intersection is
retained inside the YOLO silhouette. Saved displays preserve the same fixed
capture orientation on both sides; detector search rotations are undone and
silhouette heuristics cannot flip the feet. The legacy V4 decoder/SAM helpers
remain below for source compatibility, but the production analyzer neither
loads nor calls them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from app.core.config import settings
from app.core.weights import weights
from app.plantar_full_sam.foot_roi import (
    FootRoiConfig,
    FootRoiResult,
    YoloFootRoiDetector,
    bbox_from_mask,
    constrain_pressure_logits,
    support_mask_to_low_res,
)
from app.plantar_full_sam.segformer import SegFormerB0PressureSegmenter
from app.plantar_full_sam.preprocessing import (
    PreparedPlantarFootInput,
    prepare_plantar_foot_input,
)
from app.plantar_full_sam.full_sam_cleanup import (
    FullSamCleanupConfig,
    cleanup_full_sam_mask,
)
from app.plantar_full_sam.full_sam_smoothing import (
    FullSamSmoothingConfig,
    smooth_full_sam_mask,
)
from app.plantar_full_sam.sam_refiner import (
    MaskCandidate,
    SamRefinementResult,
    generate_candidate_masks,
    refine_with_sam,
)
from app.plantar_full_sam.sole_alpha_cleanup import cleanup_sole_alpha
from app.plantar_full_sam.sole_alpha_refiner import refine_sole_alpha_with_sam
from app.services.tinea_analysis import TineaAnalysisResult


PIPELINE_VERSION = "local-segformer-b0-contact-v2"
progress_logger = logging.getLogger("uvicorn.error")
CAPTURE_ORIENTATION_POLICY = "preserve_preprocessed_capture_orientation_v1"
CHECKPOINT_FORMAT = "feetfit_plantar_full_sam_decoder_v1"
SELECTION_POLICY = (
    "sam_candidate_numeric_rank_v3_conservative_hysteresis_intersection_s35"
)
CONSERVATIVE_METHODS = frozenset({"base", "morphology_open"})
CONSERVATIVE_REFINED_RATIO_HEADROOM = 0.05
CONSERVATIVE_MAX_EXPANSION = 1.12
COLOR_FALLBACK_RANK_TOLERANCE = 0.01
MAX_DECODED_PIXELS = 40_000_000

# Match the tinea suspicion map's pastel blue family while keeping the
# photographed pressure/contact evidence visible through the annotation.
# This module renders RGB arrays (the tinea renderer's equivalent BGR value is
# ``(255, 220, 155)``).
CONTACT_OVERLAY_COLOR_RGB = (155, 220, 255)
CONTACT_OVERLAY_ALPHA = 0.50
CONTACT_OUTLINE_COLOR_RGB = (74, 174, 232)
CONTACT_OUTLINE_ALPHA = 0.82
CONTACT_OUTLINE_WIDTH_REFERENCE_PIXELS = 256.0
CONTACT_OUTLINE_MIN_WIDTH_PIXELS = 2
CONTACT_OUTLINE_MAX_WIDTH_PIXELS = 4
CONTACT_OUTLINE_SMOOTH_SIGMA_PIXELS = 0.8

# Constants below this block support retired compatibility helpers. The active
# SegFormer path uses one sigmoid contact channel and the configured 0.5
# threshold; it does not use the tinea classes or the legacy cleanup pipeline.
CONTACT_CLASS_INDEX = 1
CONTACT_HARD_THRESHOLD = 0.78
CONTACT_HYSTERESIS_HIGH_THRESHOLD = 0.76
CONTACT_HYSTERESIS_LOW_THRESHOLD = 0.58
CONTACT_HYSTERESIS_TRIGGER_RATIO = 0.08
CONTACT_COMPONENT_MIN_PIXELS = 96
CONTACT_COMPONENT_MIN_RATIO = 0.0005
CONTACT_FINAL_COMPONENT_MIN_RATIO = 0.0015
CONTACT_GLARE_COMPONENT_MAX_OVERLAP = 0.28


class PlantarFullSamAnalysisError(RuntimeError):
    """A fail-closed local inference or rendering error."""


@dataclass(frozen=True)
class OrientationResult:
    image_rgb: np.ndarray
    sole_mask: np.ndarray
    source_alpha: np.ndarray
    rotation_degrees: float
    toe_end_margin: float
    confidence: float
    operations: tuple[str, ...]
    mirrored: bool = False


@dataclass(frozen=True)
class PlantarFullSamResult:
    side: str
    visualization_path: Path
    tinea_input_path: Path
    contact_mask_path: Path
    sole_mask_path: Path
    audit_path: Path
    region_count: int
    contact_area_ratio: float
    yolo_confidence: float
    orientation_confidence: float
    contact_probability_path: Path | None = None
    contact_model_input_path: Path | None = None
    # Kept only for route/test source compatibility.  Contact-only class-1
    # inference must never be reused as a plantar pathology report.
    tinea_analysis_result: TineaAnalysisResult | None = None


@dataclass(frozen=True)
class _PreparedPlantarCanvas:
    side: str
    clean_rgba: np.ndarray
    tinea_input_rgb: np.ndarray
    sole_canvas: np.ndarray
    contact_model_input_rgb: np.ndarray
    contact_model_sole: np.ndarray
    contact_source_shape: tuple[int, int]
    contact_source_bbox: tuple[int, int, int, int]
    contact_display_rotation_degrees_clockwise: int
    contact_display_crop_bbox: tuple[int, int, int, int]
    audit_base: dict[str, Any]
    yolo_confidence: float
    orientation_confidence: float
    # Original photographed sole used as the background for contact overlays.
    contact_display_rgba: np.ndarray


@dataclass(frozen=True)
class _PressedContactPrediction:
    """Dedicated SegFormer contact output restored to the plantar canvas."""

    mask: np.ndarray
    probability: np.ndarray
    audit: dict[str, Any]
    model_input_rgb: np.ndarray | None = None
    model_input_bbox: tuple[int, int, int, int] | None = None


def _prepare_contact_model_input(
    original_crop: np.ndarray,
    sole_mask: np.ndarray,
    side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve original sole pixels and whiten only the background."""
    sole = np.asarray(sole_mask) > 0
    source = np.asarray(original_crop)
    if (
        side not in {"left", "right"}
        or source.dtype != np.uint8
        or source.ndim != 3
        or source.shape[2] != 3
        or source.shape[:2] != sole.shape
        or not bool(sole.any())
    ):
        raise PlantarFullSamAnalysisError("Invalid original plantar contact crop.")
    model_input = np.full(source.shape, 255, dtype=np.uint8)
    model_input[sole] = source[sole]
    return model_input, {
        "policy": "original_rgb_outside_sole_white_v2",
        "source": "original_upload_rotated_crop",
        "side": side,
        "brightness_gain": 1.0,
        "contrast_factor": 1.0,
        "outside_sole_rgb": [255, 255, 255],
        "any_channel_255_ratio_in_sole": float(
            np.any(model_input[sole] == 255, axis=1).mean()
        ),
    }


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
    digest = sha256()
    digest.update(np.asarray(binary.shape, dtype=np.int64).tobytes())
    digest.update(binary.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not image_bytes:
        raise PlantarFullSamAnalysisError("Uploaded image is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            source_format = opened.format or "unknown"
            source_size = tuple(int(value) for value in opened.size)
            exif_orientation = int(opened.getexif().get(274, 1) or 1)
            transposed = ImageOps.exif_transpose(opened)
            exif_applied = exif_orientation != 1
            width, height = (int(value) for value in transposed.size)
            if width <= 0 or height <= 0 or width * height > MAX_DECODED_PIXELS:
                raise PlantarFullSamAnalysisError(
                    f"Decoded image dimensions are unsafe: {width}x{height}."
                )
            rgba = np.asarray(transposed.convert("RGBA"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, PlantarFullSamAnalysisError):
            raise
        raise PlantarFullSamAnalysisError(
            "Uploaded file is not a supported decodable image."
        ) from exc

    alpha = np.ascontiguousarray(rgba[..., 3])
    opacity = alpha.astype(np.float32)[..., None] / 255.0
    # Composite hidden transparent RGB onto white before YOLO/SAM.  The source
    # alpha is carried separately and can only reduce the published alpha.
    rgb = np.rint(rgba[..., :3].astype(np.float32) * opacity + 255.0 * (1.0 - opacity))
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return rgb, alpha, {
        "format": source_format,
        "width": width,
        "height": height,
        "exif_applied": bool(exif_applied),
        "input_alpha_present": bool(np.any(alpha < 255)),
    }


def _bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int]:
    binary = np.asarray(mask) > 0
    ys, xs = np.where(binary)
    if not len(xs):
        raise PlantarFullSamAnalysisError("The plantar silhouette is empty.")
    height, width = binary.shape
    pad = max(0, int(padding))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + pad + 1),
        min(height, int(ys.max()) + pad + 1),
    )


def _warp_bound(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    source_alpha: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    output_width = max(1, int(math.ceil(height * sine + width * cosine)))
    output_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    rotated_image = cv2.warpAffine(
        image_rgb,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        (np.asarray(mask) > 0).astype(np.uint8),
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    rotated_alpha = cv2.warpAffine(
        np.asarray(source_alpha, dtype=np.uint8),
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask, rotated_alpha


def _toe_end_margin(tight_mask: np.ndarray) -> float:
    """Return a positive value when the wider, lobed toe end is at the top."""

    binary = np.asarray(tight_mask > 0, dtype=np.uint8)
    height, width = binary.shape
    if height < 8 or width < 3:
        return 0.0
    spans = np.zeros(height, np.float32)
    mass = binary.sum(axis=1).astype(np.float32)
    runs = np.zeros(height, np.float32)
    for row_index, row in enumerate(binary):
        columns = np.flatnonzero(row)
        if len(columns):
            spans[row_index] = float(columns[-1] - columns[0] + 1)
        padded = np.pad(row, (1, 0), constant_values=0)
        runs[row_index] = float(
            np.count_nonzero(np.diff(padded.astype(np.int8)) == 1)
        )
    diameter = 2.0 * cv2.distanceTransform(binary, cv2.DIST_L2, 5).max(axis=1)
    sigma = max(1.0, height * 0.015)
    kernel = max(3, int(round(sigma * 6.0)) | 1)

    def normalized(profile: np.ndarray) -> np.ndarray:
        smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1),
            (1, kernel),
            sigmaX=0.0,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        maximum = float(smooth.max())
        return smooth / maximum if maximum > 0.0 else smooth

    profile = (
        0.45 * normalized(spans)
        + 0.25 * normalized(mass)
        + 0.30 * normalized(diameter)
    )

    def percentile(start: float, end: float, values: np.ndarray) -> float:
        begin = int(np.clip(round(height * start), 0, height - 1))
        finish = int(np.clip(round(height * end), begin + 1, height))
        return float(np.percentile(values[begin:finish], 90))

    top = percentile(0.08, 0.38, profile)
    bottom = percentile(0.62, 0.92, profile)
    threshold = float(np.percentile(profile, 90))
    peaks = np.flatnonzero(profile >= threshold)
    peak_position = (
        float(np.average(peaks, weights=profile[peaks]) / max(1, height - 1))
        if len(peaks)
        else 0.5
    )
    top_runs = percentile(0.00, 0.22, runs)
    bottom_runs = percentile(0.78, 1.00, runs)
    return float(
        (top - bottom)
        + 0.35 * (0.5 - peak_position) * 2.0
        + 0.15 * np.clip((top_runs - bottom_runs) / 3.0, -1.0, 1.0)
    )


def _canonicalize_sole(
    image_rgb: np.ndarray,
    sole_mask: np.ndarray,
    source_alpha: np.ndarray,
) -> OrientationResult:
    mask = np.asarray(sole_mask) > 0
    ys, xs = np.where(mask)
    if len(xs) < 64:
        raise PlantarFullSamAnalysisError("The detected plantar silhouette is too small.")
    coordinates = np.column_stack((xs, ys)).astype(np.float32)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(
        coordinates,
        mean=None,
        maxComponents=2,
    )
    axis_x, axis_y = eigenvectors[0]
    elongation = math.sqrt(
        float(eigenvalues[0, 0]) / max(float(eigenvalues[1, 0]), 1e-6)
    )
    if not 1.15 <= elongation <= 6.0:
        raise PlantarFullSamAnalysisError(
            f"Detected shape is not a complete plantar-facing sole (elongation={elongation:.3f})."
        )

    long_axis_angle = float(np.degrees(np.arctan2(axis_y, axis_x)))
    rotation = (long_axis_angle % 180.0) - 90.0
    image, rotated_mask, alpha = _warp_bound(
        image_rgb,
        mask,
        source_alpha,
        rotation,
    )
    operations = [f"rotate_bound:{rotation:.6f}"]

    x1, y1, x2, y2 = _bbox(rotated_mask)
    tight = rotated_mask[y1:y2, x1:x2]
    if tight.shape[1] >= tight.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_mask = cv2.rotate(
            rotated_mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE
        ) > 0
        alpha = cv2.rotate(alpha, cv2.ROTATE_90_CLOCKWISE)
        rotation += 90.0
        operations.append("rotate_90_clockwise")
        x1, y1, x2, y2 = _bbox(rotated_mask)
        tight = rotated_mask[y1:y2, x1:x2]

    margin = _toe_end_margin(tight)
    if margin < 0.0:
        image = cv2.rotate(image, cv2.ROTATE_180)
        rotated_mask = cv2.rotate(
            rotated_mask.astype(np.uint8), cv2.ROTATE_180
        ) > 0
        alpha = cv2.rotate(alpha, cv2.ROTATE_180)
        rotation += 180.0
        operations.append("rotate_180_toes_up")
        margin = -margin

    confidence = float(np.clip(abs(margin) / 0.25, 0.0, 1.0))
    # Preserve source transparency: a pre-transparent pixel can never become
    # visible merely because YOLO included it in the silhouette.
    alpha = np.where(rotated_mask, alpha, 0).astype(np.uint8)
    return OrientationResult(
        image_rgb=np.ascontiguousarray(image),
        sole_mask=np.ascontiguousarray(rotated_mask),
        source_alpha=np.ascontiguousarray(alpha),
        rotation_degrees=float(((rotation + 180.0) % 360.0) - 180.0),
        toe_end_margin=float(margin),
        confidence=confidence,
        operations=tuple(operations),
        mirrored=False,
    )


def _tight_crop(
    orientation: OrientationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    y, x = np.where(orientation.sole_mask)
    span = max(int(np.ptp(x)) + 1, int(np.ptp(y)) + 1)
    padding = max(4, int(round(span * 0.025)))
    x1, y1, x2, y2 = _bbox(orientation.sole_mask, padding)
    return (
        np.ascontiguousarray(orientation.image_rgb[y1:y2, x1:x2]),
        np.ascontiguousarray(orientation.sole_mask[y1:y2, x1:x2]),
        np.ascontiguousarray(orientation.source_alpha[y1:y2, x1:x2]),
        (x1, y1, x2, y2),
    )


def _require_unambiguous_orientation(orientation: OrientationResult) -> float:
    minimum = float(settings.plantar_min_orientation_confidence)
    if orientation.confidence < minimum:
        raise PlantarFullSamAnalysisError(
            "Toe/heel orientation is ambiguous "
            f"(confidence={orientation.confidence:.3f}, minimum={minimum:.3f})."
        )
    return minimum


def _rotate_quarter_turn_clockwise(
    values: np.ndarray,
    degrees_clockwise: int,
) -> np.ndarray:
    """Apply an exact cardinal rotation without interpolation or mirroring."""

    degrees = int(degrees_clockwise) % 360
    if degrees == 0:
        return np.ascontiguousarray(values)
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if degrees not in codes:
        raise PlantarFullSamAnalysisError(
            f"Detector view rotation must be cardinal, got {degrees_clockwise}."
        )
    return np.ascontiguousarray(cv2.rotate(values, codes[degrees]))


def _canonical_cardinal_crop(
    roi: FootRoiResult,
    source_alpha: np.ndarray,
) -> tuple[OrientationResult, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Restore the detector view to the fixed hardware capture orientation.

    Preprocessing already rotates both uploads 90 degrees clockwise. YOLO may
    select another cardinal view to find the foot, but that search angle is
    not a display instruction. Undo it for RGB and mask, matching the incoming
    preprocessed alpha. A wide or clipped heel must never trigger another
    180-degree turn based on the silhouette's toe/heel score.
    """

    if roi.cardinal_view_image is None or roi.cardinal_view_mask is None:
        raise PlantarFullSamAnalysisError(
            "YOLO did not return the selected cardinal detector view."
        )
    degrees = int(roi.cardinal_rotation_degrees_clockwise) % 360
    image = _rotate_quarter_turn_clockwise(roi.cardinal_view_image, -degrees)
    mask = _rotate_quarter_turn_clockwise(
        np.asarray(roi.cardinal_view_mask, dtype=np.uint8), -degrees,
    ) > 0
    alpha = np.ascontiguousarray(source_alpha)
    if image.shape[:2] != mask.shape or alpha.shape != mask.shape:
        raise PlantarFullSamAnalysisError(
            "Restored capture image, mask, and alpha are misaligned."
        )
    if not np.array_equal(mask, np.asarray(roi.mask) > 0):
        raise PlantarFullSamAnalysisError(
            "Restored detector mask does not match the preprocessed capture."
        )

    x1, y1, x2, y2 = _bbox(mask)
    operations = [
        f"restore_detector_view_by_{(-degrees) % 360}_clockwise",
        "preserve_preprocessed_capture_orientation",
    ]
    # Diagnostic only: orientation comes from the known capture convention.
    margin = _toe_end_margin(mask[y1:y2, x1:x2])
    alpha = np.where(mask, alpha, 0).astype(np.uint8)
    orientation = OrientationResult(
        image_rgb=np.ascontiguousarray(image),
        sole_mask=np.ascontiguousarray(mask),
        source_alpha=np.ascontiguousarray(alpha),
        rotation_degrees=0.0,
        toe_end_margin=float(margin),
        confidence=1.0,
        operations=tuple(operations),
        mirrored=False,
    )
    bbox = bbox_from_mask(
        mask,
        padding=int(settings.plantar_foot_roi_padding),
        padding_ratio=float(settings.plantar_foot_roi_padding_ratio),
    )
    x1, y1, x2, y2 = bbox
    crop_image = np.ascontiguousarray(image[y1:y2, x1:x2])
    crop_sole = np.ascontiguousarray(mask[y1:y2, x1:x2])
    crop_alpha = np.ascontiguousarray(alpha[y1:y2, x1:x2])
    if not bool(crop_sole.any()):
        raise PlantarFullSamAnalysisError("The padded cardinal plantar crop is empty.")
    return orientation, crop_image, crop_sole, crop_alpha, bbox


def _candidate_evaluation(
    candidate: MaskCandidate,
    result: SamRefinementResult,
    *,
    minimum_sam_score: float,
    maximum_contact_ratio: float,
) -> dict[str, Any]:
    qc = result.qc
    row = result.scores[result.selected_index] if result.selected_index >= 0 else None
    reasons: list[str] = []
    if result.selected_index < 0:
        reasons.append("sam_fallback_to_candidate")
    if float(result.selected_score) < minimum_sam_score:
        reasons.append("sam_score_below_minimum")
    if float(qc.area_ratio) < 0.02:
        reasons.append("area_ratio_below_minimum")
    candidate_area_ratio = float(candidate.area_ratio)
    refined_area_ratio = float(qc.area_ratio)
    expansion = refined_area_ratio / max(candidate_area_ratio, 1e-8)
    is_conservative = candidate.method in CONSERVATIVE_METHODS
    if is_conservative:
        if candidate_area_ratio > maximum_contact_ratio:
            reasons.append("candidate_area_ratio_above_maximum")
        if refined_area_ratio > (
            maximum_contact_ratio + CONSERVATIVE_REFINED_RATIO_HEADROOM
        ):
            reasons.append("refined_area_ratio_above_conservative_headroom")
        if expansion > CONSERVATIVE_MAX_EXPANSION:
            reasons.append("conservative_refinement_expansion_above_maximum")
    elif refined_area_ratio > maximum_contact_ratio:
        reasons.append("area_ratio_above_maximum")
    if float(qc.outside_support_ratio) > 0.0:
        reasons.append("pixels_outside_support")
    if float(qc.positive_prompt_recall) < 1.0:
        reasons.append("positive_prompt_missed")

    candidate_iou = 0.0 if row is None else float(row.candidate_iou)
    area_similarity = 0.0 if row is None else float(row.area_similarity)
    component_quality = min(1.0, 12.0 / max(float(qc.component_count), 1.0))
    hole_quality = max(0.0, 1.0 - float(qc.hole_area_ratio) / 0.12)
    rank = (
        0.52 * float(result.selected_score)
        + 0.12 * candidate_iou
        + 0.08 * area_similarity
        + 0.08 * float(qc.largest_component_ratio)
        + 0.07 * component_quality
        + 0.05 * hole_quality
        + 0.05 * float(np.clip(candidate.base_retention, 0.0, 1.0))
        + 0.03 * float(bool(qc.passed))
    )
    return {
        "candidate": candidate.metadata(),
        "eligible": not reasons,
        "eligibility_reasons": reasons,
        "rank_score": float(rank),
        "sam_selected_index": int(result.selected_index),
        "sam_selected_score": float(result.selected_score),
        "candidate_area_ratio": candidate_area_ratio,
        "refined_area_ratio": refined_area_ratio,
        "refined_to_candidate_area_ratio": float(expansion),
        "conservative_hysteresis_applied": bool(is_conservative),
        "sam_scores": [score.to_dict() for score in result.scores],
        "mask_qc": qc.to_dict(),
        "mask_sha256": _mask_sha256(result.mask),
    }


def _select_evaluation(evaluations: Sequence[Mapping[str, Any]]) -> int:
    if not evaluations:
        raise PlantarFullSamAnalysisError("Local SAM candidate bank is empty.")
    eligible = [index for index, value in enumerate(evaluations) if value["eligible"]]
    if not eligible:
        reasons = sorted(
            {
                str(reason)
                for value in evaluations
                for reason in value.get("eligibility_reasons", ())
            }
        )
        raise PlantarFullSamAnalysisError(
            "Every local SAM candidate failed numeric QC: "
            + (", ".join(reasons) if reasons else "unspecified QC failure")
        )
    conservative = [
        index
        for index in eligible
        if evaluations[index]["candidate"]["method"] in CONSERVATIVE_METHODS
    ]
    pool = conservative or eligible
    if not conservative:
        best_rank = max(float(evaluations[index]["rank_score"]) for index in pool)
        near_best = [
            index
            for index in pool
            if float(evaluations[index]["rank_score"])
            >= best_rank - COLOR_FALLBACK_RANK_TOLERANCE
        ]
        preferred_intersection = [
            index
            for index in near_best
            if evaluations[index]["candidate"]["method"] == "lab_hsv_intersection"
            and abs(float(evaluations[index]["candidate"]["strength"]) - 0.35)
            < 1e-8
        ]
        if preferred_intersection:
            pool = preferred_intersection
    return max(
        pool,
        key=lambda index: (
            float(evaluations[index]["rank_score"]),
            float(evaluations[index]["sam_selected_score"]),
            float(evaluations[index]["candidate"]["base_retention"]),
            -int(evaluations[index]["candidate"]["number"]),
        ),
    )


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(
        (np.asarray(mask) > 0).astype(np.uint8), connectivity=8
    )
    return max(0, int(count) - 1)


def _render_contact_overlay(
    image_rgb: np.ndarray,
    sole_mask: np.ndarray,
    contact_mask: np.ndarray,
    *,
    contact_coverage: np.ndarray | None = None,
) -> np.ndarray:
    """Render one coherent binary region with a constant fill and outline.

    ``contact_coverage`` remains an accepted compatibility argument, but its
    confidence values never change display opacity.  Probability gradients
    made the old result look like heat-map noise rather than segmentation.
    """

    sole = np.asarray(sole_mask, dtype=bool)
    selected = np.asarray(contact_mask, dtype=bool) & sole
    if contact_coverage is not None:
        supplied = np.asarray(contact_coverage, dtype=np.float32)
        if supplied.shape != selected.shape or not bool(np.isfinite(supplied).all()):
            raise PlantarFullSamAnalysisError(
                "Contact display coverage is geometrically invalid."
            )

    overlay = np.asarray(image_rgb, dtype=np.float32).copy()
    fill_color = np.asarray(CONTACT_OVERLAY_COLOR_RGB, dtype=np.float32)
    overlay[selected] = (
        overlay[selected] * (1.0 - CONTACT_OVERLAY_ALPHA)
        + fill_color * CONTACT_OVERLAY_ALPHA
    )

    # Gently soften the anti-aliased line before clipping it inward. This
    # changes only display RGB; contact masks, holes and scores stay intact.
    outline_width = int(
        np.clip(
            round(min(selected.shape) / CONTACT_OUTLINE_WIDTH_REFERENCE_PIXELS),
            CONTACT_OUTLINE_MIN_WIDTH_PIXELS,
            CONTACT_OUTLINE_MAX_WIDTH_PIXELS,
        )
    )
    edge = np.zeros(selected.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(
        selected.astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(
            edge,
            contours,
            -1,
            255,
            thickness=outline_width,
            lineType=cv2.LINE_AA,
        )
    edge_strength = edge.astype(np.float32) / 255.0
    if CONTACT_OUTLINE_SMOOTH_SIGMA_PIXELS > 0:
        edge_strength = cv2.GaussianBlur(
            edge_strength, (0, 0),
            sigmaX=CONTACT_OUTLINE_SMOOTH_SIGMA_PIXELS,
            sigmaY=CONTACT_OUTLINE_SMOOTH_SIGMA_PIXELS,
            borderType=cv2.BORDER_CONSTANT,
        )
    edge_strength[~selected] = 0.0
    outline_alpha = (edge_strength * CONTACT_OUTLINE_ALPHA)[:, :, None]
    outline_color = np.asarray(CONTACT_OUTLINE_COLOR_RGB, dtype=np.float32)
    overlay = overlay * (1.0 - outline_alpha) + outline_color * outline_alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _fungal_contact_display_coverage(
    tinea_result: TineaAnalysisResult,
    sole_mask: np.ndarray,
    contact_mask: np.ndarray,
) -> np.ndarray:
    """Validate legacy probabilities and return binary display coverage.

    Kept for source compatibility only.  Confidence no longer affects the
    visualization; every accepted contact pixel has the same strength.
    """

    sole = np.asarray(sole_mask, dtype=bool)
    contact = np.asarray(contact_mask, dtype=bool) & sole
    exported = tinea_result.authoritative_fungal_probability
    if exported is None:
        # Compatibility for injected/legacy results. Production plantar
        # inference always exports the masked probability canvas.
        return np.ascontiguousarray(contact, dtype=np.float32)

    probability = np.asarray(exported)
    if probability.ndim != 2 or probability.shape != contact.shape:
        raise PlantarFullSamAnalysisError(
            "Authoritative fungal probability must match the plantar sole canvas."
        )
    try:
        finite = bool(np.isfinite(probability).all())
    except TypeError as exc:
        raise PlantarFullSamAnalysisError(
            "Authoritative fungal probability has an unsupported value type."
        ) from exc
    if not finite or bool(np.any((probability < 0.0) | (probability > 1.0))):
        raise PlantarFullSamAnalysisError(
            "Authoritative fungal probability must contain finite values from zero to one."
        )
    if bool(np.any(probability[~contact] != 0.0)):
        raise PlantarFullSamAnalysisError(
            "Authoritative fungal probability lies outside the authoritative mask."
        )

    return np.ascontiguousarray(contact, dtype=np.float32)


def _enhance_pressed_contact_input(
    image_rgb: np.ndarray,
    sole_mask: np.ndarray,
) -> np.ndarray:
    """Apply the frozen contact-only color normalization used in evaluation."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    sole = np.asarray(sole_mask) > 0
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != sole.shape:
        raise PlantarFullSamAnalysisError(
            "Pressed-contact image and sole mask geometry do not match."
        )
    if not bool(sole.any()):
        raise PlantarFullSamAnalysisError("Pressed-contact sole mask is empty.")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(luminance), a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )
    enhanced = np.clip(
        (enhanced.astype(np.float32) - 127.5) * 1.05 + 127.5,
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation, value = cv2.split(hsv)
    red = (
        ((hue <= 12.0) | (hue >= 168.0))
        & (saturation >= 35.0)
        & (value >= 40.0)
        & sole
    )
    saturation[red] = np.clip(saturation[red] * 1.10, 0, 255)
    value[red] = np.clip(value[red] * 1.02, 0, 255)
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    enhanced[~sole] = 255
    return np.ascontiguousarray(enhanced)


def _remove_small_contact_components(
    mask: np.ndarray,
    sole_mask: np.ndarray,
    *,
    minimum_ratio: float,
) -> np.ndarray:
    selected = (np.asarray(mask) > 0) & (np.asarray(sole_mask) > 0)
    sole_pixels = int(np.count_nonzero(sole_mask))
    minimum = max(
        CONTACT_COMPONENT_MIN_PIXELS,
        int(round(sole_pixels * float(minimum_ratio))),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8),
        connectivity=8,
    )
    result = np.zeros_like(selected, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            result[labels == label] = True
    return np.ascontiguousarray(result)


def _coherent_hard_contact_mask(
    probability: np.ndarray,
    sole_mask: np.ndarray,
) -> np.ndarray:
    selected = (
        np.asarray(probability, dtype=np.float32) >= CONTACT_HARD_THRESHOLD
    ) & (np.asarray(sole_mask) > 0)
    selected_u8 = cv2.morphologyEx(
        selected.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    selected_u8 = cv2.morphologyEx(
        selected_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return _remove_small_contact_components(
        selected_u8 > 0,
        sole_mask,
        minimum_ratio=CONTACT_COMPONENT_MIN_RATIO,
    )


def _hysteresis_contact_mask(
    probability: np.ndarray,
    sole_mask: np.ndarray,
) -> np.ndarray:
    sole = np.asarray(sole_mask) > 0
    values = np.asarray(probability, dtype=np.float32)
    support = (values >= CONTACT_HYSTERESIS_LOW_THRESHOLD) & sole
    seeds = (values >= CONTACT_HYSTERESIS_HIGH_THRESHOLD) & support
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    minimum = max(
        CONTACT_COMPONENT_MIN_PIXELS,
        int(round(int(sole.sum()) * CONTACT_COMPONENT_MIN_RATIO)),
    )
    result = np.zeros_like(support, dtype=bool)
    for label in range(1, count):
        component = labels == label
        if (
            int(stats[label, cv2.CC_STAT_AREA]) >= minimum
            and bool(np.any(seeds & component))
        ):
            result[component] = True
    result_u8 = cv2.morphologyEx(
        result.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    result_u8 = cv2.morphologyEx(
        result_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return _remove_small_contact_components(
        result_u8 > 0,
        sole,
        minimum_ratio=CONTACT_COMPONENT_MIN_RATIO,
    )


def _pressed_contact_glare_mask(
    image_rgb: np.ndarray,
    sole_mask: np.ndarray,
) -> np.ndarray:
    """Locate bright neutral reflections that resemble the training class."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    sole = np.asarray(sole_mask) > 0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    yellow_axis = lab[:, :, 2] - lab[:, :, 1]
    luminance = lab[:, :, 0]
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1].astype(
        np.float32
    )
    yellow_values = yellow_axis[sole]
    center = float(np.median(yellow_values))
    q25, q75 = np.percentile(yellow_values, (25.0, 75.0))
    scale = max(2.0, float(q75 - q25))
    luminance_cutoff = max(170.0, float(np.percentile(luminance[sole], 88.0)))
    saturation_cutoff = float(np.percentile(saturation[sole], 48.0))
    glare = (
        (luminance >= luminance_cutoff)
        & (yellow_axis <= center - 0.75 * scale)
        & (saturation <= saturation_cutoff)
        & sole
    )
    glare = cv2.morphologyEx(
        glare.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    glare = cv2.dilate(
        glare,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return np.ascontiguousarray((glare > 0) & sole)


def _reject_glare_contact_components(
    mask: np.ndarray,
    glare_mask: np.ndarray,
    sole_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    selected = (np.asarray(mask) > 0) & (np.asarray(sole_mask) > 0)
    glare = np.asarray(glare_mask) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8),
        connectivity=8,
    )
    result = np.zeros_like(selected, dtype=bool)
    rejected = 0
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = float(np.count_nonzero(component & glare) / max(area, 1))
        if overlap >= CONTACT_GLARE_COMPONENT_MAX_OVERLAP:
            rejected += 1
        else:
            result[component] = True
    return np.ascontiguousarray(result), rejected


def _postprocess_pressed_contact_probability(
    image_rgb: np.ndarray,
    probability: np.ndarray,
    sole_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn soft class scores into connected, unequivocal contact regions."""

    sole = np.asarray(sole_mask) > 0
    values = np.asarray(probability, dtype=np.float32)
    if values.shape != sole.shape or not bool(np.isfinite(values).all()):
        raise PlantarFullSamAnalysisError(
            "Pressed-contact probability has invalid geometry or values."
        )
    values = np.clip(values, 0.0, 1.0).copy()
    values[~sole] = 0.0
    hard = _coherent_hard_contact_mask(values, sole)
    hard_ratio = float(hard.sum() / max(int(sole.sum()), 1))
    recovery_used = hard_ratio < CONTACT_HYSTERESIS_TRIGGER_RATIO
    candidate = _hysteresis_contact_mask(values, sole) if recovery_used else hard
    glare = _pressed_contact_glare_mask(image_rgb, sole)
    candidate, rejected_glare_components = _reject_glare_contact_components(
        candidate,
        glare,
        sole,
    )
    final_u8 = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    final_u8 = cv2.morphologyEx(
        final_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    final = _remove_small_contact_components(
        final_u8 > 0,
        sole,
        minimum_ratio=CONTACT_FINAL_COMPONENT_MIN_RATIO,
    )
    final &= sole
    return np.ascontiguousarray(final), {
        "hardThreshold": CONTACT_HARD_THRESHOLD,
        "hysteresisHighThreshold": CONTACT_HYSTERESIS_HIGH_THRESHOLD,
        "hysteresisLowThreshold": CONTACT_HYSTERESIS_LOW_THRESHOLD,
        "hysteresisTriggerRatio": CONTACT_HYSTERESIS_TRIGGER_RATIO,
        "hardMaskRatio": hard_ratio,
        "hysteresisRecoveryUsed": recovery_used,
        "glareRejectedComponentCount": rejected_glare_components,
        "glarePixelCount": int(glare.sum()),
        "cleanup": "close9_open5_relative_component_filter_v1",
        "minimumComponentPixels": max(
            CONTACT_COMPONENT_MIN_PIXELS,
            int(round(int(sole.sum()) * CONTACT_FINAL_COMPONENT_MIN_RATIO)),
        ),
    }


def _canvas_placement(
    image_shape: Sequence[int],
) -> tuple[int, int, int, int, int, int]:
    """Return canvas size, resized image size, and centered placement."""

    if len(image_shape) < 2:
        raise PlantarFullSamAnalysisError("Contact input must have height and width.")
    width = int(settings.plantar_canvas_width)
    height = int(settings.plantar_canvas_height)
    margin = float(np.clip(settings.plantar_canvas_margin_ratio, 0.0, 0.25))
    if width <= 0 or height <= width:
        raise PlantarFullSamAnalysisError(
            "Plantar canvas must be a positive portrait canvas."
        )
    usable_width = max(1, int(round(width * (1.0 - 2.0 * margin))))
    usable_height = max(1, int(round(height * (1.0 - 2.0 * margin))))
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if min(image_height, image_width) <= 0:
        raise PlantarFullSamAnalysisError("Contact input dimensions must be positive.")
    scale = min(
        usable_width / image_width,
        usable_height / image_height,
    )
    target_width = max(1, int(round(image_width * scale)))
    target_height = max(1, int(round(image_height * scale)))
    offset_x = (width - target_width) // 2
    offset_y = (height - target_height) // 2
    return width, height, target_width, target_height, offset_x, offset_y


def _fit_to_canvas(
    image_rgb: np.ndarray,
    sole_mask: np.ndarray,
    contact_mask: np.ndarray,
    source_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        width,
        height,
        target_width,
        target_height,
        offset_x,
        offset_y,
    ) = _canvas_placement(image_rgb.shape)
    scale = target_width / max(1, image_rgb.shape[1])
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_image = cv2.resize(
        image_rgb, (target_width, target_height), interpolation=interpolation
    )
    resized_sole = cv2.resize(
        sole_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    resized_contact = cv2.resize(
        contact_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    contact_coverage = cv2.resize(
        contact_mask.astype(np.float32),
        (target_width, target_height),
        interpolation=(cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR),
    )
    resized_source_alpha = cv2.resize(
        source_alpha,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    selected = resized_contact & resized_sole
    overlay = _render_contact_overlay(
        resized_image,
        resized_sole,
        selected,
        contact_coverage=contact_coverage,
    )

    # Use a soft edge only for the display alpha.  Contact/sole quantitative
    # masks remain hard and topology preserving.
    alpha = cv2.GaussianBlur(
        resized_sole.astype(np.float32) * 255.0,
        (0, 0),
        sigmaX=max(0.8, min(1.25, max(target_height, target_width) / 1024.0)),
    )
    alpha = np.minimum(alpha, resized_source_alpha.astype(np.float32))
    # Feather inward only.  Letting Gaussian support extend past the hard sole
    # would reveal the photographed background colour as a translucent halo.
    alpha = np.where(resized_sole, alpha, 0.0)
    alpha = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
    cell = np.zeros((height, width, 4), dtype=np.uint8)
    # Pathology inference must see the clean photographed colour, never the
    # sky-blue contact visualization.  Use an opaque white background so
    # PIL's later RGB conversion cannot turn transparent padding black.
    tinea_input = np.full((height, width, 3), 255, dtype=np.uint8)
    sole_canvas = np.zeros((height, width), dtype=np.uint8)
    contact_canvas = np.zeros((height, width), dtype=np.uint8)
    region = cell[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ]
    region[..., :3] = overlay
    region[..., 3] = alpha
    tinea_region = tinea_input[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ]
    tinea_region[resized_sole] = resized_image[resized_sole]
    sole_canvas[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = resized_sole.astype(np.uint8) * 255
    contact_canvas[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = selected.astype(np.uint8) * 255
    cell[cell[..., 3] == 0, :3] = 0
    return cell, sole_canvas, contact_canvas, tinea_input


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(array, dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        mode = "RGBA"
    elif image.ndim == 3 and image.shape[2] == 3:
        mode = "RGB"
    elif image.ndim == 2:
        mode = "L"
    else:
        raise PlantarFullSamAnalysisError(f"Unsupported PNG array shape: {image.shape}")
    try:
        Image.fromarray(image, mode=mode).save(path, format="PNG")
    except OSError as exc:
        raise PlantarFullSamAnalysisError(f"Could not save PNG: {path}") from exc


def _encode_png_bytes(array: np.ndarray) -> bytes:
    """Encode an RGB canvas losslessly for the existing tinea service API."""

    image = np.asarray(array, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise PlantarFullSamAnalysisError(
            f"Tinea input must be an RGB image, got {image.shape}."
        )
    try:
        with BytesIO() as stream:
            Image.fromarray(image, mode="RGB").save(stream, format="PNG")
            return stream.getvalue()
    except OSError as exc:
        raise PlantarFullSamAnalysisError(
            "Could not encode the clean plantar image for fungal segmentation."
        ) from exc


class LocalPlantarFullSamAnalyzer:
    """YOLO sole geometry plus dedicated SegFormer contact inference."""

    def __init__(self) -> None:
        self._inference_lock = threading.RLock()
        # The production constructor deliberately does not call ``_load_models``:
        # the old decoder and plantar SAM checkpoint are no longer dependencies
        # of footprint generation. SegFormer is loaded lazily on first inference.
        self.detector = get_plantar_foot_roi_detector()
        yolo_checkpoint = Path(weights.plantar_foot_segmentation)
        contact_checkpoint = Path(weights.plantar_contact_segformer)
        for checkpoint_name, checkpoint_path in (
            ("YOLO", yolo_checkpoint),
            ("SegFormer-B0", contact_checkpoint),
        ):
            if checkpoint_path.is_file():
                continue
            raise PlantarFullSamAnalysisError(
                f"Required local {checkpoint_name} weight is missing: "
                f"{checkpoint_path}"
            )
        self.checkpoint_metadata: dict[str, Any] = {
            "contact_model": "segformer_b0_real_finetune",
            "contact_class_name": "contact",
            "contact_output_channels": 1,
            "contact_activation": "sigmoid",
            "contact_model_loading": "lazy_local_singleton",
            "contact_threshold": float(settings.plantar_contact_threshold),
            "contact_checkpoint": str(contact_checkpoint.resolve()),
            "contact_checkpoint_sha256": _sha256_file(contact_checkpoint),
            "plantar_decoder_loaded": False,
            "plantar_sam_loaded": False,
            "yolo_checkpoint": str(yolo_checkpoint.resolve()),
            "yolo_checkpoint_sha256": _sha256_file(yolo_checkpoint),
        }

    @staticmethod
    def _resolve_device() -> torch.device:
        requested = str(settings.plantar_inference_device).strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise PlantarFullSamAnalysisError(
                "PLANTAR_INFERENCE_DEVICE=cuda but CUDA is unavailable."
            )
        return torch.device(requested)

    def _load_models(self) -> None:
        checkpoint = Path(weights.plantar_full_sam)
        sam_checkpoint = Path(weights.plantar_sam)
        yolo_checkpoint = Path(weights.plantar_foot_segmentation)
        for path in (checkpoint, sam_checkpoint, yolo_checkpoint):
            if not path.is_file():
                raise PlantarFullSamAnalysisError(f"Required local weight is missing: {path}")

        try:
            from segment_anything import SamPredictor
            from app.plantar_full_sam.anatomy_guided_segmenter import (
                AnatomyGuidedSamPressureSegmenter,
            )
        except ImportError as exc:
            raise PlantarFullSamAnalysisError(
                "Could not import the local SAM/anatomy runtime."
            ) from exc

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise PlantarFullSamAnalysisError(
                f"Could not load local plantar decoder checkpoint: {checkpoint}"
            ) from exc
        if not isinstance(state, dict) or state.get("format_version") != CHECKPOINT_FORMAT:
            raise PlantarFullSamAnalysisError(
                f"Unsupported plantar decoder checkpoint contract: {checkpoint}"
            )
        config = dict(state.get("segmentation_config") or {})
        if config.get("architecture") != "hybrid_foot_anatomy_fpn":
            raise PlantarFullSamAnalysisError(
                "Plantar decoder must use hybrid_foot_anatomy_fpn."
            )
        try:
            segmenter = AnatomyGuidedSamPressureSegmenter.from_decoder_config(
                sam_checkpoint,
                config,
                anatomy_map_location="cpu",
            ).to(self.device)
            segmenter.decoder.load_state_dict(state["decoder_state_dict"], strict=True)
            segmenter.eval()
            predictor = SamPredictor(segmenter.sam)
            detector = get_plantar_foot_roi_detector(
                device="0" if self.device.type == "cuda" else "cpu"
            )
        except Exception as exc:
            raise PlantarFullSamAnalysisError(
                f"Could not initialize local plantar models: {type(exc).__name__}: {exc}"
            ) from exc

        self.segmenter = segmenter
        self.predictor = predictor
        self.detector = detector
        self.threshold = float(state.get("threshold", 0.5))
        self.checkpoint_metadata = {
            "format_version": state.get("format_version"),
            "source_joint_checkpoint_sha256": state.get(
                "source_joint_checkpoint_sha256"
            ),
            "source_epoch": state.get("source_epoch"),
            "source_metrics": state.get("source_metrics", {}),
            "decoder_checkpoint": str(checkpoint.resolve()),
            "decoder_checkpoint_sha256": _sha256_file(checkpoint),
            "sam_checkpoint": str(sam_checkpoint.resolve()),
            "sam_checkpoint_sha256": _sha256_file(sam_checkpoint),
            "yolo_checkpoint": str(yolo_checkpoint.resolve()),
            "yolo_checkpoint_sha256": _sha256_file(yolo_checkpoint),
        }

    def _segment_contact(
        self,
        image_rgb: np.ndarray,
        sole_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with torch.inference_mode():
            prepared = self.segmenter.prepare_batch([image_rgb], device=self.device)
            geometry = prepared.geometries[0]
            support_low = support_mask_to_low_res(
                sole_mask,
                input_size=geometry.input_size,
                sam_image_size=int(self.segmenter.sam.image_encoder.img_size),
                output_size=int(self.segmenter.output_size),
                device=self.device,
                dtype=torch.float32,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                prediction = self.segmenter(
                    prepared.pixel_values,
                    return_embeddings=True,
                )
            base_logits = constrain_pressure_logits(
                prediction.low_res_logits.detach(),
                support_low.to(dtype=prediction.low_res_logits.dtype),
            )
            restored = self.segmenter.postprocess_logits(
                base_logits.float(), prepared.geometries
            )[0]
            base_probability = torch.sigmoid(restored).cpu().numpy()

        base_mask = (base_probability >= self.threshold) & sole_mask
        if not bool(base_mask.any()):
            raise PlantarFullSamAnalysisError(
                "Frozen V4 decoder returned an empty local contact proposal."
            )
        candidates = [
            candidate
            for candidate in generate_candidate_masks(
                image_rgb,
                base_probability,
                sole_mask,
                strengths=(0.35, 0.50, 0.65),
                mask_threshold=self.threshold,
            )
            if candidate.area_pixels > 0
        ]
        if not candidates:
            raise PlantarFullSamAnalysisError("No non-empty local SAM candidates were generated.")

        results: list[SamRefinementResult] = []
        evaluations: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            result = refine_with_sam(
                self.predictor,
                image_rgb,
                candidate,
                sole_mask,
                positive_points=None,
                negative_points=None,
                points_are_normalized=False,
                set_image=candidate_index == 0,
                multimask_output=True,
                qc_kwargs={
                    "maximum_area_ratio": float(settings.plantar_max_contact_ratio)
                },
            )
            results.append(result)
            evaluations.append(
                _candidate_evaluation(
                    candidate,
                    result,
                    minimum_sam_score=float(settings.plantar_sam_min_score),
                    maximum_contact_ratio=float(settings.plantar_max_contact_ratio),
                )
            )

        selected_index = _select_evaluation(evaluations)
        selected_candidate = candidates[selected_index]
        selected_result = results[selected_index]
        raw_mask = (np.asarray(selected_result.mask) > 0) & sole_mask
        if not bool(raw_mask.any()):
            raise PlantarFullSamAnalysisError("Selected local Full SAM mask is empty.")

        scale = max(raw_mask.shape) / 1024.0
        cleanup = cleanup_full_sam_mask(
            raw_mask,
            support_mask=sole_mask,
            config=FullSamCleanupConfig(
                toe_direction="top",
                toe_component_support_area_ratio=0.001,
                max_hole_area_pixels=max(1, int(round(1500 * scale * scale))),
                max_hole_span_pixels=max(1, int(round(60 * scale))),
                min_hole_circularity=0.55,
                small_hole_circularity_exemption_pixels=max(
                    1, int(round(100 * scale * scale))
                ),
            ),
        )
        smoothing = smooth_full_sam_mask(
            cleanup.cleaned_mask,
            support_mask=sole_mask,
            config=FullSamSmoothingConfig(),
        )
        final_mask = np.asarray(smoothing.smoothed_mask) > 0
        if not bool(final_mask.any()):
            raise PlantarFullSamAnalysisError(
                "Full SAM cleanup produced an empty contact mask."
            )

        support_pixels = max(int(sole_mask.sum()), 1)
        audit = {
            "threshold": self.threshold,
            "base_contact_pixels": int(base_mask.sum()),
            "base_contact_ratio": float(base_mask.sum() / support_pixels),
            "base_probability_sha256": sha256(
                np.ascontiguousarray(base_probability).tobytes()
            ).hexdigest(),
            "selection_policy": SELECTION_POLICY,
            "candidate_count": len(candidates),
            "candidate_evaluations": evaluations,
            "selected_candidate_index": int(selected_index),
            "selected_candidate": selected_candidate.metadata(),
            "selected_sam_score": float(selected_result.selected_score),
            "selected_sam_qc": selected_result.qc.to_dict(),
            "cleanup": cleanup.audit,
            "smoothing": smoothing.audit,
            "raw_contact_pixels": int(raw_mask.sum()),
            "final_contact_pixels": int(final_mask.sum()),
            "final_contact_ratio": float(final_mask.sum() / support_pixels),
            "final_component_count": _component_count(final_mask),
            "final_mask_sha256": _mask_sha256(final_mask),
        }
        return np.ascontiguousarray(final_mask), audit

    def _refine_sole_alpha(
        self,
        image_rgb: np.ndarray,
        yolo_sole_mask: np.ndarray,
        contact_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Build the published sole alpha with local SAM and safe geometry.

        Contact pixels are measurement evidence and therefore may never be
        removed by the display-matte refinement.  SAM is allowed only to
        suppress pixels from the YOLO maximum support; the lower-zone cleanup
        then removes residual heel/reflection lobes without editing the upper
        76 percent of the sole.
        """

        sole = np.ascontiguousarray(np.asarray(yolo_sole_mask) > 0)
        contact = np.ascontiguousarray(np.asarray(contact_mask) > 0)
        if sole.shape != contact.shape or not bool(sole.any()):
            raise PlantarFullSamAnalysisError(
                "Sole alpha inputs are empty or geometrically misaligned."
            )
        if bool(np.any(contact & ~sole)):
            raise PlantarFullSamAnalysisError(
                "Contact evidence lies outside the YOLO plantar support."
            )

        try:
            sam_result = refine_sole_alpha_with_sam(
                self.predictor,
                image_rgb,
                sole,
                contact,
                set_image=True,
            )
        except Exception as exc:
            raise PlantarFullSamAnalysisError(
                "Local SAM could not refine the complete plantar alpha matte: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # The SAM proposal is a suppress-only shape cue.  Restore every Full
        # SAM contact pixel before geometry cleanup so downstream alpha can
        # never hide measured contact evidence.
        sam_mask = np.ascontiguousarray(np.asarray(sam_result.mask) > 0) & sole
        restored_contact = contact & ~sam_mask
        sam_mask |= contact
        cleanup = cleanup_sole_alpha(sam_mask, contact)
        final_sole = np.ascontiguousarray(np.asarray(cleanup.mask) > 0)
        if not bool(final_sole.any()) or bool(np.any(contact & ~final_sole)):
            raise PlantarFullSamAnalysisError(
                "Complete-sole alpha cleanup did not preserve all contact pixels."
            )
        if bool(np.any(final_sole & ~sole)):
            raise PlantarFullSamAnalysisError(
                "Complete-sole alpha refinement expanded beyond YOLO support."
            )

        audit = {
            "policy": "local_sam_sole_alpha_then_lower_contact_cleanup_v1",
            "provenance": {
                "local_only": True,
                "gpt_used": False,
                "openai_used": False,
                "network_used": False,
            },
            "sam": sam_result.audit,
            "sam_selected_variant": sam_result.selected_variant,
            "sam_fallback_used": bool(sam_result.fallback_used),
            "contact_pixels_restored_after_sam": int(restored_contact.sum()),
            "cleanup": cleanup.audit,
            "input_yolo_pixels": int(sole.sum()),
            "output_sole_pixels": int(final_sole.sum()),
            "output_to_yolo_area_ratio": float(
                final_sole.sum() / max(int(sole.sum()), 1)
            ),
            "contact_ratio_to_input_yolo": float(
                contact.sum() / max(int(sole.sum()), 1)
            ),
            "contact_ratio_to_output_sole": float(
                contact.sum() / max(int(final_sole.sum()), 1)
            ),
            "all_contact_pixels_preserved": True,
            "contained_in_yolo": True,
            "final_mask_sha256": _mask_sha256(final_sole),
        }
        return final_sole, audit

    def _prepare_canvas(
        self,
        image_bytes: bytes,
        side: str,
        *,
        prepared_input: PreparedPlantarFootInput | None = None,
    ) -> _PreparedPlantarCanvas:
        """Build canonical YOLO-only sole artifacts before contact inference."""

        declared_side = str(side).strip().lower()
        if declared_side not in {"left", "right"}:
            raise PlantarFullSamAnalysisError("side must be 'left' or 'right'.")

        image_rgb, source_alpha, source_metadata = _decode_image(image_bytes)
        if prepared_input is None:
            prepared_input = prepare_plantar_foot_input(
                image_rgb,
                extract=lambda rgb: extract_plantar_foot_roi(
                    rgb, detector=self.detector,
                ),
            )
        elif prepared_input.audit.get("source_sha256") != _sha256_bytes(image_bytes):
            raise PlantarFullSamAnalysisError(
                "Validated plantar input does not match the original upload."
            )
        roi = prepared_input.roi
        if not roi.found:
            raise PlantarFullSamAnalysisError(
                "Local YOLO could not segment a plantar sole after preprocessing."
            )
        source_alpha = _rotate_quarter_turn_clockwise(
            source_alpha, prepared_input.rotation_degrees_clockwise,
        )
        if (
            source_alpha.shape != prepared_input.image_rgb.shape[:2]
            or source_alpha.shape != roi.mask.shape
        ):
            raise PlantarFullSamAnalysisError(
                "Preprocessed plantar image, alpha, and YOLO mask are misaligned."
            )
        (
            orientation,
            _,
            _,
            _,
            detector_crop_bbox,
        ) = _canonical_cardinal_crop(roi, source_alpha)
        minimum_orientation_confidence = _require_unambiguous_orientation(
            orientation
        )

        # YOLO is now the sole source of the complete-foot alpha support. The
        # old plantar SAM suppress-only matte refinement is deliberately not
        # called because loading it would reintroduce the retired model path.
        oriented_sole = np.ascontiguousarray(
            np.asarray(orientation.sole_mask) > 0
        )
        if not bool(oriented_sole.any()):
            raise PlantarFullSamAnalysisError(
                "The canonical YOLO plantar sole mask is empty."
            )
        contact_crop_bbox = bbox_from_mask(
            oriented_sole,
            padding=0,
            padding_ratio=float(settings.plantar_contact_crop_padding_ratio),
        )
        cx1, cy1, cx2, cy2 = contact_crop_bbox
        contact_image = np.ascontiguousarray(
            orientation.image_rgb[cy1:cy2, cx1:cx2]
        )
        contact_sole = np.ascontiguousarray(
            oriented_sole[cy1:cy2, cx1:cx2]
        )
        contact_alpha = np.ascontiguousarray(
            orientation.source_alpha[cy1:cy2, cx1:cx2]
        )
        model_sole = np.ascontiguousarray(np.asarray(roi.crop_mask) > 0)
        # Reconstruct the model crop directly from the decoded original.
        rotated_original = _rotate_quarter_turn_clockwise(
            image_rgb, prepared_input.rotation_degrees_clockwise,
        )
        mx1, my1, mx2, my2 = roi.bbox
        if (
            rotated_original.shape[:2] != roi.mask.shape
            or not (0 <= mx1 < mx2 <= rotated_original.shape[1]
                    and 0 <= my1 < my2 <= rotated_original.shape[0])
        ):
            raise PlantarFullSamAnalysisError("YOLO crop is outside the rotated original.")
        model_source = rotated_original[my1:my2, mx1:mx2]
        if (
            model_source.ndim != 3
            or model_source.shape[2] != 3
            or model_source.shape[:2] != model_sole.shape
            or not bool(model_sole.any())
        ):
            raise PlantarFullSamAnalysisError(
                "The target-compatible YOLO model crop is invalid."
            )
        contact_model_input, contact_preprocessing = _prepare_contact_model_input(
            model_source, model_sole, declared_side,
        )
        clean_rgba, sole_canvas, contact_canvas, tinea_input = _fit_to_canvas(
            contact_image,
            contact_sole,
            np.zeros_like(contact_sole, dtype=bool),
            contact_alpha,
        )
        if bool(np.any(contact_canvas)):
            raise PlantarFullSamAnalysisError(
                "Detector-only canvas preparation unexpectedly produced contact pixels."
            )

        # Use the photographed sole directly for the report background.
        # Model-only background whitening must not bleed into its edge colors
        # when the image is resized for display.
        contact_display_rgba = clean_rgba.copy()

        audit_base = {
            "contact_preprocessing": contact_preprocessing,
            "preprocessing": {
                **prepared_input.audit,
                "source_sha256": _sha256_bytes(image_bytes),
            },
            "source": {
                **source_metadata,
                "byte_count": len(image_bytes),
                "sha256": _sha256_bytes(image_bytes),
            },
            "foot_roi": {
                "confidence": float(roi.confidence),
                "detection_pass": roi.detection_pass,
                "attempted_transforms": list(roi.tta_attempted_transforms),
                "instance_count": int(roi.instance_count),
                "bbox_xyxy": [int(value) for value in roi.bbox],
                "source_mask_pixels": int(np.count_nonzero(roi.mask)),
                "cardinal_rotation_degrees_clockwise": int(
                    roi.cardinal_rotation_degrees_clockwise
                ),
                "selection_audit": roi.selection_audit,
            },
            "orientation": {
                "canonical": "toes_up_heel_down",
                "policy": CAPTURE_ORIENTATION_POLICY,
                "reference": "preprocessed_capture",
                "source_rotation_degrees_clockwise": int(
                    prepared_input.rotation_degrees_clockwise
                ),
                "rotation_degrees": orientation.rotation_degrees,
                "toe_end_margin": orientation.toe_end_margin,
                "toe_end_margin_used_for_rotation": False,
                "confidence": orientation.confidence,
                "confidence_source": "fixed_capture_convention_not_image_estimate",
                "operations": list(orientation.operations),
                "mirrored": False,
                "detector_crop_bbox_xyxy": list(detector_crop_bbox),
                "contact_crop_bbox_xyxy": list(contact_crop_bbox),
                "minimum_confidence": minimum_orientation_confidence,
                "warning": None,
            },
            "display_crop_bbox_xyxy_in_model_crop": [
                0,
                0,
                int(contact_image.shape[1]),
                int(contact_image.shape[0]),
            ],
            "segformer_model_input_bbox_xyxy_in_source": [
                int(value) for value in roi.bbox
            ],
        }
        return _PreparedPlantarCanvas(
            side=declared_side,
            clean_rgba=np.ascontiguousarray(clean_rgba),
            tinea_input_rgb=np.ascontiguousarray(tinea_input),
            sole_canvas=np.ascontiguousarray(sole_canvas),
            contact_model_input_rgb=np.ascontiguousarray(contact_model_input),
            contact_model_sole=np.ascontiguousarray(model_sole),
            contact_source_shape=tuple(int(value) for value in roi.original_shape),
            contact_source_bbox=tuple(int(value) for value in roi.bbox),
            contact_display_rotation_degrees_clockwise=(
                int(round(orientation.rotation_degrees)) % 360
            ),
            contact_display_crop_bbox=tuple(
                int(value) for value in contact_crop_bbox
            ),
            audit_base=audit_base,
            yolo_confidence=float(roi.confidence),
            orientation_confidence=float(orientation.confidence),
            contact_display_rgba=np.ascontiguousarray(contact_display_rgba),
        )

    @staticmethod
    def _predict_pressed_contact(
        prepared: _PreparedPlantarCanvas,
    ) -> _PressedContactPrediction:
        """Intersect original and restored horizontal-flip contact predictions."""

        sole = np.ascontiguousarray(np.asarray(prepared.sole_canvas) > 0)
        if not bool(sole.any()):
            raise PlantarFullSamAnalysisError(
                "YOLO plantar support is empty before SegFormer inference."
            )
        crop_sole = np.ascontiguousarray(
            np.asarray(prepared.contact_model_sole) > 0
        )
        source_input = np.asarray(prepared.contact_model_input_rgb)
        if (
            source_input.ndim != 3
            or source_input.shape[2] != 3
            or source_input.shape[:2] != crop_sole.shape
            or not bool(crop_sole.any())
        ):
            raise PlantarFullSamAnalysisError(
                "SegFormer input and YOLO crop geometry are invalid."
            )
        model_input = np.full(source_input.shape, 255, dtype=np.uint8)
        model_input[crop_sole] = np.asarray(source_input, dtype=np.uint8)[crop_sole]
        try:
            segmenter, model_metadata = get_plantar_contact_segmenter()
            prediction = segmenter.predict([model_input], device=segmenter.device)
            direct_probability = (
                prediction.original_probabilities[0].detach().cpu().numpy()
            )
            # Flip the actual crop before letterboxing, then restore the
            # original-resolution probability map, never the padded square.
            flipped_input = np.ascontiguousarray(model_input[:, ::-1])
            flipped_prediction = segmenter.predict(
                [flipped_input], device=segmenter.device,
            )
            flip_probability = np.ascontiguousarray(
                flipped_prediction.original_probabilities[0].detach().cpu().numpy()[:, ::-1]
            )
        except PlantarFullSamAnalysisError:
            raise
        except Exception as exc:
            raise PlantarFullSamAnalysisError(
                "Local SegFormer-B0 pressed-contact inference failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        direct_probability = np.asarray(direct_probability, dtype=np.float32)
        flip_probability = np.asarray(flip_probability, dtype=np.float32)
        if (
            direct_probability.shape != crop_sole.shape
            or flip_probability.shape != crop_sole.shape
        ):
            raise PlantarFullSamAnalysisError(
                "SegFormer probability does not match its YOLO foot crop."
            )
        if not (bool(np.isfinite(direct_probability).all())
                and bool(np.isfinite(flip_probability).all())):
            raise PlantarFullSamAnalysisError(
                "SegFormer probability contains non-finite values."
            )
        threshold = float(settings.plantar_contact_threshold)
        direct_mask = (direct_probability >= threshold) & crop_sole
        flip_mask = (flip_probability >= threshold) & crop_sole
        crop_contact = direct_mask & flip_mask
        # min >= threshold is exactly the two binary masks' intersection.
        # This artifact is a conservative agreement score, not a mean TTA.
        crop_probability = np.clip(
            np.minimum(direct_probability, flip_probability), 0.0, 1.0,
        )
        crop_probability[~crop_sole] = 0.0
        source_height, source_width = prepared.contact_source_shape
        sx1, sy1, sx2, sy2 = prepared.contact_source_bbox
        if (
            not (0 <= sx1 < sx2 <= source_width and 0 <= sy1 < sy2 <= source_height)
            or (sy2 - sy1, sx2 - sx1) != crop_probability.shape
        ):
            raise PlantarFullSamAnalysisError(
                "SegFormer crop cannot be restored to the source image."
            )
        source_probability = np.zeros(
            (source_height, source_width),
            dtype=np.float32,
        )
        source_probability[sy1:sy2, sx1:sx2] = crop_probability
        source_contact = np.zeros((source_height, source_width), dtype=np.uint8)
        source_contact[sy1:sy2, sx1:sx2] = crop_contact
        oriented_probability = _rotate_quarter_turn_clockwise(
            source_probability,
            prepared.contact_display_rotation_degrees_clockwise,
        )
        dx1, dy1, dx2, dy2 = prepared.contact_display_crop_bbox
        if not (
            0 <= dx1 < dx2 <= oriented_probability.shape[1]
            and 0 <= dy1 < dy2 <= oriented_probability.shape[0]
        ):
            raise PlantarFullSamAnalysisError(
                "SegFormer probability cannot be mapped to the display crop."
            )
        display_probability = np.ascontiguousarray(
            oriented_probability[dy1:dy2, dx1:dx2]
        )
        display_contact = _rotate_quarter_turn_clockwise(
            source_contact, prepared.contact_display_rotation_degrees_clockwise,
        )[dy1:dy2, dx1:dx2]
        (
            _,
            _,
            target_width,
            target_height,
            offset_x,
            offset_y,
        ) = _canvas_placement(display_probability.shape)
        restored_probability = cv2.resize(
            display_probability,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        probability = np.zeros(sole.shape, dtype=np.float32)
        probability[
            offset_y : offset_y + target_height,
            offset_x : offset_x + target_width,
        ] = restored_probability
        probability[~sole] = 0.0
        mask = np.zeros(sole.shape, dtype=bool)
        mask[
            offset_y : offset_y + target_height,
            offset_x : offset_x + target_width,
        ] = cv2.resize(
            display_contact,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        mask &= sole
        sole_values = probability[sole]
        return _PressedContactPrediction(
            mask=np.ascontiguousarray(mask),
            probability=np.ascontiguousarray(probability, dtype=np.float32),
            audit={
                "source": "dedicated_segformer_b0_contact_model",
                "architecture": "segformer_b0",
                "className": "contact",
                "classIndex": 0,
                "activation": "sigmoid",
                "analysisMode": "plantar_pressed_contact_only",
                "pathologyReportEligible": False,
                "modelInferenceExecuted": True,
                "inferenceGeometry": (
                    "yolo_foot_crop_aspect_resize_top_left_letterbox_512"
                ),
                "slidingWindowUsed": False,
                "preprocessing": (
                    "original_rgb_outside_yolo_white_imagenet"
                ),
                "contactPreprocessing": prepared.audit_base.get("contact_preprocessing", {}),
                "tta": {
                    "transforms": ["identity", "horizontal_flip"],
                    "aggregation": "binary_intersection",
                    "probabilityAggregation": "minimum",
                    "flipRestoredBeforeAggregation": True,
                    "aggregationCoordinateSpace": "original_model_crop",
                    "inferenceCount": 2,
                    "directContactPixels": int(direct_mask.sum()),
                    "flipRestoredContactPixels": int(flip_mask.sum()),
                    "intersectionContactPixels": int(crop_contact.sum()),
                    "disagreementPixels": int(np.count_nonzero(direct_mask ^ flip_mask)),
                },
                "sourceCropHeight": int(model_input.shape[0]),
                "sourceCropWidth": int(model_input.shape[1]),
                "sourceCropBBoxXyxy": [sx1, sy1, sx2, sy2],
                "displayRotationDegreesClockwise": int(
                    prepared.contact_display_rotation_degrees_clockwise
                ),
                "displayCropBBoxXyxy": [dx1, dy1, dx2, dy2],
                "cropBBoxXyxyInCanvas": [
                    int(offset_x),
                    int(offset_y),
                    int(offset_x + target_width),
                    int(offset_y + target_height),
                ],
                "cropPaddingRatio": float(
                    settings.plantar_contact_crop_padding_ratio
                ),
                "threshold": threshold,
                "probabilityMinimumInSole": float(sole_values.min()),
                "probabilityMaximumInSole": float(sole_values.max()),
                "probabilityMeanInSole": float(sole_values.mean()),
                "postprocessing": {
                    "policy": "identity_hflip_binary_intersection_then_clip_to_yolo",
                    "displayMaskInterpolation": "nearest",
                    "displayProbabilityInterpolation": "nearest",
                    "morphologyApplied": False,
                    "componentFilteringApplied": False,
                },
                "model": model_metadata,
            },
            model_input_rgb=np.ascontiguousarray(model_input),
            model_input_bbox=(sx1, sy1, sx2, sy2),
        )

    @staticmethod
    def _fungal_contact_mask(
        tinea_result: TineaAnalysisResult,
        sole_canvas: np.ndarray,
    ) -> np.ndarray:
        """Validate the authoritative full-canvas fungal mask contract."""

        exported = tinea_result.authoritative_fungal_mask
        if exported is None:
            raise PlantarFullSamAnalysisError(
                "Plantar tinea analysis did not export an authoritative fungal mask."
            )
        mask = np.asarray(exported)
        sole = np.asarray(sole_canvas) > 0
        if mask.ndim != 2 or mask.shape != sole.shape:
            raise PlantarFullSamAnalysisError(
                "Authoritative fungal mask must match the plantar sole canvas."
            )
        try:
            finite = bool(np.isfinite(mask).all())
        except TypeError as exc:
            raise PlantarFullSamAnalysisError(
                "Authoritative fungal mask has an unsupported value type."
            ) from exc
        if not finite:
            raise PlantarFullSamAnalysisError(
                "Authoritative fungal mask contains non-finite values."
            )
        contact = np.ascontiguousarray(mask > 0)
        if bool(np.any(contact & ~sole)):
            raise PlantarFullSamAnalysisError(
                "Authoritative fungal mask lies outside the YOLO plantar support."
            )
        # Empty is a valid authoritative result for a foot with no gated
        # fungal-class evidence.
        return contact

    @staticmethod
    def _pressed_contact_mask(
        prediction: _PressedContactPrediction,
        sole_canvas: np.ndarray,
    ) -> np.ndarray:
        mask = np.asarray(prediction.mask)
        sole = np.asarray(sole_canvas) > 0
        probability = np.asarray(prediction.probability)
        if mask.ndim != 2 or mask.shape != sole.shape:
            raise PlantarFullSamAnalysisError(
                "Pressed-contact mask must match the plantar sole canvas."
            )
        if probability.ndim != 2 or probability.shape != sole.shape:
            raise PlantarFullSamAnalysisError(
                "Pressed-contact probability must match the plantar sole canvas."
            )
        if not bool(np.isfinite(probability).all()) or bool(
            np.any((probability < 0.0) | (probability > 1.0))
        ):
            raise PlantarFullSamAnalysisError(
                "Pressed-contact probability must contain finite values from zero to one."
            )
        contact = np.ascontiguousarray(mask > 0)
        if bool(np.any(contact & ~sole)):
            raise PlantarFullSamAnalysisError(
                "Pressed-contact mask lies outside the YOLO plantar support."
            )
        return contact

    def _finalize_canvas(
        self,
        prepared: _PreparedPlantarCanvas,
        prediction: _PressedContactPrediction,
        request_dir: Path,
    ) -> PlantarFullSamResult:
        sole = np.ascontiguousarray(prepared.sole_canvas > 0)
        contact = self._pressed_contact_mask(prediction, prepared.sole_canvas)
        contact_canvas = contact.astype(np.uint8) * 255

        display_base = prepared.contact_display_rgba
        if not isinstance(display_base, np.ndarray):
            raise PlantarFullSamAnalysisError(
                "Original contact display is required for the plantar visualization."
            )
        if (
            display_base.dtype != np.uint8
            or display_base.shape != prepared.clean_rgba.shape
            or not np.array_equal(display_base[..., 3], prepared.clean_rgba[..., 3])
        ):
            raise PlantarFullSamAnalysisError(
                "Contact display image must preserve the clean image dimensions and alpha."
            )
        rgba = display_base.copy()
        rgba[..., :3] = _render_contact_overlay(
            display_base[..., :3],
            sole,
            contact,
        )
        rgba[rgba[..., 3] == 0, :3] = 0

        sole_pixels = int(sole.sum())
        contact_pixels = int(contact.sum())
        region_count = _component_count(contact)
        contact_ratio = float(contact_pixels / max(sole_pixels, 1))
        inference_audit = {
            **prediction.audit,
            "source": "dedicated_segformer_b0_contact_model",
            "architecture": "segformer_b0",
            "class_name": "contact",
            "class_index": 0,
            "activation": "sigmoid",
            "contact_threshold": float(settings.plantar_contact_threshold),
            "analysis_mode": "plantar_pressed_contact_only",
            "model_inference_executed": True,
            "pathology_report_eligible": False,
            "empty_mask_allowed": True,
            "final_contact_pixels": contact_pixels,
            "final_contact_ratio": contact_ratio,
            "final_component_count": region_count,
            "final_mask_sha256": _mask_sha256(contact),
            "display_confidence": {
                "source": "binary_contact_mask_only",
                "normalization": "none",
                "uniformFillAlpha": CONTACT_OVERLAY_ALPHA,
                "uniformOutlineAlpha": CONTACT_OUTLINE_ALPHA,
                "outlineSmoothingSigmaPixels": CONTACT_OUTLINE_SMOOTH_SIGMA_PIXELS,
                "outlineSmoothing": "gaussian_antialias_clipped_inside_contact",
                "outsideAuthoritativeMaskStrength": 0.0,
                "quantitativeMaskModified": False,
                "modelConfidenceAffectsDisplayOpacity": False,
            },
        }
        sole_alpha_audit = {
            "policy": "canonical_yolo_passthrough_v1",
            "provenance": {
                "local_only": True,
                "gpt_used": False,
                "openai_used": False,
                "network_used": False,
                "plantar_decoder_used": False,
                "plantar_sam_used": False,
            },
            "input_yolo_pixels": sole_pixels,
            "output_sole_pixels": sole_pixels,
            "output_to_yolo_area_ratio": 1.0,
            "contact_ratio_to_output_sole": contact_ratio,
            "all_contact_pixels_preserved": True,
            "contained_in_yolo": True,
            "final_mask_sha256": _mask_sha256(sole),
        }

        request_dir = Path(request_dir)
        request_dir.mkdir(parents=True, exist_ok=True)
        visualization_path = request_dir / f"{prepared.side}_plantar_transparent.png"
        tinea_input_path = request_dir / f"{prepared.side}_tinea_input.png"
        contact_model_input_path = (
            request_dir / f"{prepared.side}_contact_model_input.png"
        )
        contact_mask_path = request_dir / f"{prepared.side}_contact_mask.png"
        contact_probability_path = (
            request_dir / f"{prepared.side}_contact_probability.png"
        )
        sole_mask_path = request_dir / f"{prepared.side}_sole_mask.png"
        audit_path = request_dir / f"{prepared.side}_audit.json"
        model_input_rgb = (
            prepared.tinea_input_rgb
            if prediction.model_input_rgb is None
            else prediction.model_input_rgb
        )
        probability_png = np.clip(
            np.rint(np.asarray(prediction.probability) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        _write_png(visualization_path, rgba)
        _write_png(tinea_input_path, prepared.tinea_input_rgb)
        _write_png(contact_model_input_path, model_input_rgb)
        _write_png(contact_mask_path, contact_canvas)
        _write_png(contact_probability_path, probability_png)
        _write_png(sole_mask_path, prepared.sole_canvas)

        alpha = rgba[..., 3]
        transparent_rgb_nonzero = int(
            np.count_nonzero(np.any(rgba[..., :3] != 0, axis=2) & (alpha == 0))
        )
        audit = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "status": "completed",
            "declared_side": prepared.side,
            "provenance": {
                "local_only": True,
                "gpt_used": False,
                "openai_used": False,
                "external_model_api_used": False,
                "network_used_for_analysis": False,
                "models": self.checkpoint_metadata,
            },
            "source": prepared.audit_base["source"],
            "preprocessing": prepared.audit_base.get("preprocessing", {}),
            "foot_roi": prepared.audit_base["foot_roi"],
            "orientation": prepared.audit_base["orientation"],
            "segmentation": inference_audit,
            "sole_alpha_refinement": sole_alpha_audit,
            "output": {
                "visualization_path": str(visualization_path),
                "tinea_input_path": str(tinea_input_path),
                "contact_model_input_path": str(contact_model_input_path),
                "contact_mask_path": str(contact_mask_path),
                "contact_probability_path": str(contact_probability_path),
                "sole_mask_path": str(sole_mask_path),
                "media_type": "image/png",
                "color_mode": "RGBA",
                "visualization_base": "original_upload_sole",
                "alpha_semantics": (
                    "canonical_yolo_complete_sole_silhouette_with_inward_feather"
                ),
                "contact_visualization": (
                    "uniform_binary_region_fill_with_softened_antialiased_inner_outline"
                ),
                "display_crop_bbox_xyxy_in_model_crop": prepared.audit_base[
                    "display_crop_bbox_xyxy_in_model_crop"
                ],
                "width": int(rgba.shape[1]),
                "height": int(rgba.shape[0]),
                "transparent_pixels": int(np.count_nonzero(alpha == 0)),
                "transparent_rgb_nonzero_pixels": transparent_rgb_nonzero,
                "visualization_sha256": _sha256_file(visualization_path),
                "tinea_input_sha256": _sha256_file(tinea_input_path),
                "contact_model_input_sha256": _sha256_file(
                    contact_model_input_path
                ),
                "contact_mask_sha256": _sha256_file(contact_mask_path),
                "contact_probability_sha256": _sha256_file(
                    contact_probability_path
                ),
                "sole_mask_sha256": _sha256_file(sole_mask_path),
            },
        }
        audit_path.write_text(
            json.dumps(_json_value(audit), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return PlantarFullSamResult(
            side=prepared.side,
            visualization_path=visualization_path,
            tinea_input_path=tinea_input_path,
            contact_mask_path=contact_mask_path,
            sole_mask_path=sole_mask_path,
            audit_path=audit_path,
            region_count=region_count,
            contact_area_ratio=contact_ratio,
            yolo_confidence=prepared.yolo_confidence,
            orientation_confidence=prepared.orientation_confidence,
            contact_probability_path=contact_probability_path,
            contact_model_input_path=contact_model_input_path,
            tinea_analysis_result=None,
        )

    def analyze(
        self,
        image_bytes: bytes,
        side: str,
        request_dir: Path,
        *,
        prepared_input: PreparedPlantarFootInput | None = None,
    ) -> PlantarFullSamResult:
        """Prepare the sole canvas and segment coherent pressed regions."""

        queued_at = perf_counter()
        with self._inference_lock:
            started_at = perf_counter()
            prepared = self._prepare_canvas(
                image_bytes, side, prepared_input=prepared_input,
            )
            prepared_at = perf_counter()
            prediction = self._predict_pressed_contact(prepared)
            predicted_at = perf_counter()
            result = self._finalize_canvas(prepared, prediction, request_dir)
            finished_at = perf_counter()
            progress_logger.info(
                "[사진 분석][%s 발바닥] 단계별 소요 시간 | 대기=%.3f초 준비=%.3f초 "
                "추론=%.3f초 이미지생성·저장=%.3f초 전체=%.3f초",
                "왼발" if side == "left" else "오른발",
                started_at - queued_at,
                prepared_at - started_at,
                predicted_at - prepared_at,
                finished_at - predicted_at,
                finished_at - queued_at,
            )
            return result


_plantar_foot_detector: YoloFootRoiDetector | None = None
_plantar_foot_detector_key: tuple[object, ...] | None = None
_plantar_foot_detector_load_lock = threading.Lock()
_plantar_foot_detector_inference_lock = threading.RLock()


def _plantar_foot_detector_device() -> str:
    requested = str(settings.plantar_inference_device).strip().lower()
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise PlantarFullSamAnalysisError(
                "PLANTAR_INFERENCE_DEVICE=cuda but CUDA is unavailable."
            )
        return "0"
    return "cpu"


def get_plantar_foot_roi_detector(
    *,
    device: str | None = None,
) -> YoloFootRoiDetector:
    """Return the production sole detector without loading SAM/the decoder.

    Photo capture preflight uses this seam before returning 202. The later
    detector-only plantar analysis receives the same cached detector, so
    validation does not allocate a second copy of the YOLO weights.
    """

    global _plantar_foot_detector, _plantar_foot_detector_key
    checkpoint = Path(weights.plantar_foot_segmentation).expanduser().resolve()
    resolved_device = str(device or _plantar_foot_detector_device())
    key: tuple[object, ...] = (
        str(checkpoint),
        resolved_device,
        float(settings.plantar_foot_roi_confidence),
        int(settings.plantar_foot_roi_image_size),
        int(settings.plantar_foot_roi_min_component_area),
        float(settings.foot_iou_duplicate_threshold),
        0,
        float(settings.plantar_contact_crop_padding_ratio),
        bool(settings.plantar_transform_tta),
        bool(settings.plantar_foot_roi_target_compatible_mode),
    )
    if _plantar_foot_detector is not None and _plantar_foot_detector_key == key:
        return _plantar_foot_detector

    with _plantar_foot_detector_load_lock:
        if _plantar_foot_detector is None or _plantar_foot_detector_key != key:
            try:
                _plantar_foot_detector = YoloFootRoiDetector(
                    checkpoint,
                    FootRoiConfig(
                        confidence=float(settings.plantar_foot_roi_confidence),
                        image_size=int(settings.plantar_foot_roi_image_size),
                        min_component_area=int(
                            settings.plantar_foot_roi_min_component_area
                        ),
                        duplicate_iou_threshold=float(
                            settings.foot_iou_duplicate_threshold
                        ),
                        crop_padding=0,
                        crop_padding_ratio=float(
                            settings.plantar_contact_crop_padding_ratio
                        ),
                        device=resolved_device,
                        cutout_background=False,
                        select_primary_only=True,
                        transform_tta=bool(settings.plantar_transform_tta),
                        target_compatible_mode=bool(
                            settings.plantar_foot_roi_target_compatible_mode
                        ),
                    ),
                )
            except Exception as exc:
                raise PlantarFullSamAnalysisError(
                    "Could not initialize the local plantar foot detector."
                ) from exc
            _plantar_foot_detector_key = key
    assert _plantar_foot_detector is not None
    return _plantar_foot_detector


def extract_plantar_foot_roi(
    image_rgb: np.ndarray,
    *,
    detector: YoloFootRoiDetector | None = None,
) -> FootRoiResult:
    """Run the shared production YOLO sole detector under one inference lock."""

    selected = detector or get_plantar_foot_roi_detector()
    try:
        with _plantar_foot_detector_inference_lock:
            return selected.extract(image_rgb)
    except PlantarFullSamAnalysisError:
        raise
    except Exception as exc:
        raise PlantarFullSamAnalysisError(
            "Local plantar foot detection failed."
        ) from exc


_plantar_contact_segmenter: SegFormerB0PressureSegmenter | None = None
_plantar_contact_segmenter_metadata: dict[str, Any] | None = None
_plantar_contact_segmenter_key: tuple[object, ...] | None = None
_plantar_contact_segmenter_load_lock = threading.Lock()


def get_plantar_contact_segmenter(
    *,
    device: torch.device | str | None = None,
) -> tuple[SegFormerB0PressureSegmenter, dict[str, Any]]:
    """Load and cache the dedicated fine-tuned SegFormer-B0 locally."""

    global _plantar_contact_segmenter
    global _plantar_contact_segmenter_metadata
    global _plantar_contact_segmenter_key

    checkpoint = Path(weights.plantar_contact_segformer).expanduser().resolve()
    if not checkpoint.is_file():
        raise PlantarFullSamAnalysisError(
            f"Required local SegFormer-B0 weight is missing: {checkpoint}"
        )
    resolved_device = (
        torch.device(device)
        if device is not None
        else LocalPlantarFullSamAnalyzer._resolve_device()
    )
    checkpoint_stat = checkpoint.stat()
    key: tuple[object, ...] = (
        str(checkpoint),
        int(checkpoint_stat.st_size),
        int(checkpoint_stat.st_mtime_ns),
        str(resolved_device),
    )
    if (
        _plantar_contact_segmenter is not None
        and _plantar_contact_segmenter_metadata is not None
        and _plantar_contact_segmenter_key == key
    ):
        return _plantar_contact_segmenter, dict(_plantar_contact_segmenter_metadata)

    with _plantar_contact_segmenter_load_lock:
        if (
            _plantar_contact_segmenter is None
            or _plantar_contact_segmenter_metadata is None
            or _plantar_contact_segmenter_key != key
        ):
            try:
                try:
                    state = torch.load(
                        checkpoint,
                        map_location="cpu",
                        weights_only=True,
                    )
                except TypeError:
                    state = torch.load(checkpoint, map_location="cpu")
                if not isinstance(state, Mapping):
                    raise ValueError("checkpoint root must be a mapping")
                if state.get("format_version") != 1:
                    raise ValueError("checkpoint format_version must be 1")
                model_config = state.get("model_config")
                if not isinstance(model_config, Mapping):
                    raise ValueError("checkpoint model_config is missing")
                if model_config.get("architecture") != "segformer_b0":
                    raise ValueError("checkpoint architecture is not segformer_b0")
                if int(model_config.get("input_size", 0)) != 512:
                    raise ValueError("checkpoint input_size must be 512")
                if int(model_config.get("output_size", 0)) != 256:
                    raise ValueError("checkpoint output_size must be 256")
                network_config = model_config.get("network_config")
                if not isinstance(network_config, Mapping):
                    raise ValueError("checkpoint network_config is missing")
                id2label = network_config.get("id2label")
                if not isinstance(id2label, Mapping) or list(
                    str(value).lower() for value in id2label.values()
                ) != ["contact"]:
                    raise ValueError(
                        "checkpoint must expose exactly one 'contact' label"
                    )
                model_state = state.get("model_state_dict")
                if not isinstance(model_state, Mapping):
                    raise ValueError("checkpoint model_state_dict is missing")

                segmenter = SegFormerB0PressureSegmenter(
                    str(model_config.get("pretrained_model", "nvidia/mit-b0")),
                    input_size=int(model_config.get("input_size", 512)),
                    output_size=int(model_config.get("output_size", 256)),
                    pretrained=False,
                    network_config=network_config,
                )
                segmenter.network.load_state_dict(model_state, strict=True)
                segmenter.to(resolved_device)
                segmenter.eval()
                metrics = state.get("metrics")
                metadata = {
                    "architecture": "segformer_b0",
                    "pretrained_model": str(
                        model_config.get("pretrained_model", "nvidia/mit-b0")
                    ),
                    "input_size": int(model_config.get("input_size", 512)),
                    "output_size": int(model_config.get("output_size", 256)),
                    "format_version": state.get("format_version"),
                    "checkpoint_epoch": state.get("epoch"),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "device": str(resolved_device),
                    "validation_iou": (
                        float(metrics["valid_iou"])
                        if isinstance(metrics, Mapping)
                        and "valid_iou" in metrics
                        else None
                    ),
                    "validation_dice": (
                        float(metrics["valid_dice"])
                        if isinstance(metrics, Mapping)
                        and "valid_dice" in metrics
                        else None
                    ),
                    "test_iou": (
                        float(metrics["test_iou"])
                        if isinstance(metrics, Mapping) and "test_iou" in metrics
                        else None
                    ),
                    "test_dice": (
                        float(metrics["test_dice"])
                        if isinstance(metrics, Mapping) and "test_dice" in metrics
                        else None
                    ),
                }
            except PlantarFullSamAnalysisError:
                raise
            except Exception as exc:
                raise PlantarFullSamAnalysisError(
                    "Could not initialize the local SegFormer-B0 contact model: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            _plantar_contact_segmenter = segmenter
            _plantar_contact_segmenter_metadata = metadata
            _plantar_contact_segmenter_key = key

    assert _plantar_contact_segmenter is not None
    assert _plantar_contact_segmenter_metadata is not None
    return _plantar_contact_segmenter, dict(_plantar_contact_segmenter_metadata)


_analyzer: LocalPlantarFullSamAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_plantar_analyzer() -> LocalPlantarFullSamAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = LocalPlantarFullSamAnalyzer()
    return _analyzer


def analyze_plantar_image(
    image_bytes: bytes,
    side: str,
    request_dir: Path,
    *,
    prepared_input: PreparedPlantarFootInput | None = None,
) -> PlantarFullSamResult:
    """Analyze one uploaded side with local models only."""

    return get_plantar_analyzer().analyze(
        image_bytes, side, request_dir, prepared_input=prepared_input,
    )


def combine_transparent_visualizations(
    left_path: Path,
    right_path: Path,
    output_path: Path | None = None,
) -> bytes:
    """Combine two canonical RGBA canvases without flattening transparency."""

    try:
        with Image.open(left_path) as left_image:
            left = np.asarray(left_image.convert("RGBA"), dtype=np.uint8)
        with Image.open(right_path) as right_image:
            right = np.asarray(right_image.convert("RGBA"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise PlantarFullSamAnalysisError(
            "Could not read a transparent plantar visualization."
        ) from exc

    target_height = min(
        max(int(left.shape[0]), int(right.shape[0])),
        int(settings.combined_image_max_height),
    )

    def fit(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image.copy()
        scale = target_height / image.shape[0]
        target_width = max(1, int(round(image.shape[1] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        result = cv2.resize(
            image, (target_width, target_height), interpolation=interpolation
        )
        result[result[..., 3] == 0, :3] = 0
        return result

    left, right = fit(left), fit(right)
    gap = max(0, int(settings.combined_image_gap_pixels))
    combined = np.zeros(
        (target_height, left.shape[1] + gap + right.shape[1], 4),
        dtype=np.uint8,
    )
    combined[:, : left.shape[1]] = left
    combined[:, left.shape[1] + gap :] = right
    combined[combined[..., 3] == 0, :3] = 0
    with BytesIO() as stream:
        Image.fromarray(combined, mode="RGBA").save(stream, format="PNG")
        encoded = stream.getvalue()
    if output_path is not None:
        _write_png(Path(output_path), combined)
    return encoded


def new_request_dir() -> tuple[str, Path]:
    request_id = uuid4().hex
    return request_id, Path(settings.plantar_output_dir) / request_id


__all__ = [
    "PIPELINE_VERSION",
    "LocalPlantarFullSamAnalyzer",
    "PlantarFullSamAnalysisError",
    "PlantarFullSamResult",
    "analyze_plantar_image",
    "combine_transparent_visualizations",
    "extract_plantar_foot_roi",
    "get_plantar_analyzer",
    "get_plantar_contact_segmenter",
    "get_plantar_foot_roi_detector",
    "new_request_dir",
]
