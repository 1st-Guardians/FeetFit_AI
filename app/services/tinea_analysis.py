from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import threading
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights
from app.tinea.segmentation import FungalSegmenter, PREPROCESSING_POLICY
from app.services.wound_analysis import WoundPrediction, get_wound_analyzer
from app.services.tinea_policy import (
    LESION_AREA_SCORE_SENSITIVITY,
    LESION_PRESENT_SAFETY_SCORE,
    TINEA_SIGNAL_SAFETY_SCORE_CUTOFF,
    lesion_safety_score_from_area,
    safety_score_indicates_tinea_signal,
)


os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO  # noqa: E402

progress_logger = logging.getLogger("uvicorn.error")

PHOTO_FUNGAL_COLOR = (255, 0, 0)  # BGR blue
PHOTO_INFLAMMATION_COLOR = (0, 0, 255)  # BGR red
MAP_FUNGAL_COLOR = (255, 220, 155)  # BGR pastel sky blue
MAP_INFLAMMATION_COLOR = (135, 105, 255)  # BGR stronger pink-red
FOOT_OUTLINE_COLOR = (165, 165, 165)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class TineaAnalysisResult:
    suspicion_map_png: bytes
    photo_overlay_png: bytes
    original_filename: str
    fungal_safety_score: int
    skin_reaction_safety_score: int
    metrics: dict
    # Clean photographed foot using the same crop/background policy as the
    # visualizations, but without fungal/inflammation annotations.  Keep all
    # optional fields after the required fields so integrations constructing
    # the result positionally remain source-compatible.
    clean_original_png: bytes = b""
    # In-memory, full-input-canvas mask of the assessed dorsal surface.
    # It is deliberately excluded from repr/equality and is never added to the
    # metrics or persisted tinea image payloads.
    authoritative_fungal_mask: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # Full-input-canvas fungal sigmoid values used exclusively
    # to vary the display opacity inside the authoritative fungal mask. Values
    # outside that mask are exactly zero. This is model confidence for visual
    # emphasis, not a calibrated clinical probability, and is never persisted
    # in report payloads or metrics.
    authoritative_fungal_probability: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # Original EXIF-oriented dimensions; separate from display cutouts.
    wound_mask_png: bytes = b""


def sanitize_filename(filename: str | None) -> str:
    if not filename:
        return "foot.jpg"

    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "foot.jpg"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg not in {"cpu", "cuda"}:
        raise AnalysisError(f"Unsupported analysis device: {device_arg}")
    return device_arg


def load_image(upload_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    try:
        image = Image.open(BytesIO(upload_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise AnalysisError("Uploaded file is not a readable image.") from exc

    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, image_bgr


def load_tinea_model(model_path: Path, device: str) -> tuple[FungalSegmenter, int, float, float]:
    model_path = Path(model_path).resolve()
    if not model_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise AnalysisError("Fungal checkpoint must be inside this project.")
    if not model_path.is_file():
        raise AnalysisError(f"Tinea model not found: {model_path}")
    try:
        model = FungalSegmenter(model_path, device)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise AnalysisError(f"Could not load fungal SegFormer checkpoint: {exc}") from exc
    # The validation-selected checkpoint threshold is authoritative. An old
    # FUNGAL_THRESHOLD environment setting must never override this model.
    inflammation_threshold = settings.inflammation_threshold
    if inflammation_threshold is None:
        inflammation_threshold = 0.90
    return model, model.input_size, model.threshold, float(inflammation_threshold)


def predict_tinea_probs(
    model: FungalSegmenter,
    image: Image.Image,
    sam_input_size: int,
    device: str,
) -> np.ndarray:
    # Retain the service's channel contract: background, fungal, wound.
    # Wound probabilities come exclusively from the independent wound model.
    try:
        probability = model.predict_probability(image)
    except (ValueError, RuntimeError) as exc:
        raise AnalysisError(f"Fungal inference failed: {exc}") from exc
    return np.stack((1.0 - probability, probability, np.zeros_like(probability)), axis=0)


def predict_tinea_probs_with_sliding_window(
    model: FungalSegmenter,
    image: Image.Image,
    sam_input_size: int,
    device: str,
    foot_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    # Keep this entry point for callers of the existing service. The Student
    # was evaluated on foot crops in one pass, with no CLAHE or tile-max fusion.
    if foot_mask.shape != (image.height, image.width):
        raise AnalysisError("Fungal foot mask and original image are misaligned.")
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, padding=0)
    crop_probs = predict_tinea_probs(model, image.crop((x1, y1, x2, y2)), sam_input_size, device)
    probability = np.zeros((image.height, image.width), dtype=np.float32)
    probability[y1:y2, x1:x2] = crop_probs[1]
    probability[foot_mask == 0] = 0.0
    probs = np.stack((1.0 - probability, probability, np.zeros_like(probability)), axis=0)
    metrics = {
        "fungal_input_size": sam_input_size,
        "fungal_preprocessing_policy": PREPROCESSING_POLICY,
        "fungal_inference_passes": 1,
        "fungal_input_roi_xyxy": [x1, y1, x2, y2],
        "fungal_input_roi_policy": "foot_mask_bounding_box_no_padding",
        "fungal_coordinate_space": "exif_oriented_original",
        "fungal_probability_activation": "sigmoid",
        "fungal_threshold_source": "checkpoint",
        "fungal_threshold_selection": model.threshold_selection,
        "fungal_checkpoint": model.checkpoint_path.name,
        "fungal_source_checkpoint_sha256": model.source_checkpoint_sha256,
        "fungal_weight_keys_converted": model.weight_keys_converted,
        "tinea_sliding_window_enabled": False,
        "tinea_sliding_window_applied": False,
        "tinea_sliding_window_tile_count": 0,
        "tinea_preprocess_enhance_enabled": False,
    }
    return probs, metrics


def threshold_predictions(
    probs: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
    *,
    fungal_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fungal_prob = probs[1]
    inflammation_prob = probs[2]
    fungal_mask = fungal_prob >= fungal_threshold
    if fungal_only:
        return fungal_mask, np.zeros_like(fungal_mask, dtype=bool)
    inflammation_mask = inflammation_prob >= inflammation_threshold

    overlap = fungal_mask & inflammation_mask
    if overlap.any():
        fungal_wins = fungal_prob >= inflammation_prob
        fungal_mask = fungal_mask & (~overlap | fungal_wins)
        inflammation_mask = inflammation_mask & (~overlap | ~fungal_wins)

    # Resolve overlap first to preserve the fungal result when the test switch
    # suppresses the inflammation class, including probabilities exactly 1.0.
    if not settings.tinea_inflammation_enabled:
        inflammation_mask = np.zeros_like(inflammation_mask, dtype=bool)
    return fungal_mask, inflammation_mask


def masks_for_visualization(
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    foot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build component-filtered candidates shared by scoring and rendering.

    Previously the suspicion map ignored components smaller than
    ``min_suspicion_area`` while the score and photo overlay still consumed
    them. Filtering once here prevents those consumers from disagreeing.
    """

    foot_region = foot_mask > 0
    min_area = max(1, int(settings.min_suspicion_area))

    def filtered(mask: np.ndarray) -> np.ndarray:
        candidate = np.asarray(mask, dtype=bool) & foot_region
        authoritative = np.zeros_like(candidate, dtype=bool)
        for component in split_connected_components(candidate, min_area):
            authoritative[component > 0] = True
        return authoritative

    return filtered(fungal_mask), filtered(inflammation_mask)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def split_connected_components(binary_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        comp_mask[labels == label_idx] = 1
        components.append(comp_mask)
    return components


def filled_lesion_region(mask: np.ndarray) -> np.ndarray:
    """Include holes enclosed by lesion contours, solely for containment checks."""
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def resolve_fungal_wound_overlap(
    fungal_mask: np.ndarray,
    wound_mask: np.ndarray,
    fungal_probability: np.ndarray,
    wound_probability: np.ndarray,
    foot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Assign each pixel once; enclosed wounds belong to the fungal region.

    Partial overlaps use raw SAM/wound confidence, with fungal winning ties.
    A complete wound component inside a fungal contour belongs to fungal even
    when its wound confidence is greater. This also handles fungal rings with
    a wound in their hole. Only predicted wound pixels are reclassified; other
    holes are not added to either final mask.
    """
    shape = foot_mask.shape
    arrays = (fungal_mask, wound_mask, fungal_probability, wound_probability)
    if len(shape) != 2 or any(array.shape != shape for array in arrays):
        raise AnalysisError("Fungal, wound, probability and foot masks are misaligned.")
    foot = foot_mask > 0
    fungal = np.asarray(fungal_mask, dtype=bool) & foot
    wound = np.asarray(wound_mask, dtype=bool) & foot
    original_overlap = fungal & wound
    contained_pixels = np.zeros(shape, dtype=bool)
    contained_count = 0

    def assign_contained_wounds() -> None:
        nonlocal contained_count
        if not fungal.any() or not wound.any():
            return
        enclosed = filled_lesion_region(fungal) & foot
        # Containment requires a surrounding fungal border. Identical masks
        # or wounds reaching the outer boundary still compete by confidence.
        interior = cv2.erode(
            enclosed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        ) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            wound.astype(np.uint8), connectivity=8,
        )
        for label in range(1, count):
            x, y, width, height = stats[label, :4]
            bounds = np.s_[y:y + height, x:x + width]
            component = labels[bounds] == label
            if interior[bounds][component].all():
                fungal[bounds] |= component
                wound[bounds] &= ~component
                contained_pixels[bounds] |= component
                contained_count += 1

    # Check before competition so a stronger enclosed wound cannot punch a
    # hole in fungal. Check again for enclosed fragments split by competition.
    assign_contained_wounds()
    overlap = fungal & wound
    wound_wins = overlap & (wound_probability > fungal_probability)
    fungal[wound_wins] = False
    wound[overlap & ~wound_wins] = False
    assign_contained_wounds()
    return fungal, wound, {
        "lesion_overlap_policy": "higher_model_confidence_fungal_tie_contained_wound_as_fungal",
        "lesion_overlap_pixels": int(original_overlap.sum()),
        "overlap_fungal_winner_pixels": int((original_overlap & fungal).sum()),
        "overlap_wound_winner_pixels": int((original_overlap & wound).sum()),
        "contained_wound_component_count": contained_count,
        "contained_wound_pixels_reclassified_as_fungal": int(contained_pixels.sum()),
    }


def deduplicate_masks(mask_items: list[dict], iou_threshold: float) -> list[dict]:
    mask_items = sorted(mask_items, key=lambda x: x["score"], reverse=True)
    kept = []
    for item in mask_items:
        if not any(mask_iou(item["mask"], kept_item["mask"]) >= iou_threshold for kept_item in kept):
            kept.append(item)
    return kept


def predict_foot_mask(
    foot_model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    min_component_area: int,
    iou_dup_threshold: float,
) -> tuple[np.ndarray, bool]:
    h, w = image_bgr.shape[:2]
    fallback = np.ones((h, w), dtype=np.uint8)
    results = foot_model.predict(source=image_bgr, conf=conf, save=False, verbose=False)
    if not results:
        return fallback, False

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return fallback, False

    raw_masks = result.masks.data.detach().cpu().numpy()
    if result.boxes is not None and result.boxes.conf is not None:
        scores = result.boxes.conf.detach().cpu().numpy()
    else:
        scores = np.ones(len(raw_masks), dtype=np.float32)

    components = []
    for mask_idx, raw_mask in enumerate(raw_masks):
        score = float(scores[mask_idx]) if mask_idx < len(scores) else 1.0
        mask_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        for comp in split_connected_components(binary_mask, min_component_area):
            components.append({"mask": comp, "score": score})

    final_components = deduplicate_masks(components, iou_dup_threshold)
    if not final_components:
        return fallback, False

    combined = np.zeros((h, w), dtype=np.uint8)
    for item in final_components:
        combined[item["mask"] > 0] = 1
    return combined, True


def validate_precomputed_foot_mask(
    foot_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Normalize a shared upstream foot mask without running YOLO again."""

    mask = np.asarray(foot_mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2 or mask.shape != image_shape:
        raise AnalysisError(
            "Precomputed foot mask must be a 2D array matching the input image."
        )
    if not np.isfinite(mask).all():
        raise AnalysisError("Precomputed foot mask contains non-finite values.")
    binary = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(binary)) < 10:
        raise AnalysisError("Precomputed foot mask is empty or too small.")
    return binary


def score_from_burden(burden: float, sensitivity: float) -> float:
    """Legacy signal gate, kept separate from the displayed area-based score."""
    score = 100.0 * float(np.exp(-sensitivity * max(0.0, burden)))
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def class_score_metrics(
    prob: np.ndarray,
    mask: np.ndarray,
    foot_mask: np.ndarray,
    threshold: float,
    *,
    restrict_confidence_to_mask: bool = False,
) -> dict:
    foot_region = foot_mask > 0
    foot_pixels = int(foot_region.sum())
    if foot_pixels == 0:
        return {
            "pixels": 0,
            "pixel_ratio": 0.0,
            "confidence_weighted_ratio": 0.0,
            "burden": 0.0,
            "score": 100.0,
            "detection_score": 100.0,
        }

    hard_weight = float(np.clip(settings.score_hard_weight, 0.0, 1.0))
    class_region = mask & foot_region
    class_pixels = int(class_region.sum())
    pixel_ratio = class_pixels / foot_pixels

    denom = max(1e-6, 1.0 - threshold)
    confidence_excess = np.clip((prob - threshold) / denom, 0.0, 1.0)
    confidence_region = class_region if restrict_confidence_to_mask else foot_region
    confidence_weighted_ratio = float(confidence_excess[confidence_region].sum() / foot_pixels)
    burden = hard_weight * pixel_ratio + (1.0 - hard_weight) * confidence_weighted_ratio

    return {
        "pixels": class_pixels,
        "pixel_ratio": round(float(pixel_ratio), 6),
        "confidence_weighted_ratio": round(confidence_weighted_ratio, 6),
        "burden": round(float(burden), 6),
        "score": lesion_safety_score_from_area(pixel_ratio),
        "detection_score": score_from_burden(burden, settings.score_sensitivity),
    }


def score_label(score: float) -> str:
    if score >= 90.0:
        return "healthy"
    if score >= 70.0:
        return "mild"
    if score >= 40.0:
        return "moderate"
    return "high_suspicion"


def compute_health_scores(
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    foot_mask: np.ndarray,
    foot_found: bool,
    fungal_threshold: float,
    inflammation_threshold: float,
    *,
    restrict_fungal_confidence_to_mask: bool = False,
) -> dict:
    fungal = class_score_metrics(
        probs[1], fungal_mask, foot_mask, fungal_threshold,
        restrict_confidence_to_mask=restrict_fungal_confidence_to_mask,
    )
    inflammation_enabled = settings.tinea_inflammation_enabled
    if not inflammation_enabled:
        inflammation_mask = np.zeros_like(inflammation_mask, dtype=bool)
    # Excluded or gated-out inflammation must not lower the skin score after
    # its red mask has been removed. Preserve the existing fungal scoring.
    inflammation = class_score_metrics(
        probs[2],
        inflammation_mask,
        foot_mask,
        inflammation_threshold,
        restrict_confidence_to_mask=True,
    )
    overall_score = round(min(fungal["score"], inflammation["score"]), 2)

    return {
        "foot_area_pixels": int((foot_mask > 0).sum()),
        "score_reliability": "normal" if foot_found else "low_full_image_fallback",
        "fungal_pixels_in_foot": fungal["pixels"],
        "fungal_pixel_ratio": fungal["pixel_ratio"],
        "fungal_confidence_weighted_ratio": fungal["confidence_weighted_ratio"],
        "fungal_burden": fungal["burden"],
        "fungal_score": fungal["score"],
        "fungal_detection_score": fungal["detection_score"],
        "fungal_score_label": score_label(fungal["score"]),
        "inflammation_pixels_in_foot": inflammation["pixels"],
        "inflammation_pixel_ratio": inflammation["pixel_ratio"],
        "inflammation_confidence_weighted_ratio": inflammation["confidence_weighted_ratio"],
        "inflammation_burden": inflammation["burden"],
        "inflammation_score": inflammation["score"],
        "inflammation_detection_score": inflammation["detection_score"],
        "inflammation_score_label": (
            score_label(inflammation["score"])
            if inflammation_enabled else "not_analyzed_forced_clear"
        ),
        "inflammation_analysis_enabled": inflammation_enabled,
        "inflammation_analysis_mode": (
            "enabled" if inflammation_enabled else "disabled_for_testing"
        ),
        "overall_health_score": overall_score,
        "overall_health_label": score_label(overall_score),
    }


def build_authoritative_analysis_masks(
    *,
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    foot_mask: np.ndarray,
    foot_found: bool,
    fungal_threshold: float,
    inflammation_threshold: float,
    restrict_fungal_confidence_to_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Return the only class masks allowed to drive scores or visual output.

    Thresholded components first pass the shared minimum-area filter and the
    existing confidence/area signal gate. The new displayed score is separate
    from that gate, so scoring changes do not change fungal detection. Display
    scores are recomputed from the accepted masks used by images and reports.
    """

    # Also gate precomputed candidate masks; callers cannot bypass the switch
    # by providing masks that were thresholded before testing was disabled.
    if not settings.tinea_inflammation_enabled:
        inflammation_mask = np.zeros_like(inflammation_mask, dtype=bool)
    candidate_fungal, candidate_inflammation = masks_for_visualization(
        fungal_mask,
        inflammation_mask,
        foot_mask,
    )
    candidate_metrics = compute_health_scores(
        probs=probs,
        fungal_mask=candidate_fungal,
        inflammation_mask=candidate_inflammation,
        foot_mask=foot_mask,
        foot_found=foot_found,
        fungal_threshold=fungal_threshold,
        inflammation_threshold=inflammation_threshold,
        restrict_fungal_confidence_to_mask=restrict_fungal_confidence_to_mask,
    )
    fungal_signal_detected = bool(candidate_fungal.any()) and safety_score_indicates_tinea_signal(
        candidate_metrics["fungal_detection_score"]
    )
    inflammation_signal_detected = bool(candidate_inflammation.any()) and safety_score_indicates_tinea_signal(
        candidate_metrics["inflammation_detection_score"]
    )

    authoritative_fungal = (
        candidate_fungal
        if fungal_signal_detected
        else np.zeros_like(candidate_fungal, dtype=bool)
    )
    authoritative_inflammation = (
        candidate_inflammation
        if inflammation_signal_detected
        else np.zeros_like(candidate_inflammation, dtype=bool)
    )
    final_metrics = compute_health_scores(
        probs=probs,
        fungal_mask=authoritative_fungal,
        inflammation_mask=authoritative_inflammation,
        foot_mask=foot_mask,
        foot_found=foot_found,
        fungal_threshold=fungal_threshold,
        inflammation_threshold=inflammation_threshold,
        restrict_fungal_confidence_to_mask=restrict_fungal_confidence_to_mask,
    )
    gate_metrics = {
        "signal_safety_score_cutoff": TINEA_SIGNAL_SAFETY_SCORE_CUTOFF,
        "signal_score_source": "legacy_confidence_area_detection_score",
        "fungal_candidate_pixels": int(candidate_fungal.sum()),
        "inflammation_candidate_pixels": int(candidate_inflammation.sum()),
        "fungal_candidate_safety_score": candidate_metrics["fungal_score"],
        "inflammation_candidate_safety_score": candidate_metrics[
            "inflammation_score"
        ],
        "fungal_candidate_detection_score": candidate_metrics["fungal_detection_score"],
        "inflammation_candidate_detection_score": candidate_metrics["inflammation_detection_score"],
        "lesion_present_safety_score": LESION_PRESENT_SAFETY_SCORE,
        "lesion_area_score_sensitivity": LESION_AREA_SCORE_SENSITIVITY,
        "fungal_signal_detected": fungal_signal_detected,
        "inflammation_signal_detected": inflammation_signal_detected,
        "inflammation_analysis_enabled": settings.tinea_inflammation_enabled,
        "inflammation_analysis_mode": (
            "enabled" if settings.tinea_inflammation_enabled else "disabled_for_testing"
        ),
    }
    return (
        authoritative_fungal,
        authoritative_inflammation,
        final_metrics,
        gate_metrics,
    )


def build_wound_skin_metrics(
    prediction: WoundPrediction,
    foot_mask: np.ndarray,
    foot_found: bool,
) -> tuple[np.ndarray, dict]:
    """Score the final wound mask inside the foot, without legacy score gating."""
    if prediction.mask.shape != foot_mask.shape:
        raise AnalysisError("Wound mask and original-space foot mask are misaligned.")
    mask = prediction.mask & (foot_mask > 0)
    threshold = float(prediction.metrics["wound_threshold"])
    values = class_score_metrics(
        prediction.probability, mask, foot_mask, threshold,
        restrict_confidence_to_mask=True,
    )
    detected = bool(mask.any())
    return mask, {
        **prediction.metrics,
        "inflammation_pixels_in_foot": values["pixels"],
        "inflammation_pixels": values["pixels"],
        "inflammation_pixel_ratio": values["pixel_ratio"],
        "inflammation_confidence_weighted_ratio": values["confidence_weighted_ratio"],
        "inflammation_burden": values["burden"],
        "inflammation_score": values["score"],
        "inflammation_score_label": "wound_detected" if detected else "no_wound_detected",
        "inflammation_analysis_enabled": True,
        "inflammation_analysis_mode": "wound_segformer_single_pass",
        "inflammation_threshold": threshold,
        "inflammation_candidate_pixels": values["pixels"],
        "inflammation_candidate_safety_score": values["score"],
        "inflammation_signal_detected": detected,
        "wound_detected": detected,
        "wound_pixels_in_foot": values["pixels"],
        "wound_area_ratio": values["pixel_ratio"],
        "wound_confidence_weighted_ratio": values["confidence_weighted_ratio"],
        "wound_safety_score": values["score"],
        "wound_score_formula": "100 if no final lesion else 65 * exp(-7 * final_area_ratio)",
        "wound_score_present_baseline": LESION_PRESENT_SAFETY_SCORE,
        "wound_score_sensitivity": LESION_AREA_SCORE_SENSITIVITY,
        "wound_mask_policy": "final_probability_threshold_intersect_original_foot_only",
        "score_reliability": "normal" if foot_found else "low_full_image_fallback",
        "foot_area_pixels": int((foot_mask > 0).sum()),
    }


def max_probability_in_mask(prob: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(mask, dtype=bool)
    if not region.any():
        return 0.0
    return float(np.asarray(prob)[region].max())


def confidence_to_alpha(confidence: float, threshold: float, min_alpha: float, max_alpha: float) -> float:
    denom = max(1e-6, 1.0 - threshold)
    scaled = np.clip((confidence - threshold) / denom, 0.0, 1.0)
    return float(min_alpha + scaled * (max_alpha - min_alpha))


def blend_mask_by_confidence(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
    min_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    if not mask.any():
        return image_bgr

    out = image_bgr.copy()
    denom = max(1e-6, 1.0 - threshold)
    alpha = min_alpha + np.clip((prob - threshold) / denom, 0.0, 1.0) * (max_alpha - min_alpha)
    color_arr = np.zeros_like(out, dtype=np.float32)
    color_arr[:, :] = np.array(color, dtype=np.float32)

    mask_3 = mask[:, :, None]
    blended = out.astype(np.float32) * (1.0 - alpha[:, :, None]) + color_arr * alpha[:, :, None]
    out[mask_3.repeat(3, axis=2)] = blended[mask_3.repeat(3, axis=2)].astype(np.uint8)
    return out


def render_photo_overlay(
    image_bgr: np.ndarray,
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay = blend_mask_by_confidence(
        overlay,
        fungal_mask,
        probs[1],
        fungal_threshold,
        PHOTO_FUNGAL_COLOR,
        min_alpha=0.28,
        max_alpha=0.68,
    )
    overlay = blend_mask_by_confidence(
        overlay,
        inflammation_mask,
        probs[2],
        inflammation_threshold,
        PHOTO_INFLAMMATION_COLOR,
        min_alpha=0.34,
        max_alpha=0.78,
    )

    contour_thickness = max(3, settings.suspicion_line_thickness + 1)
    for mask, color in [
        (fungal_mask, PHOTO_FUNGAL_COLOR),
        (inflammation_mask, PHOTO_INFLAMMATION_COLOR),
    ]:
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, color, thickness=contour_thickness, lineType=cv2.LINE_AA)

    return overlay


def apply_cutout_background(
    image_bgr: np.ndarray,
    foot_mask: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    mask = (foot_mask > 0).astype(np.float32)
    if mask.all():
        return image_bgr

    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    background = np.full_like(image_bgr, background_color, dtype=np.uint8)
    cutout = image_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return cutout.astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y2 = min(h, int(ys.max()) + padding + 1)
    return x1, y1, x2, y2


def apply_cutout_alpha(image_bgr: np.ndarray, foot_mask: np.ndarray) -> np.ndarray:
    alpha = (foot_mask > 0).astype(np.float32) * 255.0
    if not alpha.all():
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0, sigmaY=1.0)

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return np.dstack([image_bgr, alpha])


def crop_to_foot(image: np.ndarray, foot_mask: np.ndarray, padding: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, padding)
    return image[y1:y2, x1:x2].copy()


def _toe_names_left_to_right(foot_side: str | None) -> list[str]:
    if foot_side == "left":
        return ["새끼발가락", "넷째 발가락", "셋째 발가락", "둘째 발가락", "엄지발가락"]
    if foot_side == "right":
        return ["엄지발가락", "둘째 발가락", "셋째 발가락", "넷째 발가락", "새끼발가락"]
    return ["가쪽 발가락", "넷째 발가락", "가운데 발가락", "둘째 발가락", "안쪽 발가락"]


def _horizontal_region_label(x_norm: float) -> str:
    if x_norm < 0.33:
        return "왼쪽"
    if x_norm > 0.67:
        return "오른쪽"
    return "중앙"


def _toe_region_label(foot_side: str | None, x_norm: float, x_min_norm: float, x_max_norm: float) -> str:
    names = _toe_names_left_to_right(foot_side)
    center_idx = int(np.clip(np.floor(x_norm * len(names)), 0, len(names) - 1))
    start_idx = int(np.clip(np.floor(x_min_norm * len(names)), 0, len(names) - 1))
    end_idx = int(np.clip(np.floor(x_max_norm * len(names)), 0, len(names) - 1))

    if end_idx > start_idx:
        span = names[start_idx : end_idx + 1]
        if len(span) == 2:
            return f"{span[0]}과 {span[1]} 사이"
        return f"{span[0]}~{span[-1]} 주변"
    return f"{names[center_idx]} 주변"


def _component_location_label(
    component: np.ndarray,
    foot_mask: np.ndarray,
    foot_side: str | None,
    foot_surface: str = "dorsal",
) -> str:
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, 0)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    ys, xs = np.where(component)
    if xs.size == 0 or ys.size == 0:
        return "발 영역"

    cx = float(xs.mean())
    cy = float(ys.mean())
    x_norm = float(np.clip((cx - x1) / width, 0.0, 0.999))
    y_norm = float(np.clip((cy - y1) / height, 0.0, 0.999))
    x_min_norm = float(np.clip((float(xs.min()) - x1) / width, 0.0, 0.999))
    x_max_norm = float(np.clip((float(xs.max()) - x1) / width, 0.0, 0.999))

    if y_norm < 0.34:
        return _toe_region_label(foot_side, x_norm, x_min_norm, x_max_norm)
    if y_norm < 0.52:
        surface_label = "앞꿈치/발바닥" if foot_surface == "plantar" else "앞발/발등"
        return f"{_horizontal_region_label(x_norm)} {surface_label}"
    if y_norm < 0.76:
        surface_label = "발바닥 중앙부" if foot_surface == "plantar" else "발 중앙부"
        return f"{_horizontal_region_label(x_norm)} {surface_label}"
    return f"{_horizontal_region_label(x_norm)} 뒤꿈치 주변"


def summarize_suspicion_regions(
    mask: np.ndarray,
    prob: np.ndarray,
    foot_mask: np.ndarray,
    threshold: float,
    foot_side: str | None,
    foot_surface: str = "dorsal",
    max_regions: int = 5,
    *,
    min_area_pixels: int | None = None,
) -> list[dict]:
    foot_region = foot_mask > 0
    foot_pixels = int(foot_region.sum())
    if foot_pixels == 0:
        return []

    regions = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask & foot_region).astype(np.uint8), 8)
    min_area = max(1, int(settings.min_suspicion_area if min_area_pixels is None else min_area_pixels))
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == label_idx
        ys, xs = np.where(component)
        if xs.size == 0 or ys.size == 0:
            continue

        values = prob[ys, xs]
        weights = np.clip(values - threshold, 0.0, None) + 1e-3
        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))
        x1, y1, x2, y2 = bbox_from_mask(foot_mask, 0)
        regions.append(
            {
                "location": _component_location_label(
                    component,
                    foot_mask,
                    foot_side,
                    foot_surface,
                ),
                "area_pixels": area,
                "foot_area_ratio": round(float(area / foot_pixels), 6),
                "confidence_p90": round(float(np.percentile(values, 90)), 4),
                "center_x_ratio": round(float(np.clip((cx - x1) / max(1, x2 - x1), 0.0, 1.0)), 4),
                "center_y_ratio": round(float(np.clip((cy - y1) / max(1, y2 - y1), 0.0, 1.0)), 4),
            }
        )

    regions.sort(key=lambda item: (item["area_pixels"], item["confidence_p90"]), reverse=True)
    return regions[:max_regions]


def render_hallux_style_png(image_bgr: np.ndarray, foot_mask: np.ndarray) -> bytes:
    cutout = apply_cutout_alpha(image_bgr, foot_mask)
    cropped = crop_to_foot(cutout, foot_mask, settings.photo_cutout_padding)
    return encode_png(cropped)


def circle_from_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    circle_scale: float,
    min_radius: int,
    max_radius: int,
    layer: int,
    level: float,
) -> dict | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    values = prob[ys, xs]
    weights_for_center = np.clip(values - threshold, 0.0, None) + 1e-3
    cx = float(np.average(xs, weights=weights_for_center))
    cy = float(np.average(ys, weights=weights_for_center))
    area = int(len(xs))
    area_radius = np.sqrt(area / np.pi) * circle_scale
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    spread_radius = float(np.percentile(distances, 70)) * 0.96 if distances.size else 0.0
    radius = int(round(max(area_radius, spread_radius)))
    radius = int(np.clip(radius, min_radius, max_radius))

    return {
        "center": (int(round(cx)), int(round(cy))),
        "radius": radius,
        "confidence": float(np.percentile(values, 90)),
        "threshold": threshold,
        "level": float(level),
        "layer": layer,
        "area": area,
    }


def hierarchical_class_circles(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> list[dict]:
    if not mask.any():
        return []

    area = int(mask.sum())
    if area < settings.min_suspicion_area:
        return []
    if area <= settings.max_dot_area_for_suspicion_map:
        return []

    outer = circle_from_mask(
        mask=mask,
        prob=prob,
        threshold=threshold,
        circle_scale=settings.circle_scale,
        min_radius=settings.min_circle_radius,
        max_radius=settings.max_circle_radius,
        layer=0,
        level=threshold,
    )
    if outer is None:
        return []

    circles = [outer]
    inner_level = max(threshold, float(np.percentile(prob[mask], settings.inner_confidence_percentile)))
    if inner_level < threshold + 0.03:
        return circles

    inner_mask = mask & (prob >= inner_level)
    inner_area = int(inner_mask.sum())
    min_inner_area = max(settings.min_suspicion_area, int(round(area * settings.inner_min_area_ratio)))
    if inner_area < min_inner_area:
        return circles

    inner = circle_from_mask(
        mask=inner_mask,
        prob=prob,
        threshold=threshold,
        circle_scale=settings.circle_scale,
        min_radius=2,
        max_radius=max(2, int(round(outer["radius"] * 0.58))),
        layer=1,
        level=inner_level,
    )
    if inner is None:
        return circles

    outer_cx, outer_cy = outer["center"]
    inner_cx, inner_cy = inner["center"]
    center_distance = float(np.hypot(inner_cx - outer_cx, inner_cy - outer_cy))
    max_inner_radius = int(np.floor(outer["radius"] - center_distance - 1))
    if max_inner_radius < 2:
        inner["center"] = outer["center"]
        max_inner_radius = max(2, int(round(outer["radius"] * 0.52)))

    inner["radius"] = int(min(inner["radius"], max_inner_radius))
    if inner["radius"] >= 2:
        circles.append(inner)
    return circles


def blend_circle(
    canvas: np.ndarray,
    foot_mask: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = canvas.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    circle_mask[foot_mask == 0] = 0

    if not circle_mask.any():
        return

    color_img = np.zeros_like(canvas, dtype=np.uint8)
    color_img[:, :] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(canvas, 1.0 - alpha, color_img, alpha, 0.0)
    canvas[circle_mask > 0] = blended[circle_mask > 0]


def circle_contains_point(circle: dict, point: tuple[int, int], margin: float = 0.92) -> bool:
    cx, cy = circle["center"]
    px, py = point
    return float(np.hypot(px - cx, py - cy)) <= float(circle["radius"]) * margin


def evidence_dots_from_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    existing_circles: list[dict],
    color: tuple[int, int, int],
) -> list[dict]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    dots = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = labels == label_idx
        ys, xs = np.where(component)
        values = prob[ys, xs]
        weights = np.clip(values - threshold, 0.0, None) + 1e-3
        center = (int(round(float(np.average(xs, weights=weights)))), int(round(float(np.average(ys, weights=weights)))))

        if any(circle_contains_point(circle, center) for circle in existing_circles):
            continue

        radius = int(np.clip(round(np.sqrt(area / np.pi) * 1.4), 3, 7))
        confidence = float(np.percentile(values, 90))
        alpha = confidence_to_alpha(confidence, threshold, min_alpha=0.42, max_alpha=0.68)
        dots.append(
            {
                "center": center,
                "radius": radius,
                "color": color,
                "alpha": alpha,
                "area": area,
                "confidence": confidence,
            }
        )

    dots.sort(key=lambda item: (item["area"], item["confidence"]), reverse=True)
    return dots


def connected_components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label_idx
        components.append(component)

    components.sort(key=lambda component: int(component.sum()), reverse=True)
    return components


def circles_and_dots_from_overlay_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
) -> tuple[list[dict], list[dict]]:
    circles = []
    dots = []

    for component in connected_components_from_mask(mask):
        area = int(component.sum())
        if area < settings.min_suspicion_area:
            continue

        if area > settings.max_dot_area_for_suspicion_map and not circles:
            component_circles = hierarchical_class_circles(component, prob, threshold)
            for circle in component_circles:
                circle["color"] = color
                circles.append(circle)
            if component_circles:
                continue

        dots.extend(evidence_dots_from_mask(component, prob, threshold, circles, color))

    return circles, dots


def render_suspicion_map(
    image_shape: tuple[int, int],
    foot_mask: np.ndarray,
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> tuple[np.ndarray, int, int]:
    h, w = image_shape
    canvas = np.full((h, w, 3), (250, 253, 255), dtype=np.uint8)
    canvas[foot_mask > 0] = (255, 255, 255)
    # Circular markers may extend beyond the underlying lesion. Clip them
    # so an adjacent wound marker cannot paint inside a fungal enclosure.
    class_support = {
        MAP_FUNGAL_COLOR: (foot_mask > 0) & ~inflammation_mask,
        MAP_INFLAMMATION_COLOR: (foot_mask > 0) & ~filled_lesion_region(fungal_mask),
    }

    circles = []
    evidence_dots = []
    for mask, prob, threshold, color in [
        (fungal_mask, probs[1], fungal_threshold, MAP_FUNGAL_COLOR),
        (inflammation_mask, probs[2], inflammation_threshold, MAP_INFLAMMATION_COLOR),
    ]:
        draw_mask = mask & (foot_mask > 0)
        class_circles, class_dots = circles_and_dots_from_overlay_mask(draw_mask, prob, threshold, color)
        circles.extend(class_circles)
        evidence_dots.extend(class_dots)

    circles.sort(key=lambda item: (item["layer"], -item["radius"]))
    for circle in circles:
        if circle["layer"] == 0:
            alpha = confidence_to_alpha(circle["confidence"], circle["threshold"], min_alpha=0.16, max_alpha=0.34)
        else:
            alpha = confidence_to_alpha(circle["confidence"], circle["threshold"], min_alpha=0.32, max_alpha=0.58)
        blend_circle(
            canvas=canvas,
            foot_mask=class_support[circle["color"]],
            center=circle["center"],
            radius=circle["radius"],
            color=circle["color"],
            alpha=alpha,
        )

    for dot in evidence_dots:
        blend_circle(
            canvas=canvas,
            foot_mask=class_support[dot["color"]],
            center=dot["center"],
            radius=dot["radius"],
            color=dot["color"],
            alpha=dot["alpha"],
        )

    foot_contours, _ = cv2.findContours(foot_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if foot_contours:
        cv2.drawContours(
            canvas,
            foot_contours,
            -1,
            FOOT_OUTLINE_COLOR,
            thickness=settings.suspicion_line_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas, len(circles), len(evidence_dots)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise AnalysisError("Failed to encode analysis image.")
    return encoded.tobytes()


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise AnalysisError("Failed to encode analysis image.")
    return encoded.tobytes()


def build_forced_clear_plantar_tinea_result(
    image_bytes: bytes,
    filename: str | None,
    foot_side: str | None,
    *,
    foot_mask: np.ndarray,
    foot_mask_source: str = "plantar_yolo_sole_mask",
) -> TineaAnalysisResult:
    """Plantar dermatology is excluded; no model or safety score is produced."""
    raise AnalysisError("Plantar skin assessment is disabled; use dorsal photographs.")


def build_plantar_skin_result(
    image_bytes: bytes,
    filename: str | None,
    foot_side: str | None,
    *,
    foot_mask: np.ndarray,
    foot_mask_source: str = "plantar_yolo_sole_mask",
    prepared_input=None,
) -> TineaAnalysisResult:
    """Plantar dermatology is excluded; no model or safety score is produced."""
    raise AnalysisError("Plantar skin assessment is disabled; use dorsal photographs.")


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AnalysisError("Failed to decode analysis image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    return image


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_height:
        return image

    scale = target_height / h
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def ensure_bgra(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        return image
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return np.dstack([image, alpha])
    raise AnalysisError(f"Unsupported image channel count: {image.shape[2]}")


def combine_png_images_side_by_side(
    left_image: bytes,
    right_image: bytes,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    left = decode_image(left_image)
    right = decode_image(right_image)
    use_alpha = (left.ndim == 3 and left.shape[2] == 4) or (right.ndim == 3 and right.shape[2] == 4)
    if use_alpha:
        left = ensure_bgra(left)
        right = ensure_bgra(right)

    target_height = min(max(left.shape[0], right.shape[0]), settings.combined_image_max_height)
    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    gap = max(0, settings.combined_image_gap_pixels)
    combined_height = target_height
    combined_width = left.shape[1] + gap + right.shape[1]
    if use_alpha:
        canvas = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)
        canvas[:, :, :3] = np.array(background_color, dtype=np.uint8)
    else:
        canvas = np.full((combined_height, combined_width, 3), background_color, dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    return encode_png(canvas)


class TineaAnalyzer:
    def __init__(self) -> None:
        self.device = resolve_device(settings.analysis_device)
        self.tinea_model, self.sam_input_size, self.fungal_threshold, self.inflammation_threshold = load_tinea_model(
            weights.tinea_pedis_segmentation,
            self.device,
        )
        # Compatibility name retained for integrations; this is now 384.
        self.input_size = self.sam_input_size
        self.foot_model = None
        self._foot_model_lock = threading.Lock()

    def get_foot_model(self):
        if self.foot_model is not None:
            return self.foot_model
        with self._foot_model_lock:
            if self.foot_model is None:
                if not weights.foot_segmentation.exists():
                    raise AnalysisError(
                        f"Foot segmentation model not found: {weights.foot_segmentation}"
                    )
                self.foot_model = YOLO(str(weights.foot_segmentation))
        return self.foot_model

    def analyze(
        self,
        image_bytes: bytes,
        filename: str | None,
        foot_side: str | None = None,
        *,
        foot_mask: np.ndarray | None = None,
        foot_surface: str = "dorsal",
        foot_mask_source: str | None = None,
    ) -> TineaAnalysisResult:
        normalized_surface = str(foot_surface).strip().lower()
        if normalized_surface != "dorsal":
            raise AnalysisError("Only dorsal skin assessment is enabled.")
        pil_image, image_bgr = load_image(image_bytes)
        h, w = image_bgr.shape[:2]
        if foot_mask is None:
            analyzed_foot_mask, foot_found = predict_foot_mask(
                foot_model=self.get_foot_model(),
                image_bgr=image_bgr,
                conf=settings.foot_confidence,
                min_component_area=settings.foot_min_component_area,
                iou_dup_threshold=settings.foot_iou_duplicate_threshold,
            )
            mask_source = foot_mask_source or "tinea_yolo"
        else:
            analyzed_foot_mask = validate_precomputed_foot_mask(
                foot_mask,
                (h, w),
            )
            foot_found = True
            mask_source = foot_mask_source or "shared_aruco_segmentation"
        probs, inference_metrics = predict_tinea_probs_with_sliding_window(
            self.tinea_model,
            pil_image,
            self.sam_input_size,
            self.device,
            analyzed_foot_mask,
        )
        fungal_mask, inflammation_mask = threshold_predictions(
            probs,
            self.fungal_threshold,
            self.inflammation_threshold,
            fungal_only=True,
        )
        # The binary Student supplies fungal sigmoid probabilities only.
        # The independent wound model supplies the skin reaction channel.
        probs = probs.copy()
        probs[2] = 0.0
        use_wound = settings.tinea_inflammation_enabled and settings.wound_segmentation_enabled
        (
            authoritative_fungal_mask,
            authoritative_inflammation_mask,
            metrics,
            gate_metrics,
        ) = build_authoritative_analysis_masks(
            probs=probs,
            fungal_mask=fungal_mask,
            inflammation_mask=inflammation_mask,
            foot_mask=analyzed_foot_mask,
            foot_found=foot_found,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=self.inflammation_threshold,
        )
        skin_threshold = self.inflammation_threshold
        if use_wound:
            prediction = get_wound_analyzer().analyze(pil_image)
            if prediction.mask.shape != analyzed_foot_mask.shape:
                raise AnalysisError("Wound mask and original-space foot mask are misaligned.")
            authoritative_fungal_mask, resolved_wound_mask, overlap_metrics = resolve_fungal_wound_overlap(
                authoritative_fungal_mask, prediction.mask,
                probs[1], prediction.probability, analyzed_foot_mask,
            )
            if (overlap_metrics["lesion_overlap_pixels"]
                    or overlap_metrics["contained_wound_pixels_reclassified_as_fungal"]):
                # Preserve the fungal component/score gate, but remove every
                # confidence contribution from pixels assigned to wound.
                authoritative_fungal_mask, _, metrics, gate_metrics = build_authoritative_analysis_masks(
                    probs=probs,
                    fungal_mask=authoritative_fungal_mask,
                    inflammation_mask=np.zeros_like(resolved_wound_mask),
                    foot_mask=analyzed_foot_mask,
                    foot_found=foot_found,
                    fungal_threshold=self.fungal_threshold,
                    inflammation_threshold=self.inflammation_threshold,
                    restrict_fungal_confidence_to_mask=True,
                )
            authoritative_inflammation_mask, wound_metrics = build_wound_skin_metrics(
                replace(prediction, mask=resolved_wound_mask), analyzed_foot_mask, foot_found,
            )
            wound_metrics["wound_mask_policy"] = "final_threshold_foot_intersection_then_fungal_wound_arbitration"
            probs = probs.copy()
            probs[2] = prediction.probability
            metrics.update(wound_metrics)
            metrics.update(overlap_metrics)
            gate_metrics.update({key: value for key, value in wound_metrics.items()
                                 if key.startswith("inflammation_")})
            skin_threshold = float(wound_metrics["inflammation_threshold"])
            overall = min(metrics["fungal_score"], metrics["inflammation_score"])
            metrics.update(overall_health_score=overall, overall_health_label=score_label(overall))
        else:
            disabled_skin = {
                "inflammation_analysis_enabled": False,
                "inflammation_analysis_mode": "disabled_for_testing",
                "inflammation_score_label": "not_analyzed_forced_clear",
            }
            metrics.update(disabled_skin)
            gate_metrics.update(disabled_skin)
        suspicion_map, circle_count, evidence_dot_count = render_suspicion_map(
            image_shape=(h, w),
            foot_mask=analyzed_foot_mask,
            probs=probs,
            fungal_mask=authoritative_fungal_mask,
            inflammation_mask=authoritative_inflammation_mask,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=skin_threshold,
        )
        photo_overlay = render_photo_overlay(
            image_bgr=image_bgr,
            probs=probs,
            fungal_mask=authoritative_fungal_mask,
            inflammation_mask=authoritative_inflammation_mask,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=skin_threshold,
        )
        display_rotation = 90 if normalized_surface == "plantar" else 0
        display_foot_mask = analyzed_foot_mask
        if display_rotation:
            # Turn each plantar display before side-by-side composition.
            # Scores, region coordinates and native exported masks still use
            # the EXIF-oriented original photograph's coordinate system.
            suspicion_map = cv2.rotate(suspicion_map, cv2.ROTATE_90_CLOCKWISE)
            photo_overlay = cv2.rotate(photo_overlay, cv2.ROTATE_90_CLOCKWISE)
            display_foot_mask = cv2.rotate(analyzed_foot_mask, cv2.ROTATE_90_CLOCKWISE)
        if settings.photo_cutout_background:
            suspicion_map_png = render_hallux_style_png(
                suspicion_map, display_foot_mask
            )
            photo_overlay_png = render_hallux_style_png(
                photo_overlay, display_foot_mask
            )
            clean_original_png = render_hallux_style_png(
                image_bgr, analyzed_foot_mask
            )
        else:
            suspicion_map_png = encode_png(suspicion_map)
            photo_overlay_png = encode_png(photo_overlay)
            clean_original_png = encode_png(image_bgr)
        metrics.update(
            {
                "foot_outline_found": foot_found,
                "foot_mask_source": mask_source,
                "analysis_mode": "segformer_fungal_only_segformer_wound" if use_wound else "segformer_fungal_only",
                "model_inference_executed": True,
                "fungal_analysis_enabled": True,
                "fungal_model": "segformer_b2_teacher_student",
                "fungal_active_classes": ["fungal"],
                "sam_inflammation_class_used": False,
                **inference_metrics,
                **gate_metrics,
                "fungal_pixels": int(authoritative_fungal_mask.sum()),
                "inflammation_pixels": int(
                    authoritative_inflammation_mask.sum()
                ),
                "circle_count": circle_count,
                "evidence_dot_count": evidence_dot_count,
                "fungal_threshold": self.fungal_threshold,
                "inflammation_threshold": skin_threshold,
                "suspicion_map_source": "authoritative_classification_gated_mask",
                "suspicious_area_map_visualization": "circle_suspicion_map_with_evidence_dots",
                "original_foot_image_visualization": (
                    "clean_photo_transparent_cutout_crop"
                    if settings.photo_cutout_background
                    else "clean_photo"
                ),
                "photo_overlay_visualization": (
                    "fungal_blue_inflammation_red_transparent_cutout_crop"
                    if settings.photo_cutout_background
                    else "fungal_blue_inflammation_red"
                ),
                "photo_overlay_rotation_degrees_clockwise": display_rotation,
                "suspicion_map_rotation_degrees_clockwise": display_rotation,
                "max_fungal_prob": max_probability_in_mask(
                    probs[1], authoritative_fungal_mask
                ),
                "max_inflammation_prob": max_probability_in_mask(
                    probs[2], authoritative_inflammation_mask
                ),
                "raw_max_fungal_prob_in_foot": max_probability_in_mask(
                    probs[1], analyzed_foot_mask
                ),
                "raw_max_inflammation_prob_in_foot": max_probability_in_mask(
                    probs[2], analyzed_foot_mask
                ) if settings.tinea_inflammation_enabled else None,
                "foot_side": foot_side or "unknown",
                "foot_surface": normalized_surface,
                "fungal_regions": summarize_suspicion_regions(
                    authoritative_fungal_mask,
                    probs[1],
                    analyzed_foot_mask,
                    self.fungal_threshold,
                    foot_side,
                    normalized_surface,
                ),
                "inflammation_regions": summarize_suspicion_regions(
                    authoritative_inflammation_mask,
                    probs[2],
                    analyzed_foot_mask,
                    skin_threshold,
                    foot_side,
                    normalized_surface,
                    min_area_pixels=1 if use_wound else None,
                ),
            }
        )

        exported_fungal_mask = (
            np.ascontiguousarray(
                authoritative_fungal_mask,
                dtype=np.bool_,
            ).copy()
        )
        exported_fungal_probability = (
            np.ascontiguousarray(
                np.where(
                    authoritative_fungal_mask,
                    np.clip(probs[1], 0.0, 1.0),
                    0.0,
                ),
                dtype=np.float32,
            ).copy()
        )
        return TineaAnalysisResult(
            suspicion_map_png=suspicion_map_png,
            photo_overlay_png=photo_overlay_png,
            original_filename=sanitize_filename(filename),
            fungal_safety_score=int(round(float(metrics["fungal_score"]))),
            skin_reaction_safety_score=int(round(float(metrics["inflammation_score"]))),
            metrics=metrics,
            clean_original_png=clean_original_png,
            wound_mask_png=encode_png(authoritative_inflammation_mask.astype(np.uint8) * 255) if use_wound else b"",
            authoritative_fungal_mask=exported_fungal_mask,
            authoritative_fungal_probability=exported_fungal_probability,
        )


_analyzer: TineaAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_tinea_analyzer() -> TineaAnalyzer:
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    with _analyzer_lock:
        if _analyzer is None:
            _analyzer = TineaAnalyzer()
    return _analyzer


def analyze_foot_image(
    image_bytes: bytes,
    filename: str | None,
    foot_side: str | None = None,
    *,
    foot_mask: np.ndarray | None = None,
    foot_surface: str = "dorsal",
    foot_mask_source: str | None = None,
) -> TineaAnalysisResult:
    if str(foot_surface).strip().lower() != "dorsal":
        raise AnalysisError("Plantar skin assessment is disabled; use dorsal photographs.")
    side_label = {"left": "왼발", "right": "오른발"}.get(foot_side, "발")
    started_at = perf_counter()
    progress_logger.info(
        "[사진 분석][%s] 사진 입력을 받았습니다. 무좀·피부 상태 분석을 시작합니다.",
        side_label,
    )
    analyzer = get_tinea_analyzer()
    result = analyzer.analyze(
        image_bytes,
        filename,
        foot_side,
        foot_mask=foot_mask,
        foot_surface=foot_surface,
        foot_mask_source=foot_mask_source,
    )
    progress_logger.info(
        "[사진 분석][%s] 무좀·피부 상태 분석을 완료했습니다. | elapsedSeconds=%.2f",
        side_label,
        perf_counter() - started_at,
    )
    return result
