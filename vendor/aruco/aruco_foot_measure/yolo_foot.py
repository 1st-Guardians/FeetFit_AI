from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .pencil import PencilMeasurementError, RectifiedBoard
from .static_foot import FOOT_BAY_ROIS_MM


DEFAULT_YOLO_MODEL_PATH = Path("foot_seg_yolo11n_best.pt")
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MIN_COMPONENT_AREA_MM2 = 1_000.0
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.6
DEFAULT_TOE_MAX_EXTENSION_MM = 3.0


@dataclass(frozen=True)
class YoloRawDetection:
    """One model prediction before metric rectification.

    ``mask`` may have the model output resolution.  It is resized to the
    oriented source image before it is transformed into the rectified plane.
    """

    mask: np.ndarray
    confidence: float
    class_name: str
    class_id: int | None = None


@dataclass(frozen=True)
class YoloFootMask:
    """The unfilled union of accepted YOLO components in one board bay."""

    side: str
    mask: np.ndarray
    confidence: float
    component_confidences: tuple[float, ...]
    component_count: int
    area_mm2: float
    source_detection_indices: tuple[int, ...]


@dataclass(frozen=True)
class YoloFootSegmentationMetadata:
    model_path: str | None
    confidence_threshold: float
    image_size: int
    mask_threshold: float
    min_component_area_mm2: float
    duplicate_iou_threshold: float
    retina_masks: bool
    inference_backend: str
    requested_sides: tuple[str, ...]
    raw_prediction_count: int
    foot_class_prediction_count: int
    component_candidate_count: int
    kept_candidate_count: int
    assignments: Mapping[str, tuple[int, ...]]
    assignment_confidences: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class YoloFootSegmentation:
    by_side: Mapping[str, YoloFootMask]
    metadata: YoloFootSegmentationMetadata


@dataclass(frozen=True)
class ToeCapRefinementMetadata:
    """Quality evidence for one conservative, local toe-cap refinement."""

    applied: bool
    reason: str
    raw_tip_y_mm: float | None
    refined_tip_y_mm: float | None
    extension_mm: float
    raw_mask_area_mm2: float
    refined_mask_area_mm2: float
    added_area_mm2: float
    maximum_added_area_mm2: float
    maximum_extension_mm: float
    colour_separation_delta_e: float | None
    edge_support_fraction: float | None
    stability_spread_mm: float | None
    touches_search_limit: bool
    side_and_heel_unchanged: bool


@dataclass(frozen=True)
class ToeCapRefinement:
    mask: np.ndarray
    metadata: ToeCapRefinementMetadata


@dataclass(frozen=True)
class _ComponentCandidate:
    mask: np.ndarray
    confidence: float
    source_detection_index: int
    area_mm2: float


InjectedPredictor = Callable[..., Sequence[YoloRawDetection]]


def _validate_requested_sides(sides: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(sides))
    if not requested:
        raise PencilMeasurementError("At least one YOLO foot bay must be requested")
    unknown = [side for side in requested if side not in FOOT_BAY_ROIS_MM]
    if unknown:
        raise PencilMeasurementError(
            "Unknown YOLO foot bay(s): " + ", ".join(unknown)
        )
    return requested


def _bay_mask(rectified: RectifiedBoard, side: str) -> np.ndarray:
    left_mm, top_mm, right_mm, bottom_mm = FOOT_BAY_ROIS_MM[side]
    origin_x, origin_y = rectified.world_origin_mm
    scale = float(rectified.pixels_per_mm)
    height, width = rectified.image_bgr.shape[:2]
    x_world = origin_x + np.arange(width, dtype=np.float32) / scale
    y_world = origin_y + np.arange(height, dtype=np.float32) / scale
    inside_x = (x_world >= left_mm) & (x_world < right_mm)
    inside_y = (y_world >= top_mm) & (y_world < bottom_mm)
    return (inside_y[:, None] & inside_x[None, :]).astype(np.uint8) * 255


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_on = first > 0
    second_on = second > 0
    union = int(np.count_nonzero(first_on | second_on))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(first_on & second_on))
    return float(intersection / union)


def _largest_binary_component(mask: np.ndarray) -> np.ndarray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def _robust_lab_model(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    centre = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - centre), axis=0)
    # L varies strongly under a foot shadow.  Broad L and tighter chroma floors
    # make the decision depend primarily on the skin/sock colour rather than a
    # dark halo on the otherwise neutral board.
    scale = np.maximum(mad, np.array([12.0, 4.0, 4.0], dtype=np.float32))
    return centre, scale


def _connected_addition(
    core: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    combined = np.where((core > 0) | candidate, 1, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(combined, connectivity=8)
    if count <= 1:
        return np.zeros_like(core, dtype=bool)
    core_labels = labels[core > 0]
    core_labels = core_labels[core_labels > 0]
    if core_labels.size == 0:
        return np.zeros_like(core, dtype=bool)
    values, counts = np.unique(core_labels, return_counts=True)
    label = int(values[int(np.argmax(counts))])
    return (labels == label) & (core == 0)


def refine_yolo_toe_cap_local(
    rectified: RectifiedBoard,
    foot_mask: np.ndarray,
    side: str,
    *,
    maximum_extension_mm: float = DEFAULT_TOE_MAX_EXTENSION_MM,
    toe_context_depth_mm: float = 12.0,
) -> ToeCapRefinement:
    """Recover a small, colour-supported toe cap without dilating the foot.

    Ultralytics exposes an already-binary mask in ``Results.masks.data``.  A
    lower application-level mask threshold therefore cannot restore a soft
    model boundary.  This optional second stage instead examines only a narrow
    metric band around the toe in the perspective-corrected colour image.

    The original mask is never removed or changed.  Pixels may only be added
    within ``maximum_extension_mm`` of its top twelve millimetres, and the
    addition must be connected, colour-separated from the board, stable across
    two classification margins, and supported by an image edge.  Any ambiguous
    case returns the byte-for-byte original mask.
    """

    if side not in FOOT_BAY_ROIS_MM:
        raise PencilMeasurementError(f"Unknown YOLO foot bay: {side}")
    image = np.asarray(rectified.image_bgr)
    mask = np.asarray(foot_mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise PencilMeasurementError("Toe refinement needs a non-empty BGR image")
    if mask.shape != image.shape[:2]:
        raise PencilMeasurementError("Toe-refinement mask shape is inconsistent")
    pixels_per_mm = float(rectified.pixels_per_mm)
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0.0:
        raise PencilMeasurementError("Toe refinement needs a positive board scale")
    if (
        not np.isfinite(maximum_extension_mm)
        or not 0.0 < maximum_extension_mm <= DEFAULT_TOE_MAX_EXTENSION_MM
    ):
        raise PencilMeasurementError(
            "Maximum toe extension must be greater than 0 and at most 3 mm"
        )
    if not np.isfinite(toe_context_depth_mm) or toe_context_depth_mm < 8.0:
        raise PencilMeasurementError("Toe context depth must be at least 8 mm")

    original = np.where(mask > 0, 255, 0).astype(np.uint8)
    raw_area_mm2 = float(np.count_nonzero(original) / pixels_per_mm**2)
    maximum_added_area_mm2 = min(80.0, max(12.0, raw_area_mm2 * 0.01))
    core = _largest_binary_component(original)
    origin_y = float(rectified.world_origin_mm[1])

    def finish(
        output: np.ndarray,
        *,
        applied: bool,
        reason: str,
        raw_tip_px: int | None,
        refined_tip_px: int | None = None,
        colour_separation: float | None = None,
        edge_support: float | None = None,
        stability_spread: float | None = None,
        touches_limit: bool = False,
    ) -> ToeCapRefinement:
        output_binary = np.where(output > 0, 255, 0).astype(np.uint8)
        output_area = float(np.count_nonzero(output_binary) / pixels_per_mm**2)
        if refined_tip_px is None:
            refined_tip_px = raw_tip_px
        raw_tip_mm = (
            None
            if raw_tip_px is None
            else origin_y + float(raw_tip_px) / pixels_per_mm
        )
        refined_tip_mm = (
            None
            if refined_tip_px is None
            else origin_y + float(refined_tip_px) / pixels_per_mm
        )
        extension_mm = (
            0.0
            if raw_tip_px is None or refined_tip_px is None
            else max(0.0, float(raw_tip_px - refined_tip_px) / pixels_per_mm)
        )
        return ToeCapRefinement(
            mask=output_binary,
            metadata=ToeCapRefinementMetadata(
                applied=bool(applied),
                reason=reason,
                raw_tip_y_mm=raw_tip_mm,
                refined_tip_y_mm=refined_tip_mm,
                extension_mm=extension_mm,
                raw_mask_area_mm2=raw_area_mm2,
                refined_mask_area_mm2=output_area,
                added_area_mm2=max(0.0, output_area - raw_area_mm2),
                maximum_added_area_mm2=maximum_added_area_mm2,
                maximum_extension_mm=float(maximum_extension_mm),
                colour_separation_delta_e=(
                    None
                    if colour_separation is None
                    else float(colour_separation)
                ),
                edge_support_fraction=(
                    None if edge_support is None else float(edge_support)
                ),
                stability_spread_mm=(
                    None
                    if stability_spread is None
                    else float(stability_spread)
                ),
                touches_search_limit=bool(touches_limit),
                side_and_heel_unchanged=True,
            ),
        )

    if core is None:
        return finish(
            original,
            applied=False,
            reason="empty_foot_component",
            raw_tip_px=None,
        )

    core_y, core_x = np.where(core > 0)
    raw_tip_y = int(core_y.min())
    context_bottom = min(
        image.shape[0] - 1,
        raw_tip_y + int(round(toe_context_depth_mm * pixels_per_mm)),
    )
    context_points = (
        (core > 0)
        & (np.arange(image.shape[0])[:, None] <= context_bottom)
    )
    context_x = np.where(context_points)[1]
    if context_x.size == 0:
        return finish(
            original,
            applied=False,
            reason="toe_context_is_empty",
            raw_tip_px=raw_tip_y,
        )

    x_margin = int(round((maximum_extension_mm + 3.0) * pixels_per_mm))
    local_left = max(0, int(context_x.min()) - x_margin)
    local_right = min(image.shape[1] - 1, int(context_x.max()) + x_margin)
    search_top = max(
        0, raw_tip_y - int(np.ceil(maximum_extension_mm * pixels_per_mm))
    )
    search_bottom = context_bottom

    roi = _bay_mask(rectified, side) > 0
    rows = np.arange(image.shape[0])[:, None]
    columns = np.arange(image.shape[1])[None, :]
    local_rectangle = (
        (rows >= search_top)
        & (rows <= search_bottom)
        & (columns >= local_left)
        & (columns <= local_right)
        & roi
    )

    outside_distance_mm = cv2.distanceTransform(
        np.where(core == 0, 1, 0).astype(np.uint8), cv2.DIST_L2, 5
    ) / pixels_per_mm
    inside_distance_mm = cv2.distanceTransform(
        core.astype(np.uint8), cv2.DIST_L2, 5
    ) / pixels_per_mm
    sure_foreground = (
        local_rectangle
        & (core > 0)
        & (inside_distance_mm >= 1.0)
        & (rows >= raw_tip_y + int(round(0.5 * pixels_per_mm)))
    )
    # Samples start beyond the maximum permitted correction, so genuinely
    # missing foot pixels can never train the local board/background model.
    sure_background = (
        local_rectangle
        & (core == 0)
        & (outside_distance_mm >= maximum_extension_mm + 0.8)
        & (outside_distance_mm <= maximum_extension_mm + 5.0)
    )
    minimum_samples = max(30, int(round(5.0 * pixels_per_mm**2)))
    if (
        np.count_nonzero(sure_foreground) < minimum_samples
        or np.count_nonzero(sure_background) < minimum_samples
    ):
        return finish(
            original,
            applied=False,
            reason="insufficient_local_colour_samples",
            raw_tip_px=raw_tip_y,
        )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    foreground_centre, foreground_scale = _robust_lab_model(
        lab[sure_foreground]
    )
    background_centre, background_scale = _robust_lab_model(
        lab[sure_background]
    )
    colour_weights = np.array([0.25, 1.0, 1.0], dtype=np.float32)
    colour_separation = float(
        np.linalg.norm((foreground_centre - background_centre) * colour_weights)
    )
    if colour_separation < 5.0:
        return finish(
            original,
            applied=False,
            reason="ambiguous_foot_background_colour",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
        )

    foreground_distance = np.sqrt(
        np.sum(
            colour_weights
            * ((lab - foreground_centre) / foreground_scale) ** 2,
            axis=2,
        )
    )
    background_distance = np.sqrt(
        np.sum(
            colour_weights
            * ((lab - background_centre) / background_scale) ** 2,
            axis=2,
        )
    )
    classification_margin = background_distance - foreground_distance
    allowed_addition = (
        local_rectangle
        & (core == 0)
        & (outside_distance_mm <= maximum_extension_mm)
    )

    additions: list[np.ndarray] = []
    tips: list[int] = []
    # Agreement between a neutral and a stricter local colour decision is a
    # useful ambiguity check, without pretending the binary YOLO mask contains
    # model probabilities.
    for margin in (0.0, 0.25):
        addition = _connected_addition(
            core, allowed_addition & (classification_margin >= margin)
        )
        additions.append(addition)
        if np.any(addition):
            tips.append(min(raw_tip_y, int(np.where(addition)[0].min())))
        else:
            tips.append(raw_tip_y)
    stability_spread = float(abs(tips[0] - tips[1]) / pixels_per_mm)
    if stability_spread > 0.8:
        return finish(
            original,
            applied=False,
            reason="unstable_local_colour_boundary",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )

    # The stricter mask is deliberately selected; loose-only fringe pixels are
    # diagnostic evidence, not permission to lengthen the foot.
    added = additions[1]
    if not np.any(added):
        return finish(
            original,
            applied=False,
            reason="no_supported_toe_extension",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )
    refined_tip_y = int(np.where(added)[0].min())
    extension_mm = float(raw_tip_y - refined_tip_y) / pixels_per_mm
    if extension_mm < 0.2:
        return finish(
            original,
            applied=False,
            reason="extension_below_resolution_limit",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )

    limit_guard = max(1, int(round(0.2 * pixels_per_mm)))
    touches_limit = bool(
        np.any(added[search_top : min(image.shape[0], search_top + limit_guard)])
    )
    if touches_limit:
        return finish(
            original,
            applied=False,
            reason="toe_colour_reaches_search_limit",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
            touches_limit=True,
        )

    added_area_mm2 = float(np.count_nonzero(added) / pixels_per_mm**2)
    if added_area_mm2 > maximum_added_area_mm2:
        return finish(
            original,
            applied=False,
            reason="toe_addition_area_is_implausible",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )

    tentative = original.copy()
    tentative[added] = 255
    tip_band_bottom = min(
        image.shape[0], refined_tip_y + max(2, int(round(pixels_per_mm)))
    )
    tip_support_area_mm2 = float(
        np.count_nonzero(tentative[refined_tip_y:tip_band_bottom])
        / pixels_per_mm**2
    )
    if tip_support_area_mm2 < 0.8:
        return finish(
            original,
            applied=False,
            reason="refined_tip_has_insufficient_support",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )

    sigma = max(0.6, 0.3 * pixels_per_mm)
    smooth_lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gradient_squared = np.zeros(image.shape[:2], dtype=np.float32)
    for channel in range(3):
        gradient_x = cv2.Sobel(
            smooth_lab[:, :, channel], cv2.CV_32F, 1, 0, ksize=3
        )
        gradient_y = cv2.Sobel(
            smooth_lab[:, :, channel], cv2.CV_32F, 0, 1, ksize=3
        )
        gradient_squared += gradient_x**2 + gradient_y**2
    gradient = np.sqrt(gradient_squared)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outer_boundary = (tentative > 0) & (
        cv2.erode(tentative, erosion_kernel) == 0
    )
    near_added_kernel_size = max(3, int(round(1.0 * pixels_per_mm)))
    if near_added_kernel_size % 2 == 0:
        near_added_kernel_size += 1
    near_added = cv2.dilate(
        added.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (near_added_kernel_size, near_added_kernel_size),
        ),
    ) > 0
    proposed_boundary = outer_boundary & near_added & local_rectangle
    if np.count_nonzero(proposed_boundary) < max(5, int(round(2.0 * pixels_per_mm))):
        return finish(
            original,
            applied=False,
            reason="refined_boundary_is_too_short",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            stability_spread=stability_spread,
        )
    background_gradients = gradient[sure_background]
    gradient_centre = float(np.median(background_gradients))
    gradient_mad = float(
        1.4826 * np.median(np.abs(background_gradients - gradient_centre))
    )
    edge_threshold = max(4.0, gradient_centre + 1.5 * gradient_mad)
    edge_support = float(
        np.mean(gradient[proposed_boundary] >= edge_threshold)
    )
    if edge_support < 0.25:
        return finish(
            original,
            applied=False,
            reason="refined_boundary_lacks_image_edge_support",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            edge_support=edge_support,
            stability_spread=stability_spread,
        )

    # The only mutation is an OR inside the permitted toe band.  This explicit
    # check protects the invariant if the implementation above changes later.
    changed = tentative != original
    if np.any(changed & ~allowed_addition):
        return finish(
            original,
            applied=False,
            reason="internal_toe_refinement_scope_error",
            raw_tip_px=raw_tip_y,
            colour_separation=colour_separation,
            edge_support=edge_support,
            stability_spread=stability_spread,
        )

    return finish(
        tentative,
        applied=True,
        reason="local_colour_and_edge_supported",
        raw_tip_px=raw_tip_y,
        refined_tip_px=refined_tip_y,
        colour_separation=colour_separation,
        edge_support=edge_support,
        stability_spread=stability_spread,
    )


def _split_metric_components(
    mask: np.ndarray,
    *,
    pixels_per_mm: float,
    min_component_area_mm2: float,
    confidence: float,
    source_detection_index: int,
) -> list[_ComponentCandidate]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    candidates: list[_ComponentCandidate] = []
    for label in range(1, labels_count):
        area_pixels = int(stats[label, cv2.CC_STAT_AREA])
        area_mm2 = float(area_pixels / (pixels_per_mm**2))
        if area_mm2 < min_component_area_mm2:
            continue
        component = np.zeros(mask.shape, dtype=np.uint8)
        component[labels == label] = 255
        candidates.append(
            _ComponentCandidate(
                mask=component,
                confidence=float(confidence),
                source_detection_index=source_detection_index,
                area_mm2=area_mm2,
            )
        )
    return candidates


def _deduplicate_candidates(
    candidates: Sequence[_ComponentCandidate], iou_threshold: float
) -> list[_ComponentCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.confidence, item.area_mm2),
        reverse=True,
    )
    kept: list[_ComponentCandidate] = []
    for candidate in ordered:
        if any(
            _mask_iou(candidate.mask, previous.mask) >= iou_threshold
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return kept


def postprocess_yolo_foot_masks(
    detections: Sequence[YoloRawDetection],
    source_shape: tuple[int, int] | Sequence[int],
    rectified: RectifiedBoard,
    sides: Sequence[str] = ("image_left", "image_right"),
    *,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    min_component_area_mm2: float = DEFAULT_MIN_COMPONENT_AREA_MM2,
    iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD,
    model_path: str | Path | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    inference_backend: str = "injected",
) -> YoloFootSegmentation:
    """Convert raw YOLO masks into metric, bay-specific binary masks.

    This helper deliberately performs no morphology, contour filling, colour
    segmentation, or HSV fallback.  Holes and boundaries from the network are
    preserved, apart from nearest-neighbour perspective resampling and clipping
    to the declared metric bay.
    """

    requested = _validate_requested_sides(sides)
    if len(source_shape) < 2:
        raise PencilMeasurementError("YOLO source shape must contain height and width")
    source_height, source_width = int(source_shape[0]), int(source_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise PencilMeasurementError("YOLO source image has invalid dimensions")
    if not 0.0 < mask_threshold < 1.0:
        raise PencilMeasurementError("YOLO mask threshold must be between 0 and 1")
    if min_component_area_mm2 <= 0.0:
        raise PencilMeasurementError("Minimum YOLO component area must be positive")
    if not 0.0 < iou_threshold <= 1.0:
        raise PencilMeasurementError("YOLO duplicate IoU threshold must be in (0, 1]")
    if rectified.pixels_per_mm <= 0.0:
        raise PencilMeasurementError("Rectified board scale must be positive")

    transform = np.asarray(rectified.image_to_rectified, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise PencilMeasurementError("Invalid image-to-rectified transform")
    rectified_height, rectified_width = rectified.image_bgr.shape[:2]

    foot_predictions: list[tuple[int, YoloRawDetection]] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, YoloRawDetection):
            raise PencilMeasurementError(
                "Injected YOLO predictor must return YoloRawDetection objects"
            )
        if detection.class_name.strip().casefold() != "foot":
            continue
        foot_predictions.append((index, detection))

    candidates: list[_ComponentCandidate] = []
    for source_index, detection in foot_predictions:
        raw_mask = np.asarray(detection.mask)
        raw_mask = np.squeeze(raw_mask)
        if raw_mask.ndim != 2 or raw_mask.size == 0:
            continue
        if not np.isfinite(raw_mask).all():
            continue
        resized = cv2.resize(
            raw_mask.astype(np.float32),
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        source_binary = (resized > mask_threshold).astype(np.uint8) * 255
        rectified_binary = cv2.warpPerspective(
            source_binary,
            transform,
            (rectified_width, rectified_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rectified_binary = (rectified_binary > 0).astype(np.uint8) * 255
        candidates.extend(
            _split_metric_components(
                rectified_binary,
                pixels_per_mm=float(rectified.pixels_per_mm),
                min_component_area_mm2=float(min_component_area_mm2),
                confidence=float(detection.confidence),
                source_detection_index=source_index,
            )
        )

    kept = _deduplicate_candidates(candidates, float(iou_threshold))
    roi_by_side = {side: _bay_mask(rectified, side) for side in requested}
    assigned: dict[str, list[tuple[_ComponentCandidate, np.ndarray]]] = {
        side: [] for side in requested
    }
    for candidate in kept:
        overlaps = {
            side: int(np.count_nonzero((candidate.mask > 0) & (roi > 0)))
            for side, roi in roi_by_side.items()
        }
        side = max(requested, key=lambda value: overlaps[value])
        if overlaps[side] == 0:
            continue
        clipped = cv2.bitwise_and(candidate.mask, roi_by_side[side])
        assigned[side].append((candidate, clipped))

    missing = [side for side in requested if not assigned[side]]
    if missing:
        detail = ", ".join(missing)
        raise PencilMeasurementError(
            f"YOLO did not detect a foot in every requested bay; missing: {detail}"
        )

    by_side: dict[str, YoloFootMask] = {}
    assignment_indices: dict[str, tuple[int, ...]] = {}
    assignment_confidences: dict[str, tuple[float, ...]] = {}
    for side in requested:
        union = np.zeros((rectified_height, rectified_width), dtype=np.uint8)
        items = assigned[side]
        for _, component in items:
            union = cv2.bitwise_or(union, component)
        confidences = tuple(float(item.confidence) for item, _ in items)
        source_indices = tuple(item.source_detection_index for item, _ in items)
        assignment_indices[side] = source_indices
        assignment_confidences[side] = confidences
        by_side[side] = YoloFootMask(
            side=side,
            mask=union,
            confidence=max(confidences),
            component_confidences=confidences,
            component_count=len(items),
            area_mm2=float(
                np.count_nonzero(union) / (rectified.pixels_per_mm**2)
            ),
            source_detection_indices=source_indices,
        )

    metadata = YoloFootSegmentationMetadata(
        model_path=None if model_path is None else str(model_path),
        confidence_threshold=float(confidence_threshold),
        image_size=int(imgsz),
        mask_threshold=float(mask_threshold),
        min_component_area_mm2=float(min_component_area_mm2),
        duplicate_iou_threshold=float(iou_threshold),
        retina_masks=True,
        inference_backend=inference_backend,
        requested_sides=requested,
        raw_prediction_count=len(detections),
        foot_class_prediction_count=len(foot_predictions),
        component_candidate_count=len(candidates),
        kept_candidate_count=len(kept),
        assignments=assignment_indices,
        assignment_confidences=assignment_confidences,
    )
    return YoloFootSegmentation(by_side=by_side, metadata=metadata)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return str(class_id)


def _extract_ultralytics_detections(
    results: Any, *, fallback_names: Any = None
) -> list[YoloRawDetection]:
    if results is None:
        return []
    result_items = list(results)
    if not result_items:
        return []
    result = result_items[0]
    masks_object = getattr(result, "masks", None)
    masks_data = None if masks_object is None else getattr(masks_object, "data", None)
    if masks_data is None:
        return []
    masks = _as_numpy(masks_data)
    if masks.ndim == 2:
        masks = masks[None, ...]

    boxes = getattr(result, "boxes", None)
    classes_data = None if boxes is None else getattr(boxes, "cls", None)
    confidences_data = None if boxes is None else getattr(boxes, "conf", None)
    if classes_data is None:
        return []
    classes = _as_numpy(classes_data).reshape(-1)
    confidences = (
        np.ones(len(classes), dtype=np.float32)
        if confidences_data is None
        else _as_numpy(confidences_data).reshape(-1)
    )
    names = getattr(result, "names", fallback_names)

    count = min(len(masks), len(classes))
    detections: list[YoloRawDetection] = []
    for index in range(count):
        class_id = int(classes[index])
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        detections.append(
            YoloRawDetection(
                mask=np.asarray(masks[index]),
                confidence=confidence,
                class_name=_class_name(names, class_id),
                class_id=class_id,
            )
        )
    return detections


def _predict_with_ultralytics(
    image_bgr: np.ndarray,
    *,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> list[YoloRawDetection]:
    if not model_path.is_file():
        raise PencilMeasurementError(
            f"YOLO foot-segmentation model was not found: {model_path}"
        )
    try:
        from ultralytics import YOLO
    except (ImportError, ModuleNotFoundError) as error:
        raise PencilMeasurementError(
            "Ultralytics is required for YOLO foot segmentation; "
            "install the 'ultralytics' package"
        ) from error

    try:
        model = YOLO(str(model_path))
        predict_options: dict[str, Any] = {
            "source": image_bgr,
            "conf": float(conf),
            "imgsz": int(imgsz),
            "retina_masks": True,
            "save": False,
            "verbose": False,
        }
        if device is not None:
            predict_options["device"] = device
        # One source frame and exactly one inference call.  Orientation must be
        # settled by the caller before entering this function.
        results = model.predict(**predict_options)
        return _extract_ultralytics_detections(
            results, fallback_names=getattr(model, "names", None)
        )
    except PencilMeasurementError:
        raise
    except Exception as error:
        raise PencilMeasurementError(
            f"YOLO foot-segmentation inference failed: {error}"
        ) from error


def segment_feet_yolo(
    image_bgr: np.ndarray,
    rectified: RectifiedBoard,
    sides: Sequence[str] = ("image_left", "image_right"),
    *,
    model_path: str | Path = DEFAULT_YOLO_MODEL_PATH,
    predictor: InjectedPredictor | None = None,
    conf: float = DEFAULT_CONFIDENCE,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    min_component_area_mm2: float = DEFAULT_MIN_COMPONENT_AREA_MM2,
    iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD,
    device: str | None = None,
) -> YoloFootSegmentation:
    """Run YOLO once on an oriented BGR frame and return rectified bay masks."""

    requested = _validate_requested_sides(sides)
    source = np.asarray(image_bgr)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise PencilMeasurementError("YOLO source must be a non-empty BGR image")
    if not 0.0 < conf <= 1.0:
        raise PencilMeasurementError("YOLO confidence threshold must be in (0, 1]")
    if imgsz <= 0:
        raise PencilMeasurementError("YOLO inference image size must be positive")

    path = Path(model_path)
    if predictor is None:
        detections = _predict_with_ultralytics(
            source,
            model_path=path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )
        backend = "ultralytics"
    else:
        options: dict[str, Any] = {
            "conf": float(conf),
            "imgsz": int(imgsz),
            "retina_masks": True,
            "mask_threshold": float(mask_threshold),
        }
        if device is not None:
            options["device"] = device
        try:
            # The injectable seam is intentionally before all postprocessing,
            # so unit tests and alternate runtimes do not import torch.
            detections = list(predictor(source, **options))
        except PencilMeasurementError:
            raise
        except Exception as error:
            raise PencilMeasurementError(
                f"Injected YOLO foot predictor failed: {error}"
            ) from error
        backend = "injected"

    return postprocess_yolo_foot_masks(
        detections,
        source.shape[:2],
        rectified,
        requested,
        mask_threshold=mask_threshold,
        min_component_area_mm2=min_component_area_mm2,
        iou_threshold=iou_threshold,
        model_path=path,
        confidence_threshold=conf,
        imgsz=imgsz,
        inference_backend=backend,
    )
