"""Visual evidence that a foot covers one missing rig marker.

Missing IDs alone are not evidence of occlusion. A single decoded marker only
provides a noisy extrapolation of its partner: all bounded corner-perturbation
projections must remain in the calibrated image and overlap the same confident
YOLO foot component. This is a diagnostic heuristic, never a measurement fit.
If that projection is unavailable, actual visible partial-marker corners and
printed cells can provide local evidence that the other corners are inside a
foot mask. Neither diagnostic supplies inferred corners for measurement.
Without enough visual evidence, the diagnostic remains unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.config import settings
from app.services import integrated_foot_analysis as integrated
from app.services.aruco_partial_marker_occlusion import (
    PARTIAL_MARKER_DIAGNOSTIC_REASON_CODES,
    find_partial_marker_foot_occlusion,
)
from app.services.two_marker_foot_geometry import (
    REQUIRED_MARKER_IDS,
    RigGeometry,
    marker_world_corners,
)


_MINIMUM_OVERLAP = 0.25
_CORNER_PERTURBATION_PX = 0.5
_MAXIMUM_PROJECTION_SHIFT_PX = 128.0
_MAXIMUM_PROJECTION_SHIFT_MARKER_WIDTHS = 2.5

FOOT_MARKER_DIAGNOSTIC_REASON_CODES = frozenset({
    "SOURCE_IMAGE_INVALID", "ARUCO_RUNTIME_UNAVAILABLE", "MARKER_DETECTION_FAILED",
    "REQUIRED_MARKERS_NOT_DETECTED", "REQUIRED_MARKERS_VISIBLE",
    "DUPLICATE_MARKERS_DETECTED", "MARKER_IDENTITY_UNVERIFIED",
    "CALIBRATION_UNAVAILABLE", "CALIBRATION_QUALITY_INVALID", "LENS_CORRECTION_FAILED",
    "MARKER_DETECTIONS_INCONSISTENT", "MARKER_PROJECTION_UNCERTAIN",
    "MARKER_PROJECTION_OUTSIDE_VALID_IMAGE", "FOOT_MODEL_UNAVAILABLE",
    "FOOT_MODEL_INFERENCE_FAILED", "FOOT_MODEL_NO_RESULTS", "FOOT_MODEL_RESULT_INVALID",
    "NO_FOOT_DETECTED", "NO_CONFIDENT_FOOT_DETECTED", "FOOT_COMPONENT_TOO_SMALL",
    "FOOT_MASK_REMAP_FAILED", "FOOT_MASK_COVERS_VISIBLE_MARKER",
    "FOOT_OVERLAP_AMBIGUOUS", "NO_FOOT_OVERLAP", "FOOT_OVERLAP_CONFIRMED",
    "DIAGNOSTIC_PROCESSING_FAILED",
    *PARTIAL_MARKER_DIAGNOSTIC_REASON_CODES,
})


@dataclass(frozen=True, slots=True)
class FootMarkerOcclusionEvidence:
    missing_marker_ids: tuple[int, ...]
    occluded_marker_ids: tuple[int, ...]
    overlap_ratio: float
    method: str = "calibrated_single_marker_projection_yolo_overlap"
    visible_corner_count: int | None = None
    covered_corner_count: int | None = None
    pattern_score: float | None = None


@dataclass(frozen=True, slots=True)
class FootMarkerOcclusionDiagnostic:
    """Safe diagnostic facts; unknown is not a negative observation.

    ``foot_detected`` describes the model result, not proof that every foot was
    segmented. Projected overlap is the conservative minimum for a positive
    result and the observed maximum for ambiguous coverage. Partial-marker
    overlap describes its selected quad and covering foot; unavailable is None.
    """
    status: Literal["foot_occlusion", "no_foot_overlap", "unknown", "markers_visible"]
    reason_code: str
    detected_marker_ids: tuple[int, ...] = ()
    missing_marker_ids: tuple[int, ...] = ()
    foot_detected: bool | None = None
    overlap_ratio: float | None = None
    evidence: FootMarkerOcclusionEvidence | None = None
    projection_reason_code: str | None = None
    partial_marker_reason_code: str | None = None
    partial_marker_candidate_count: int | None = None
    partial_marker_edge_count: int | None = None
    partial_marker_covered_candidate_count: int | None = None
    partial_marker_pattern_candidate_count: int | None = None


@dataclass(frozen=True, slots=True)
class _FootComponentsDiagnostic:
    components: tuple[np.ndarray, ...] = ()
    foot_detected: bool | None = None
    # An ignored uncertain prediction prevents a negative overlap conclusion,
    # while leaving the previous positive criteria unchanged for valid feet.
    uncertainty_reason: str | None = None


def _sane_polygon(
    polygon: np.ndarray,
    image_shape: tuple[int, int],
    *,
    minimum_side: float,
) -> bool:
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return False
    height, width = image_shape
    if (
        np.any(polygon[:, 0] < 2.0)
        or np.any(polygon[:, 0] > width - 3.0)
        or np.any(polygon[:, 1] < 2.0)
        or np.any(polygon[:, 1] > height - 3.0)
    ):
        return False
    contour = polygon.astype(np.float32)
    # Decoded corners have the same winding as the known rig. Do not silently
    # interpret a reflection, a bow tie, or a degenerate square as a valid pose.
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour, oriented=True) <= 0:
        return False
    edges = np.roll(polygon, -1, axis=0) - polygon
    sides = np.linalg.norm(edges, axis=1)
    if float(sides.min()) < minimum_side or float(sides.max() / sides.min()) > 2.5:
        return False
    angle_cosines = np.sum(edges * np.roll(edges, 1, axis=0), axis=1) / (
        sides * np.roll(sides, 1)
    )
    return bool(np.max(np.abs(angle_cosines)) < 0.85)


def _project_partner(
    visible_world: np.ndarray,
    missing_world: np.ndarray,
    visible_image: np.ndarray,
) -> np.ndarray | None:
    transform = cv2.getPerspectiveTransform(
        visible_world.astype(np.float32), visible_image.astype(np.float32)
    )
    if not np.isfinite(transform).all() or abs(float(np.linalg.det(transform))) < 1e-12:
        return None
    span = np.concatenate((visible_world, missing_world), axis=0)
    denominators = span @ transform[2, :2] + transform[2, 2]
    # A horizon in the rig span or an extreme projective scale makes a
    # single-marker extrapolation unsuitable for this diagnostic.
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        return None
    magnitudes = np.abs(denominators)
    if float(magnitudes.min() / magnitudes.max()) < 0.35:
        return None
    return cv2.perspectiveTransform(
        missing_world.reshape(1, 4, 2), transform
    ).reshape(4, 2)


def _project_missing_marker_candidates(
    visible_image: np.ndarray,
    visible_id: int,
    missing_id: int,
    geometry: RigGeometry,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    minimum_side = max(30.0, float(settings.aruco_minimum_marker_side_px))
    if not _sane_polygon(visible_image, image_shape, minimum_side=minimum_side):
        return ()
    visible_world = marker_world_corners(
        geometry.world_points_mm[visible_id], geometry.marker_size_mm
    )
    missing_world = marker_world_corners(
        geometry.world_points_mm[missing_id], geometry.marker_size_mm
    )
    baseline = _project_partner(visible_world, missing_world, visible_image)
    if baseline is None or not _sane_polygon(
        baseline, image_shape, minimum_side=minimum_side
    ):
        return ()
    visible_area = float(cv2.contourArea(visible_image.astype(np.float32)))
    reference_side = float(np.median(np.linalg.norm(
        np.roll(visible_image, -1, axis=0) - visible_image, axis=1
    )))
    maximum_shift = min(
        _MAXIMUM_PROJECTION_SHIFT_PX,
        _MAXIMUM_PROJECTION_SHIFT_MARKER_WIDTHS * reference_side,
    )
    candidates = [baseline]
    # Include every single-corner axial and diagonal +/-0.5 px perturbation.
    # These sensitivity probes are not a statistical uncertainty guarantee.
    # At the rig's 195 mm span even a subpixel change can move the projection
    # tens of pixels, so checking only the baseline would be overconfident.
    for corner in range(4):
        for dx, dy in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            perturbed = visible_image.copy()
            perturbed[corner] += _CORNER_PERTURBATION_PX * np.array((dx, dy))
            projected = _project_partner(visible_world, missing_world, perturbed)
            if projected is None or not _sane_polygon(
                projected, image_shape, minimum_side=max(20.0, minimum_side * 0.75)
            ):
                return ()
            if float(np.linalg.norm(projected - baseline, axis=1).max()) > maximum_shift:
                return ()
            candidates.append(projected)
    for polygon in candidates:
        area_ratio = float(cv2.contourArea(polygon.astype(np.float32))) / visible_area
        if not 0.5 <= area_ratio <= 2.0:
            return ()
    return tuple(candidates)


def _polygon_patch(polygon: np.ndarray) -> tuple[slice, slice, np.ndarray]:
    left, top = np.floor(polygon.min(axis=0)).astype(int)
    right, bottom = np.ceil(polygon.max(axis=0)).astype(int) + 1
    patch = np.zeros((bottom - top, right - left), dtype=np.uint8)
    points = np.rint(polygon - np.array((left, top))).astype(np.int32)
    cv2.fillConvexPoly(patch, points, 1)
    return slice(top, bottom), slice(left, right), patch.astype(bool)


def _foot_components_in_corrected_frame(
    source_bgr: np.ndarray,
    maps: Any,
    minimum_area: float,
) -> _FootComponentsDiagnostic:
    try:
        device = settings.aruco_yolo_device.strip() or None
        cached = integrated._get_cached_yolo(integrated._segmentation_weights_path(), device)
        minimum_confidence = max(0.5, float(settings.aruco_yolo_confidence))
        options: dict[str, Any] = {
            "source": source_bgr,
            "conf": minimum_confidence,
            "imgsz": int(settings.aruco_yolo_image_size),
            "retina_masks": True,
            "save": False,
            "verbose": False,
        }
        if device:
            options["device"] = device
    except Exception:
        return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_UNAVAILABLE")
    try:
        with cached.inference_lock:
            predictions = cached.model.predict(**options)
    except Exception:
        return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_INFERENCE_FAILED")
    if not predictions:
        return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_NO_RESULTS")
    try:
        result = predictions[0]
        masks_object, boxes = getattr(result, "masks", None), getattr(result, "boxes", None)
        if boxes is None:
            return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_RESULT_INVALID")
        classes = integrated._as_numpy(boxes.cls).reshape(-1)
        confidences = integrated._as_numpy(boxes.conf).reshape(-1)
        if masks_object is None:
            if len(classes) == len(confidences) == 0:
                return _FootComponentsDiagnostic(foot_detected=False, uncertainty_reason="NO_FOOT_DETECTED")
            return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_RESULT_INVALID")
        masks = integrated._as_numpy(masks_object.data)
        names = getattr(result, "names", getattr(cached.model, "names", {}))
    except Exception:
        return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_RESULT_INVALID")
    if masks.ndim == 2:
        masks = masks[None, ...]
    # retina_masks produces source-resolution masks. Resizing a letterboxed
    # tensor here could invent overlap at the expected marker location.
    if (
        masks.ndim != 3
        or masks.shape[1:] != source_bgr.shape[:2]
        or not len(masks) == len(classes) == len(confidences)
    ):
        return _FootComponentsDiagnostic(uncertainty_reason="FOOT_MODEL_RESULT_INVALID")
    components: list[np.ndarray] = []
    uncertainty_reason: str | None = None
    foot_detected = False
    for mask, class_value, confidence in zip(masks, classes, confidences):
        try:
            valid_values = (
                math.isfinite(float(class_value))
                and float(class_value).is_integer()
                and math.isfinite(float(confidence))
                and 0.0 <= float(confidence) <= 1.0
            )
        except (TypeError, ValueError, OverflowError):
            valid_values = False
        if not valid_values:
            uncertainty_reason = "FOOT_MODEL_RESULT_INVALID"
            continue
        class_id = int(class_value)
        if isinstance(names, Mapping):
            class_name = names.get(class_id, "")
        elif isinstance(names, (tuple, list)) and 0 <= class_id < len(names):
            class_name = names[class_id]
        else:
            uncertainty_reason = "FOOT_MODEL_RESULT_INVALID"
            continue
        if not isinstance(class_name, str) or not class_name.strip():
            uncertainty_reason = "FOOT_MODEL_RESULT_INVALID"
            continue
        if str(class_name).strip().casefold() != "foot":
            continue
        if float(confidence) < minimum_confidence:
            uncertainty_reason = uncertainty_reason or "NO_CONFIDENT_FOOT_DETECTED"
            continue
        if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
            uncertainty_reason = "FOOT_MODEL_RESULT_INVALID"
            continue
        binary = (mask > max(0.5, float(settings.aruco_yolo_mask_threshold))).astype(np.uint8)
        if not bool(binary.any()):
            uncertainty_reason = "FOOT_MODEL_RESULT_INVALID"
            continue
        foot_detected = True
        try:
            corrected = cv2.remap(
                binary, maps.map1, maps.map2, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            corrected[np.asarray(maps.valid_mask) == 0] = 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(corrected, connectivity=8)
        except Exception:
            return _FootComponentsDiagnostic(
                foot_detected=True, uncertainty_reason="FOOT_MASK_REMAP_FAILED",
            )
        if count == 1:
            uncertainty_reason = uncertainty_reason or "FOOT_MASK_REMAP_FAILED"
        for label in range(1, count):
            if float(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                components.append(labels == label)
            else:
                uncertainty_reason = uncertainty_reason or "FOOT_COMPONENT_TOO_SMALL"
    if not components and uncertainty_reason is None:
        uncertainty_reason = "NO_FOOT_DETECTED"
    return _FootComponentsDiagnostic(
        tuple(components), foot_detected if foot_detected or uncertainty_reason == "NO_FOOT_DETECTED" else None,
        uncertainty_reason,
    )


def _diagnose_projected_marker_occlusion(source_bgr: np.ndarray) -> FootMarkerOcclusionDiagnostic:
    """Distinguish positive, negative, and unavailable visual evidence.

    A negative result concerns only detected foot masks and the bounded marker
    projections. It does not identify why a marker was unreadable. Unknown
    results never masquerade as an observed lack of foot overlap.
    """
    detected_ids: tuple[int, ...] = ()
    missing_ids: tuple[int, ...] = ()
    foot_detected: bool | None = None

    def unknown(reason: str, overlap: float | None = None) -> FootMarkerOcclusionDiagnostic:
        return FootMarkerOcclusionDiagnostic(
            "unknown", reason, detected_ids, missing_ids, foot_detected, overlap,
        )

    def markers_visible() -> FootMarkerOcclusionDiagnostic:
        return FootMarkerOcclusionDiagnostic(
            "markers_visible", "REQUIRED_MARKERS_VISIBLE", detected_ids, missing_ids,
        )

    if (
        not isinstance(source_bgr, np.ndarray)
        or source_bgr.dtype != np.uint8
        or source_bgr.ndim != 3
        or source_bgr.shape[2] != 3
        or min(source_bgr.shape[:2]) < 8
    ):
        return unknown("SOURCE_IMAGE_INVALID")
    failure_reason = "ARUCO_RUNTIME_UNAVAILABLE"
    try:
        runtime = integrated._load_aruco_runtime()
        detector = runtime.detector_type(
            settings.aruco_dictionary,
            expected_marker_ids=set(REQUIRED_MARKER_IDS),
            robust=True,
        )
        failure_reason = "MARKER_DETECTION_FAILED"
        raw = detector.detect(source_bgr)
        raw_ids = set(getattr(raw, "corners_by_id", {})).intersection(REQUIRED_MARKER_IDS)
        detected_ids = tuple(sorted(raw_ids))
        missing_ids = tuple(sorted(set(REQUIRED_MARKER_IDS) - raw_ids))
        if set(getattr(raw, "duplicate_ids", ())).intersection(REQUIRED_MARKER_IDS):
            return unknown("DUPLICATE_MARKERS_DETECTED")
        if not raw_ids.issubset(set(getattr(raw, "dictionary_verified_ids", ()))):
            return unknown("MARKER_IDENTITY_UNVERIFIED")
        if not missing_ids:
            return markers_visible()
        failure_reason = "CALIBRATION_UNAVAILABLE"
        calibration = runtime.lens_module.load_lens_calibration(integrated._calibration_path(runtime))
        failure_reason = "CALIBRATION_QUALITY_INVALID"
        rms = getattr(calibration, "rms", None)
        if rms is None or not math.isfinite(float(rms)) or not 0 <= float(rms) <= settings.aruco_max_camera_rms_px:
            return unknown("CALIBRATION_QUALITY_INVALID")
        failure_reason = "LENS_CORRECTION_FAILED"
        height, width = source_bgr.shape[:2]
        maps = runtime.lens_module.build_undistortion_maps(
            calibration, (width, height),
            balance=float(settings.aruco_undistort_balance),
            allow_same_aspect_resize=bool(settings.aruco_allow_calibration_resize),
        )
        corrected = runtime.lens_module.apply_undistortion(source_bgr, maps)
        failure_reason = "MARKER_DETECTION_FAILED"
        primary = detector.detect(corrected)
        if raw_ids:
            merged = detector.merge_mapped_distorted_fallback(
                corrected, primary, raw,
                lambda points: runtime.lens_module.undistort_points(points, maps),
            )
        else:
            # A complete marker can become readable only after lens correction.
            # Its decoded corners already share the corrected mask coordinates;
            # no distorted-corner fallback or second point correction is needed.
            merged = primary
        merged_ids = set(getattr(merged, "corners_by_id", {})).intersection(REQUIRED_MARKER_IDS)
        detected_ids = tuple(sorted(raw_ids | merged_ids))
        missing_ids = tuple(sorted(set(REQUIRED_MARKER_IDS) - set(detected_ids)))
        if set(getattr(merged, "duplicate_ids", ())).intersection(REQUIRED_MARKER_IDS):
            return unknown("DUPLICATE_MARKERS_DETECTED")
        if not merged_ids.issubset(set(getattr(merged, "dictionary_verified_ids", ()))):
            return unknown("MARKER_IDENTITY_UNVERIFIED")
        if merged_ids == set(REQUIRED_MARKER_IDS):
            return markers_visible()
        if not merged_ids:
            return unknown("REQUIRED_MARKERS_NOT_DETECTED")
        if raw_ids and merged_ids != raw_ids:
            return unknown("MARKER_DETECTIONS_INCONSISTENT")
        visible_id = next(iter(merged_ids))
        missing_id = next(iter(set(REQUIRED_MARKER_IDS) - merged_ids))
        failure_reason = "MARKER_PROJECTION_UNCERTAIN"
        if raw_ids:
            # Preserve the original sensor-corner path when an anchor was
            # decoded there, even if corrected detection also found that ID.
            visible = np.asarray(runtime.lens_module.undistort_points(
                np.asarray(raw.corners_by_id[visible_id], dtype=np.float64).reshape(4, 2), maps
            ), dtype=np.float64).reshape(4, 2)
        else:
            visible = np.asarray(primary.corners_by_id[visible_id], dtype=np.float64).reshape(4, 2)
        candidates = _project_missing_marker_candidates(
            visible, visible_id, missing_id, integrated._two_marker_rig(), corrected.shape[:2]
        )
        if not candidates:
            return unknown("MARKER_PROJECTION_UNCERTAIN")
        patches = [_polygon_patch(polygon) for polygon in candidates]
        valid = np.asarray(maps.valid_mask) > 0
        visible_patch = _polygon_patch(visible)
        for ys, xs, polygon_mask in [visible_patch, *patches]:
            if not np.all(valid[ys, xs][polygon_mask]):
                return unknown("MARKER_PROJECTION_OUTSIDE_VALID_IMAGE")
        maximum_marker_area = max(int(np.count_nonzero(mask)) for _, _, mask in patches)
        failure_reason = "FOOT_MODEL_RESULT_INVALID"
        feet = _foot_components_in_corrected_frame(source_bgr, maps, 4.0 * maximum_marker_area)
        foot_detected = feet.foot_detected
        if not feet.components:
            return unknown(feet.uncertainty_reason or "NO_FOOT_DETECTED")
        failure_reason = "DIAGNOSTIC_PROCESSING_FAILED"
        ambiguous_visible_marker = False
        maximum_overlap = 0.0
        for component in feet.components:
            ys, xs, polygon_mask = visible_patch
            # A mask covering the decoded marker as well is suspect (for
            # example a whole-board false positive), not occlusion evidence.
            if float(np.mean(component[ys, xs][polygon_mask])) > 0.10:
                ambiguous_visible_marker = True
                continue
            overlaps = tuple(
                float(np.mean(component[ys, xs][polygon_mask]))
                for ys, xs, polygon_mask in patches
            )
            minimum_overlap = min(overlaps)
            maximum_overlap = max(maximum_overlap, max(overlaps))
            if minimum_overlap >= _MINIMUM_OVERLAP:
                evidence = FootMarkerOcclusionEvidence(
                    missing_marker_ids=(missing_id,),
                    occluded_marker_ids=(missing_id,),
                    overlap_ratio=minimum_overlap,
                )
                return FootMarkerOcclusionDiagnostic(
                    "foot_occlusion", "FOOT_OVERLAP_CONFIRMED", detected_ids, missing_ids,
                    True, minimum_overlap, evidence,
                )
        if ambiguous_visible_marker:
            return unknown("FOOT_MASK_COVERS_VISIBLE_MARKER")
        if feet.uncertainty_reason is not None:
            return unknown(feet.uncertainty_reason)
        if maximum_overlap > 0.0:
            return unknown("FOOT_OVERLAP_AMBIGUOUS", maximum_overlap)
        return FootMarkerOcclusionDiagnostic(
            "no_foot_overlap", "NO_FOOT_OVERLAP", detected_ids, missing_ids, True, 0.0,
        )
    except Exception:
        # Report only the stage's closed reason code, never exception text,
        # filenames, model metadata, or a made-up negative overlap observation.
        return unknown(failure_reason)


def diagnose_foot_marker_occlusion(source_bgr: np.ndarray) -> FootMarkerOcclusionDiagnostic:
    """Use visible partial-marker corners if distant marker projection fails.

    Partial corners are evidence for a retake only; they are never returned to
    the measurement geometry pipeline as decoded ArUco measurement anchors.
    """
    result = _diagnose_projected_marker_occlusion(source_bgr)
    if result.status != "unknown" or result.reason_code not in {
        "MARKER_PROJECTION_UNCERTAIN", "MARKER_PROJECTION_OUTSIDE_VALID_IMAGE",
        "LENS_CORRECTION_FAILED",
    } or len(result.detected_marker_ids) != 1:
        return result
    # A partial printed black marker must have observable dark pixels. This
    # inexpensive gate avoids loading YOLO for blank, synthetic or faded data.
    foot_detected = result.foot_detected
    try:
        gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        if int(np.count_nonzero(gray < 70)) < 24:
            return result
        runtime = integrated._load_aruco_runtime()
        detector = runtime.detector_type(settings.aruco_dictionary, expected_marker_ids=set(REQUIRED_MARKER_IDS), robust=True)
        raw = detector.detect(source_bgr)
        raw_ids = set(raw.corners_by_id).intersection(REQUIRED_MARKER_IDS)
        if len(raw_ids) != 1 or set(getattr(raw, "duplicate_ids", ())).intersection(REQUIRED_MARKER_IDS):
            return result
        visible_id = next(iter(raw_ids))
        if visible_id not in getattr(raw, "dictionary_verified_ids", ()):
            return result
        visible = np.asarray(raw.corners_by_id[visible_id], dtype=np.float64).reshape(4, 2)
        height, width = source_bgr.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        maps = SimpleNamespace(map1=xx, map2=yy, valid_mask=np.ones((height, width), np.uint8))
        side = float(np.min(np.linalg.norm(np.roll(visible, -1, axis=0) - visible, axis=1)))
        feet = _foot_components_in_corrected_frame(source_bgr, maps, 4.0 * max(20.0, side * 0.5) ** 2)
        foot_detected = feet.foot_detected
        if not feet.components:
            return FootMarkerOcclusionDiagnostic(
                "unknown", feet.uncertainty_reason or "NO_FOOT_DETECTED",
                tuple(sorted(raw_ids)), result.missing_marker_ids, feet.foot_detected,
            )
        partial_details: dict[str, Any] = {}
        cover = find_partial_marker_foot_occlusion(
            source_bgr, visible_marker_id=visible_id, visible_corners=visible,
            foot_components=feet.components, dictionary_name=settings.aruco_dictionary,
            diagnostics=partial_details,
        )
        partial_reason = partial_details.get("reason_code")
        if not isinstance(partial_reason, str) or partial_reason not in PARTIAL_MARKER_DIAGNOSTIC_REASON_CODES:
            partial_reason = None
        trace: dict[str, Any] = {
            "projection_reason_code": result.reason_code,
            "partial_marker_reason_code": partial_reason,
        }
        for field, key in (
            ("partial_marker_candidate_count", "dark_component_count"),
            ("partial_marker_edge_count", "edge_count"),
            ("partial_marker_covered_candidate_count", "covered_candidate_count"),
            ("partial_marker_pattern_candidate_count", "pattern_candidate_count"),
        ):
            value = partial_details.get(key)
            trace[field] = value if type(value) is int and 0 <= value <= 1000000 else None
        if cover is None:
            return FootMarkerOcclusionDiagnostic(
                result.status, partial_reason or result.reason_code, result.detected_marker_ids,
                result.missing_marker_ids, feet.foot_detected, **trace,
            )
        evidence = FootMarkerOcclusionEvidence(
            missing_marker_ids=(cover.missing_marker_id,), occluded_marker_ids=(cover.missing_marker_id,),
            overlap_ratio=cover.overlap_ratio, method="visible_partial_marker_corners_yolo_overlap",
            visible_corner_count=cover.visible_corner_count, covered_corner_count=cover.covered_corner_count,
            pattern_score=cover.pattern_score,
        )
        return FootMarkerOcclusionDiagnostic(
            "foot_occlusion", "PARTIAL_MARKER_CORNERS_COVERED_BY_FOOT",
            tuple(sorted(raw_ids)), (cover.missing_marker_id,), True, cover.overlap_ratio, evidence,
            **trace,
        )
    except Exception:
        return FootMarkerOcclusionDiagnostic(
            "unknown", "PARTIAL_MARKER_PROCESSING_FAILED", result.detected_marker_ids,
            result.missing_marker_ids, foot_detected,
            projection_reason_code=result.reason_code,
            partial_marker_reason_code="PARTIAL_MARKER_PROCESSING_FAILED",
        )


def detect_foot_marker_occlusion(source_bgr: np.ndarray) -> FootMarkerOcclusionEvidence | None:
    """Compatibility wrapper: only a positive diagnostic returns evidence."""
    return diagnose_foot_marker_occlusion(source_bgr).evidence
