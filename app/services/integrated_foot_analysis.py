from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Mapping

import cv2
import numpy as np

from app.core.config import CHARUCO_FOOT_RIG_PROFILE_COMMIT, settings
from app.services.hallux_valgus_analysis import (
    HalluxValgusAnalysisResult,
    analyze_hallux_valgus_image,
)
from app.services.tinea_analysis import (
    AnalysisError,
    TineaAnalysisResult,
    analyze_foot_image,
    load_image,
)
from app.services.two_marker_foot_geometry import (
    IMAGE_SIDES,
    REQUIRED_MARKER_IDS,
    FootLengthMeasurement,
    HomographyFit,
    RigGeometry,
    TwoMarkerGeometryError,
    fit_two_marker_homography,
    marker_image_geometry,
    measure_foot_mask,
    rectification_transform,
    roi_pixels,
)


class IntegratedAnalysisError(AnalysisError):
    """Raised when the shared photo/measurement pipeline cannot finish."""


@dataclass(frozen=True)
class PreparedFoot:
    anatomical_side: str
    board_side: str
    image_png: bytes
    mask: np.ndarray
    length_mm: float
    ball_width_mm: float
    segmentation_confidence: float | None
    length_details: dict[str, Any]
    ball_width_details: dict[str, Any]


@dataclass(frozen=True)
class FootGeometryResult:
    original_filename: str
    input_width: int
    input_height: int
    measurement_valid: bool
    measurement_status: str
    measurement_invalid_reasons: tuple[str, ...]
    orientation_transform: str
    detected_marker_ids: tuple[int, ...]
    missing_marker_ids: tuple[int, ...]
    lens_correction: dict[str, Any]
    global_calibration: dict[str, Any]
    feet: dict[str, PreparedFoot]


@dataclass(frozen=True)
class IntegratedFootAnalysisResult:
    geometry: FootGeometryResult
    tinea: dict[str, TineaAnalysisResult]
    hallux_valgus: dict[str, HalluxValgusAnalysisResult]


@dataclass(frozen=True)
class _ArucoRuntime:
    source_dir: Path
    lens_module: ModuleType
    detector_type: type


@dataclass(frozen=True)
class _TwoMarkerCapture:
    runtime: _ArucoRuntime
    geometry: RigGeometry
    canonical_bgr: np.ndarray
    rectified_bgr: np.ndarray
    rectified_valid_mask: np.ndarray
    image_to_rectified: np.ndarray
    detector_result: Any
    corners_by_id: Mapping[int, np.ndarray]
    orientation_operations: tuple[str, ...]
    image_geometry: dict[str, float]
    homography_fit: HomographyFit
    lens_metadata: dict[str, Any]


@dataclass(frozen=True)
class _TwoMarkerSegmentation:
    measurement_masks_by_side: Mapping[str, np.ndarray]
    analysis_masks_by_side: Mapping[str, np.ndarray]
    confidence_by_side: Mapping[str, float]


@dataclass
class _CachedYolo:
    model: Any
    inference_lock: threading.Lock


_runtime: _ArucoRuntime | None = None
_runtime_lock = threading.Lock()
_yolo_cache: dict[tuple[str, str], _CachedYolo] = {}
_yolo_cache_lock = threading.Lock()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _module_from_source(name: str, source_dir: Path) -> ModuleType:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file or not _path_is_within(Path(module_file), source_dir):
        raise IntegratedAnalysisError(
            f"ArUco module '{name}' was imported from an unexpected location: "
            f"{module_file or 'unknown'}"
        )
    return module


def _get_cached_yolo(model_path: Path, device: str | None) -> _CachedYolo:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise IntegratedAnalysisError(
            f"ArUco foot segmentation weights not found: {resolved}"
        )
    key = (str(resolved), device or "")
    cached = _yolo_cache.get(key)
    if cached is not None:
        return cached

    with _yolo_cache_lock:
        cached = _yolo_cache.get(key)
        if cached is None:
            try:
                from ultralytics import YOLO

                model = YOLO(str(resolved))
            except Exception as exc:  # pragma: no cover - backend-specific detail
                raise IntegratedAnalysisError(
                    f"Failed to load ArUco foot segmentation model: {exc}"
                ) from exc
            cached = _CachedYolo(model=model, inference_lock=threading.Lock())
            _yolo_cache[key] = cached
    return cached


def _load_aruco_runtime() -> _ArucoRuntime:
    global _runtime
    source_dir = settings.aruco_source_dir.expanduser().resolve()
    if _runtime is not None:
        if _runtime.source_dir != source_dir:
            raise IntegratedAnalysisError(
                "ARUCO_SOURCE_DIR changed after the ArUco runtime was loaded; "
                "restart the server to apply it."
            )
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        if not (source_dir / "aruco_foot_measure" / "app.py").is_file() or not (
            source_dir / "aruco_foot_measure" / "lens.py"
        ).is_file():
            raise IntegratedAnalysisError(
                f"ArUco source repository not found or incomplete: {source_dir}"
            )
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        try:
            lens_module = _module_from_source(
                "aruco_foot_measure.lens", source_dir
            )
            app_module = _module_from_source("aruco_foot_measure.app", source_dir)
        except IntegratedAnalysisError:
            raise
        except Exception as exc:
            raise IntegratedAnalysisError(
                f"Failed to load ArUco measurement code: {exc}"
            ) from exc

        _runtime = _ArucoRuntime(
            source_dir=source_dir,
            lens_module=lens_module,
            detector_type=app_module.ArucoMarkerDetector,
        )
    return _runtime


def _calibration_path(runtime: _ArucoRuntime) -> Path:
    path = settings.aruco_camera_calibration
    return (
        path.expanduser().resolve()
        if path is not None
        else runtime.source_dir / "models" / "hybrid_best_1280x720.npz"
    )


def _segmentation_weights_path() -> Path:
    path = settings.aruco_foot_segmentation_weights
    return (
        path.expanduser().resolve()
        if path is not None
        else (settings.weights_dir / "foot_seg_yolo11n_best.pt").resolve()
    )


def _two_marker_rig() -> RigGeometry:
    return RigGeometry(
        marker_size_mm=float(settings.aruco_marker_size_mm),
        marker_edge_gap_mm=float(settings.aruco_marker_edge_gap_mm),
        # The shared rectified canvas must include both virtual heel lines.
        heel_offset_mm=max(
            float(settings.aruco_left_heel_offset_mm),
            float(settings.aruco_right_heel_offset_mm),
        ),
        board_width_mm=float(settings.aruco_board_width_mm),
        marker_center_x_mm=float(settings.aruco_marker_center_x_mm),
    )


def _apply_orientation_operations(
    image: np.ndarray, operations: tuple[str, ...]
) -> np.ndarray:
    result = image
    for operation in operations:
        if operation == "horizontal_flip":
            result = cv2.flip(result, 1)
        elif operation == "rotate_180":
            result = cv2.flip(result, -1)
        else:  # pragma: no cover - operations are an internal allowlist
            raise TwoMarkerGeometryError(
                f"Unsupported two-marker image operation: {operation}"
            )
    return np.ascontiguousarray(result)


def _apply_point_operations(
    points: np.ndarray,
    operations: tuple[str, ...],
    image_shape: tuple[int, ...],
    *,
    inverse: bool = False,
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    selected = tuple(reversed(operations)) if inverse else operations
    height, width = int(image_shape[0]), int(image_shape[1])
    for operation in selected:
        if operation in {"horizontal_flip", "rotate_180"}:
            result[..., 0] = float(width - 1) - result[..., 0]
        if operation == "rotate_180":
            result[..., 1] = float(height - 1) - result[..., 1]
        if operation not in {"horizontal_flip", "rotate_180"}:
            raise TwoMarkerGeometryError(
                f"Unsupported two-marker point operation: {operation}"
            )
    return result


def _detect_two_marker_canonical(
    *,
    corrected_bgr: np.ndarray,
    distorted_bgr: np.ndarray,
    corrected_valid_mask: np.ndarray,
    detector: Any,
    runtime: _ArucoRuntime,
    undistortion_maps: Any,
    mirror: bool = False,
) -> tuple[np.ndarray, np.ndarray, Any, tuple[str, ...]]:
    base = ("horizontal_flip",) if mirror else ()
    attempts: list[str] = []
    for operations in (base, base + ("rotate_180",)):
        canonical = _apply_orientation_operations(corrected_bgr, operations)
        canonical_distorted = _apply_orientation_operations(
            distorted_bgr, operations
        )
        canonical_valid = _apply_orientation_operations(
            corrected_valid_mask, operations
        )
        primary = detector.detect(canonical)
        raw = detector.detect(canonical_distorted)

        def transform_raw_corners(points: np.ndarray) -> np.ndarray:
            sensor_points = _apply_point_operations(
                points, operations, distorted_bgr.shape, inverse=True
            )
            corrected_points = runtime.lens_module.undistort_points(
                sensor_points, undistortion_maps
            )
            return _apply_point_operations(
                corrected_points, operations, corrected_bgr.shape
            )

        merged = detector.merge_mapped_distorted_fallback(
            canonical,
            primary,
            raw,
            transform_raw_corners,
        )
        duplicate_ids = sorted(
            set(getattr(merged, "duplicate_ids", ())).intersection(
                REQUIRED_MARKER_IDS
            )
        )
        corners_by_id = getattr(merged, "corners_by_id", {})
        missing_ids = sorted(set(REQUIRED_MARKER_IDS).difference(corners_by_id))
        if duplicate_ids:
            attempts.append(f"{operations or ('none',)} duplicate={duplicate_ids}")
            continue
        if missing_ids:
            attempts.append(f"{operations or ('none',)} missing={missing_ids}")
            continue
        centres = {
            marker_id: np.asarray(corners_by_id[marker_id], dtype=np.float64)
            .reshape(4, 2)
            .mean(axis=0)
            for marker_id in REQUIRED_MARKER_IDS
        }
        if centres[1][1] >= centres[4][1]:
            attempts.append(f"{operations or ('none',)} reversed_ID_order")
            continue
        return canonical, canonical_valid, merged, operations
    raise TwoMarkerGeometryError(
        "Could not obtain canonical ID 1 -> ID 4 order; " + "; ".join(attempts)
    )


def _count_two_marker_anchors(image_bgr: np.ndarray) -> int:
    """Best-effort diagnostic count; never used to calculate measurements."""

    runtime = _load_aruco_runtime()
    detector = runtime.detector_type(
        settings.aruco_dictionary,
        expected_marker_ids=set(REQUIRED_MARKER_IDS),
        robust=True,
    )
    result = detector.detect(image_bgr)
    duplicates = set(getattr(result, "duplicate_ids", ()))
    return len(
        set(getattr(result, "corners_by_id", {}))
        .intersection(REQUIRED_MARKER_IDS)
        .difference(duplicates)
    )


def _prepare_two_marker_capture(
    source_bgr: np.ndarray,
    *,
    runtime: _ArucoRuntime | None = None,
) -> _TwoMarkerCapture:
    """Lens-correct and rectify a capture from the eight ID 1/4 corners."""

    selected_runtime = runtime or _load_aruco_runtime()
    calibration_path = _calibration_path(selected_runtime)
    if not calibration_path.is_file():
        raise IntegratedAnalysisError(
            f"ArUco camera calibration not found: {calibration_path}"
        )
    try:
        calibration = selected_runtime.lens_module.load_lens_calibration(
            calibration_path
        )
        calibration_rms = getattr(calibration, "rms", None)
        if calibration_rms is None:
            raise IntegratedAnalysisError(
                "ArUco camera calibration does not contain RMS metadata."
            )
        if float(calibration_rms) > float(settings.aruco_max_camera_rms_px):
            raise IntegratedAnalysisError(
                "ArUco camera calibration RMS exceeds the configured limit."
            )
        height, width = source_bgr.shape[:2]
        maps = selected_runtime.lens_module.build_undistortion_maps(
            calibration,
            (width, height),
            balance=float(settings.aruco_undistort_balance),
            allow_same_aspect_resize=bool(settings.aruco_allow_calibration_resize),
        )
        corrected = selected_runtime.lens_module.apply_undistortion(
            source_bgr, maps
        )
    except IntegratedAnalysisError:
        raise
    except Exception as exc:
        raise IntegratedAnalysisError(
            f"ArUco lens correction failed: {exc}"
        ) from exc

    detector = selected_runtime.detector_type(
        settings.aruco_dictionary,
        expected_marker_ids=set(REQUIRED_MARKER_IDS),
        robust=True,
    )
    canonical, canonical_valid, detection, operations = (
        _detect_two_marker_canonical(
            corrected_bgr=corrected,
            distorted_bgr=source_bgr,
            corrected_valid_mask=maps.valid_mask,
            detector=detector,
            runtime=selected_runtime,
            undistortion_maps=maps,
            mirror=False,
        )
    )
    rig = _two_marker_rig()
    corners_by_id = {
        marker_id: np.asarray(detection.corners_by_id[marker_id], dtype=np.float64)
        for marker_id in REQUIRED_MARKER_IDS
    }
    image_geometry = marker_image_geometry(
        corners_by_id,
        rig,
        minimum_marker_side_px=float(settings.aruco_minimum_marker_side_px),
    )
    fit = fit_two_marker_homography(corners_by_id, rig)
    if fit.world_fit_rmse_mm > float(settings.aruco_maximum_fit_rmse_mm):
        raise TwoMarkerGeometryError(
            f"Two-marker fit RMSE {fit.world_fit_rmse_mm:.2f} mm exceeds "
            f"{settings.aruco_maximum_fit_rmse_mm:.2f} mm"
        )
    if fit.leave_one_corner_out_p95_mm > float(settings.aruco_maximum_loocv_mm):
        raise TwoMarkerGeometryError(
            "Leave-one-corner-out error "
            f"{fit.leave_one_corner_out_p95_mm:.2f} mm exceeds "
            f"{settings.aruco_maximum_loocv_mm:.2f} mm"
        )
    image_to_rectified, output_size = rectification_transform(
        fit, rig, float(settings.aruco_pixels_per_mm)
    )
    rectified_bgr = cv2.warpPerspective(
        canonical,
        image_to_rectified,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_valid = cv2.warpPerspective(
        canonical_valid,
        image_to_rectified,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return _TwoMarkerCapture(
        runtime=selected_runtime,
        geometry=rig,
        canonical_bgr=canonical,
        rectified_bgr=rectified_bgr,
        rectified_valid_mask=rectified_valid,
        image_to_rectified=image_to_rectified,
        detector_result=detection,
        corners_by_id=corners_by_id,
        orientation_operations=operations,
        image_geometry=image_geometry,
        homography_fit=fit,
        lens_metadata=dict(maps.metadata),
    )


def _as_numpy(value: Any) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _side_roi_mask(
    shape: tuple[int, int],
    geometry: RigGeometry,
    side: str,
    *,
    full_height: bool,
) -> np.ndarray:
    left, top, right, bottom = roi_pixels(
        geometry,
        side,
        float(settings.aruco_pixels_per_mm),
        shape,
    )
    if full_height:
        bottom = shape[0]
    result = np.zeros(shape, dtype=np.uint8)
    result[top:bottom, left:right] = 255
    return result


def _segment_two_marker_feet(
    capture: _TwoMarkerCapture,
) -> _TwoMarkerSegmentation:
    """Run the existing YOLO once and retain measurement and analysis masks."""

    device = settings.aruco_yolo_device.strip() or None
    cached = _get_cached_yolo(_segmentation_weights_path(), device)
    options: dict[str, Any] = {
        "source": capture.canonical_bgr,
        "conf": float(settings.aruco_yolo_confidence),
        "imgsz": int(settings.aruco_yolo_image_size),
        "retina_masks": True,
        "save": False,
        "verbose": False,
    }
    if device:
        options["device"] = device
    try:
        with cached.inference_lock:
            predictions = cached.model.predict(**options)
    except Exception as exc:  # pragma: no cover - backend-specific detail
        raise IntegratedAnalysisError(
            f"ArUco foot segmentation inference failed: {exc}"
        ) from exc
    if not predictions:
        raise TwoMarkerGeometryError("YOLO returned no foot result")
    result = predictions[0]
    masks_object = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks_object is None or boxes is None:
        raise TwoMarkerGeometryError("YOLO did not detect every requested foot")
    masks = _as_numpy(masks_object.data)
    classes = _as_numpy(boxes.cls).reshape(-1)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    names = getattr(result, "names", getattr(cached.model, "names", {}))
    if masks.ndim == 2:
        masks = masks[None, ...]

    rectified_shape = capture.rectified_bgr.shape[:2]
    rectified_height, rectified_width = rectified_shape
    source_height, source_width = capture.canonical_bgr.shape[:2]
    measurement_rois = {
        side: _side_roi_mask(
            rectified_shape, capture.geometry, side, full_height=False
        )
        for side in IMAGE_SIDES
    }
    analysis_rois = {
        side: _side_roi_mask(
            rectified_shape, capture.geometry, side, full_height=True
        )
        for side in IMAGE_SIDES
    }
    candidates: list[tuple[np.ndarray, float]] = []
    for index in range(min(len(masks), len(classes), len(confidences))):
        class_id = int(classes[index])
        class_name = (
            str(names.get(class_id, class_id))
            if isinstance(names, Mapping)
            else str(names[class_id])
        )
        if class_name.strip().casefold() != "foot":
            continue
        resized = cv2.resize(
            np.asarray(masks[index], dtype=np.float32),
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        source_mask = np.where(
            resized > float(settings.aruco_yolo_mask_threshold), 255, 0
        ).astype(np.uint8)
        rectified = cv2.warpPerspective(
            source_mask,
            capture.image_to_rectified,
            (rectified_width, rectified_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (rectified > 0).astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            area_mm2 = float(stats[label, cv2.CC_STAT_AREA]) / (
                float(settings.aruco_pixels_per_mm) ** 2
            )
            if area_mm2 < 1_000.0:
                continue
            candidates.append(
                (
                    np.where(labels == label, 255, 0).astype(np.uint8),
                    float(confidences[index]),
                )
            )

    assigned: dict[str, list[tuple[np.ndarray, float]]] = {
        side: [] for side in IMAGE_SIDES
    }
    for mask, confidence in candidates:
        overlaps = {
            side: int(np.count_nonzero((mask > 0) & (roi > 0)))
            for side, roi in measurement_rois.items()
        }
        chosen = max(IMAGE_SIDES, key=lambda side: overlaps[side])
        if overlaps[chosen] > 0:
            assigned[chosen].append((mask, confidence))
    missing = [side for side in IMAGE_SIDES if not assigned[side]]
    if missing:
        raise TwoMarkerGeometryError(
            "YOLO did not detect every requested foot: " + ", ".join(missing)
        )

    measurement_masks: dict[str, np.ndarray] = {}
    analysis_masks: dict[str, np.ndarray] = {}
    confidence_by_side: dict[str, float] = {}
    for side, items in assigned.items():
        union = np.zeros(rectified_shape, dtype=np.uint8)
        for mask, _ in items:
            union = cv2.bitwise_or(union, mask)
        measurement_masks[side] = cv2.bitwise_and(
            union, measurement_rois[side]
        )
        # The upstream metric mask stops at ID 4's reference edge. Preserve the
        # same YOLO component below that edge for tinea/hallux image analysis so
        # changing the ruler does not remove the heel from downstream models.
        analysis_masks[side] = cv2.bitwise_and(union, analysis_rois[side])
        confidence_by_side[side] = max(confidence for _, confidence in items)
    return _TwoMarkerSegmentation(
        measurement_masks_by_side=measurement_masks,
        analysis_masks_by_side=analysis_masks,
        confidence_by_side=confidence_by_side,
    )


def _anatomical_side_by_board_side() -> dict[str, str]:
    image_left_side = settings.aruco_image_left_anatomical_side.strip().lower()
    if image_left_side not in {"left", "right"}:
        raise IntegratedAnalysisError(
            "ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE must be 'left' or 'right'."
        )
    return {
        "image_left": image_left_side,
        "image_right": "right" if image_left_side == "left" else "left",
    }


def _encode_png(image: np.ndarray, label: str) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IntegratedAnalysisError(f"Failed to encode {label} as PNG.")
    return encoded.tobytes()


@dataclass(frozen=True)
class _BottomEdgeFit:
    angle_degree: float
    inlier_ratio: float
    span_ratio: float
    rmse_px: float


def _bottom_edge_fit(mask: np.ndarray) -> _BottomEdgeFit | None:
    """Fit the long straight lower cut edge from column-wise bottom points."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == label)
    if xs.size < 10:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    foot_width = x_max - x_min + 1
    if foot_width < 10:
        return None
    bottom_by_x = np.full(binary.shape[1], -1, dtype=np.int32)
    np.maximum.at(bottom_by_x, xs, ys)
    occupied_x = np.flatnonzero(bottom_by_x >= 0)
    points = np.column_stack(
        (occupied_x, bottom_by_x[occupied_x])
    ).astype(np.float64)
    if len(points) < 10:
        return None

    rng = np.random.default_rng(0)
    threshold_px = 1.5
    minimum_pair_span = max(5.0, foot_width * 0.20)
    best_inliers: np.ndarray | None = None
    best_score = (-1, -1.0)
    for _ in range(400):
        first_index, second_index = rng.choice(
            len(points),
            size=2,
            replace=False,
        )
        first = points[first_index]
        second = points[second_index]
        delta_x = float(second[0] - first[0])
        if abs(delta_x) < minimum_pair_span:
            continue
        slope = float(second[1] - first[1]) / delta_x
        intercept = float(first[1] - slope * first[0])
        distances = np.abs(
            slope * points[:, 0] - points[:, 1] + intercept
        ) / np.sqrt(slope * slope + 1.0)
        inliers = distances <= threshold_px
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count == 0:
            continue
        inlier_span = float(np.ptp(points[inliers, 0]))
        score = (inlier_count, inlier_span)
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or np.count_nonzero(best_inliers) < 10:
        return None
    for _ in range(2):
        slope, intercept = np.polyfit(
            points[best_inliers, 0],
            points[best_inliers, 1],
            1,
        )
        distances = np.abs(
            slope * points[:, 0] - points[:, 1] + intercept
        ) / np.sqrt(slope * slope + 1.0)
        best_inliers = distances <= threshold_px
        if np.count_nonzero(best_inliers) < 10:
            return None

    slope, intercept = np.polyfit(
        points[best_inliers, 0],
        points[best_inliers, 1],
        1,
    )
    inlier_points = points[best_inliers]
    residuals = np.abs(
        slope * inlier_points[:, 0] - inlier_points[:, 1] + intercept
    ) / np.sqrt(slope * slope + 1.0)
    inlier_ratio = float(np.mean(best_inliers))
    span_ratio = float(np.ptp(inlier_points[:, 0]) / foot_width)
    rmse_px = float(np.sqrt(np.mean(residuals**2)))
    angle_degree = float(np.degrees(np.arctan(float(slope))))
    if (
        inlier_ratio < 0.70
        or span_ratio < 0.55
        or rmse_px > 1.0
        or abs(angle_degree) > 20.0
    ):
        return None
    return _BottomEdgeFit(
        angle_degree=angle_degree,
        inlier_ratio=inlier_ratio,
        span_ratio=span_ratio,
        rmse_px=rmse_px,
    )


def _expanded_rotation(
    image_shape: tuple[int, int] | tuple[int, int, int],
    angle_degree: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((corners, np.ones(4, dtype=np.float64)))
    transformed = (matrix @ homogeneous.T).T
    minimum = np.floor(transformed.min(axis=0))
    maximum = np.ceil(transformed.max(axis=0))
    matrix[:, 2] -= minimum
    output_width = int(maximum[0] - minimum[0] + 1)
    output_height = int(maximum[1] - minimum[1] + 1)
    return matrix, (output_width, output_height)


def _align_foot_bottom(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit = _bottom_edge_fit(binary_mask)
    if fit is None:
        raise IntegratedAnalysisError(
            "Rectified foot mask does not contain a reliable straight bottom "
            "alignment edge."
        )

    total_angle = fit.angle_degree
    aligned_mask: np.ndarray | None = None
    matrix: np.ndarray | None = None
    output_size: tuple[int, int] | None = None
    for attempt in range(2):
        matrix, output_size = _expanded_rotation(
            binary_mask.shape,
            total_angle,
        )
        aligned_mask = cv2.warpAffine(
            binary_mask,
            matrix,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_mask = (aligned_mask > 0).astype(np.uint8) * 255
        residual_fit = _bottom_edge_fit(aligned_mask)
        if residual_fit is None:
            raise IntegratedAnalysisError(
                "Rectified foot bottom alignment could not be verified."
            )
        if abs(residual_fit.angle_degree) <= 0.25:
            break
        if attempt == 1:
            raise IntegratedAnalysisError(
                "Rectified foot bottom alignment did not converge to a "
                "horizontal edge."
            )
        total_angle += residual_fit.angle_degree
        if abs(total_angle) > 20.0:
            raise IntegratedAnalysisError(
                "Rectified foot bottom alignment requires an unsafe rotation."
            )

    if matrix is None or output_size is None or aligned_mask is None:
        raise IntegratedAnalysisError(
            "Rectified foot bottom alignment did not produce an output frame."
        )
    aligned_image = cv2.warpAffine(
        image_bgr,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned_image, aligned_mask


def _crop_foot(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    pixels_per_mm: float,
) -> tuple[bytes, np.ndarray]:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.shape != image_bgr.shape[:2]:
        raise IntegratedAnalysisError(
            "Rectified foot mask shape does not match the rectified photo."
        )
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        raise IntegratedAnalysisError("ArUco foot segmentation returned an empty mask.")

    aligned_bgr, aligned_mask = _align_foot_bottom(image_bgr, binary)
    padding = max(0, int(round(settings.aruco_crop_padding_mm * pixels_per_mm)))
    aligned_ys, aligned_xs = np.where(aligned_mask > 0)
    desired_x1 = int(aligned_xs.min()) - padding
    desired_y1 = int(aligned_ys.min()) - padding
    desired_x2 = int(aligned_xs.max()) + padding + 1
    desired_y2 = int(aligned_ys.max()) + padding + 1
    height, width = aligned_mask.shape
    source_x1 = max(0, desired_x1)
    source_y1 = max(0, desired_y1)
    source_x2 = min(width, desired_x2)
    source_y2 = min(height, desired_y2)
    crop_bgr = aligned_bgr[source_y1:source_y2, source_x1:source_x2].copy()
    crop_mask = aligned_mask[source_y1:source_y2, source_x1:source_x2].copy()
    border_left = source_x1 - desired_x1
    border_top = source_y1 - desired_y1
    border_right = desired_x2 - source_x2
    border_bottom = desired_y2 - source_y2
    if border_left or border_top or border_right or border_bottom:
        crop_bgr = cv2.copyMakeBorder(
            crop_bgr,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        crop_mask = cv2.copyMakeBorder(
            crop_mask,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return _encode_png(crop_bgr, "rectified foot crop"), crop_mask


def _length_details(value: FootLengthMeasurement) -> dict[str, Any]:
    raw_length_mm = float(value.foot_length_mm)
    offset_mm = float(settings.aruco_foot_length_offset_mm)
    corrected_length_mm = raw_length_mm + offset_mm
    raw_details = value.to_dict()
    return {
        **raw_details,
        "raw_foot_length_mm": round(raw_length_mm, 3),
        "raw_formula_check_mm": raw_details["formula_check_mm"],
        "foot_length_offset_mm": round(offset_mm, 3),
        "foot_length_offset_cm": round(offset_mm / 10.0, 3),
        "foot_length_mm": round(corrected_length_mm, 3),
        "foot_length_cm": round(corrected_length_mm / 10.0, 3),
        "formula_check_mm": round(corrected_length_mm, 3),
        "final_length_formula": (
            "visible_to_id4_outer_edge_mm + fixed_heel_offset_mm "
            "+ foot_length_offset_mm"
        ),
    }


def _ball_width_details(value: FootLengthMeasurement) -> dict[str, Any]:
    start, end = value.ball_region_fraction_from_toe
    return {
        "ball_width_mm": round(float(value.ball_width_mm), 3),
        "ball_width_cm": round(float(value.ball_width_mm) / 10.0, 3),
        "ball_region_fraction_from_toe": [round(start, 6), round(end, 6)],
        "ball_line_world_mm": [list(point) for point in value.ball_line_world_mm],
        "method": "90th_percentile_horizontal_mask_chord_in_forefoot_band",
        "measurement_definition": "straight-line MTP-zone width, not girth",
    }


def prepare_foot_geometry(
    image_bytes: bytes,
    filename: str | None,
) -> FootGeometryResult:
    """Run the ID 1/4 ruler, shared YOLO, length, and ball width once."""

    runtime = _load_aruco_runtime()
    _, source_bgr = load_image(image_bytes)
    height, width = source_bgr.shape[:2]
    expected_size = (
        settings.aruco_expected_image_width,
        settings.aruco_expected_image_height,
    )
    if (
        not settings.aruco_allow_calibration_resize
        and (width, height) != expected_size
    ):
        raise IntegratedAnalysisError(
            "ArUco calibration requires an unmodified "
            f"{expected_size[0]}x{expected_size[1]} image; received {width}x{height}."
        )

    side_mapping = _anatomical_side_by_board_side()
    heel_offsets = {
        "left": float(settings.aruco_left_heel_offset_mm),
        "right": float(settings.aruco_right_heel_offset_mm),
    }
    try:
        capture = _prepare_two_marker_capture(source_bgr, runtime=runtime)
        segmentation = _segment_two_marker_feet(capture)
        measurements = {
            board_side: measure_foot_mask(
                segmentation.measurement_masks_by_side[board_side],
                replace(
                    capture.geometry,
                    heel_offset_mm=heel_offsets[side_mapping[board_side]],
                ),
                side=board_side,
                pixels_per_mm=float(settings.aruco_pixels_per_mm),
                minimum_area_mm2=float(settings.aruco_minimum_mask_area_mm2),
                minimum_foot_length_mm=float(
                    settings.aruco_minimum_foot_length_mm
                    - settings.aruco_foot_length_offset_mm
                ),
                maximum_foot_length_mm=float(
                    settings.aruco_maximum_foot_length_mm
                    - settings.aruco_foot_length_offset_mm
                ),
                ball_region_start_fraction=float(
                    settings.aruco_forefoot_start_fraction
                ),
                ball_region_end_fraction=float(
                    settings.aruco_forefoot_end_fraction
                ),
            )
            for board_side in IMAGE_SIDES
        }
    except IntegratedAnalysisError:
        raise
    except TwoMarkerGeometryError as exc:
        raise IntegratedAnalysisError(
            f"Two-marker geometry analysis failed: {exc}"
        ) from exc
    except Exception as exc:
        raise IntegratedAnalysisError(
            f"Two-marker geometry analysis failed: {exc}"
        ) from exc

    prepared_feet: dict[str, PreparedFoot] = {}
    for board_side in IMAGE_SIDES:
        if (
            board_side not in measurements
            or board_side not in segmentation.analysis_masks_by_side
        ):
            raise IntegratedAnalysisError(
                f"Two-marker pipeline did not produce both feet; missing {board_side}."
            )
        anatomical_side = side_mapping[board_side]
        measurement = measurements[board_side]
        length_details = _length_details(measurement)
        image_png, crop_mask = _crop_foot(
            capture.rectified_bgr,
            segmentation.analysis_masks_by_side[board_side],
            float(settings.aruco_pixels_per_mm),
        )
        prepared_feet[anatomical_side] = PreparedFoot(
            anatomical_side=anatomical_side,
            board_side=board_side,
            image_png=image_png,
            mask=crop_mask,
            length_mm=float(length_details["foot_length_mm"]),
            ball_width_mm=round(float(measurement.ball_width_mm), 3),
            segmentation_confidence=round(
                float(segmentation.confidence_by_side[board_side]), 6
            ),
            length_details=length_details,
            ball_width_details=_ball_width_details(measurement),
        )

    if set(prepared_feet) != {"left", "right"}:
        raise IntegratedAnalysisError(
            "The configured ArUco board-to-anatomy mapping is not one-to-one."
        )

    return FootGeometryResult(
        original_filename=Path(filename or "foot-photo.jpg").name,
        input_width=width,
        input_height=height,
        measurement_valid=True,
        measurement_status="measured",
        measurement_invalid_reasons=(),
        orientation_transform=(
            "+".join(capture.orientation_operations) or "none"
        ),
        detected_marker_ids=tuple(REQUIRED_MARKER_IDS),
        missing_marker_ids=(),
        lens_correction={
            **capture.lens_metadata,
            "operation_order": (
                "lens_remap -> orientation -> ID1/ID4 Homography -> YOLO -> metric"
            ),
        },
        global_calibration={
            "profile": "charuco_two_marker_id1_id4_v1",
            "source_commit": CHARUCO_FOOT_RIG_PROFILE_COMMIT,
            **capture.homography_fit.to_dict(),
            "rig_geometry": capture.geometry.to_dict(),
            "heel_offsets_mm_by_anatomical_side": heel_offsets,
            "marker_image_geometry": capture.image_geometry,
        },
        feet=prepared_feet,
    )


def analyze_integrated_foot_photo(
    image_bytes: bytes,
    filename: str | None,
) -> IntegratedFootAnalysisResult:
    """Prepare geometry, then analyze one calibrated two-foot photo."""

    return analyze_prepared_foot_geometry(
        prepare_foot_geometry(image_bytes, filename)
    )


def analyze_prepared_foot_geometry(
    geometry: FootGeometryResult,
) -> IntegratedFootAnalysisResult:
    """Run skin and hallux models on geometry already accepted before HTTP 202."""

    tinea_results: dict[str, TineaAnalysisResult] = {}
    for side in ("left", "right"):
        foot = geometry.feet[side]
        tinea_results[side] = analyze_foot_image(
            foot.image_png,
            f"{side}_{geometry.original_filename}.png",
            side,
            foot_mask=foot.mask,
        )

    hallux_results: dict[str, HalluxValgusAnalysisResult] = {}
    for side in ("left", "right"):
        foot = geometry.feet[side]
        hallux_results[side] = analyze_hallux_valgus_image(
            foot.image_png,
            f"{side}_{geometry.original_filename}.png",
            side,
            foot_mask=foot.mask,
        )

    return IntegratedFootAnalysisResult(
        geometry=geometry,
        tinea=tinea_results,
        hallux_valgus=hallux_results,
    )
