from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


MARKER_SIZE_MM = 30.0


class CalibrationError(ValueError):
    """Raised when a stable marker/background calibration cannot be built."""


@dataclass(frozen=True)
class SingleMarkerMeasurement:
    marker_id: int
    length_mm: float
    width_mm: float
    image_endpoints: np.ndarray


@dataclass(frozen=True)
class FootMeasurement:
    length_mm: float
    width_mm: float
    marker_spread_mm: float
    marker_count: int
    image_endpoints: np.ndarray
    image_centroid: np.ndarray
    per_marker_lengths: Mapping[int, float]


@dataclass(frozen=True)
class MotionCheck:
    common_marker_count: int
    median_error_px: float
    max_error_px: float
    per_marker_error_px: Mapping[int, float]


@dataclass
class LockedCalibration:
    background_bgr: np.ndarray
    background_lab: np.ndarray
    marker_homographies: dict[int, np.ndarray]
    reference_corners: dict[int, np.ndarray]
    roi_polygon: np.ndarray
    roi_mask: np.ndarray
    image_left_bay_mask: np.ndarray
    image_right_bay_mask: np.ndarray
    divider_top: np.ndarray
    divider_bottom: np.ndarray
    pixels_per_mm: float
    marker_size_mm: float
    frame_size: tuple[int, int]


def _points2d(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("2D points must be non-empty and finite")
    return array


def build_image_to_marker_homography(
    image_corners: np.ndarray, marker_size_mm: float = MARKER_SIZE_MM
) -> np.ndarray:
    """Build an image-pixel -> marker-local-millimetre homography.

    ArUco corners must be in canonical TL, TR, BR, BL order. The transform is
    valid for every point on the same physical plane. Its origin/orientation is
    marker-local, which is fine because Euclidean lengths are invariant to a
    planar translation and rotation.
    """

    corners = _points2d(image_corners)
    if corners.shape != (4, 2):
        raise ValueError("An ArUco marker must provide exactly four corners")
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be positive")
    if abs(cv2.contourArea(corners)) < 4.0:
        raise CalibrationError("Detected marker is too small or degenerate")

    size = np.float32(marker_size_mm)
    destination = np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, destination)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise CalibrationError("Could not create a valid marker homography")
    return homography / homography[2, 2]


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = _points2d(points)
    transformed = cv2.perspectiveTransform(
        source.reshape(1, -1, 2), np.asarray(homography, dtype=np.float64)
    )
    result = transformed.reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError("Homography produced non-finite coordinates")
    return result


def _measure_with_one_marker(
    image_points: np.ndarray, marker_id: int, homography: np.ndarray
) -> SingleMarkerMeasurement | None:
    metric_points = transform_points(image_points, homography).astype(np.float64)
    if len(metric_points) < 5:
        return None

    centered = metric_points - metric_points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)

    # Explicit 2-D dot products avoid a warning emitted by some NumPy/Accelerate
    # combinations for an otherwise valid (N, 2) @ (2,) matmul.
    major_projection = (
        metric_points[:, 0] * major_axis[0]
        + metric_points[:, 1] * major_axis[1]
    )
    minor_projection = (
        metric_points[:, 0] * minor_axis[0]
        + metric_points[:, 1] * minor_axis[1]
    )
    start_index = int(np.argmin(major_projection))
    end_index = int(np.argmax(major_projection))
    length_mm = float(major_projection[end_index] - major_projection[start_index])
    width_mm = float(np.ptp(minor_projection))
    if not np.isfinite([length_mm, width_mm]).all() or length_mm <= 0:
        return None

    return SingleMarkerMeasurement(
        marker_id=marker_id,
        length_mm=length_mm,
        width_mm=width_mm,
        image_endpoints=np.asarray(image_points, dtype=np.float32).reshape(-1, 2)[
            [start_index, end_index]
        ],
    )


def measure_foot_contour(
    contour: np.ndarray,
    marker_homographies: Mapping[int, np.ndarray],
    *,
    minimum_length_mm: float = 100.0,
    maximum_length_mm: float = 400.0,
    minimum_width_mm: float = 30.0,
    maximum_width_mm: float = 200.0,
) -> FootMeasurement | None:
    """Measure a foot contour independently with every locked marker.

    The median is deliberately used instead of averaging homographies: marker
    spacing is not assumed to be known, and a bad/far marker should not pull the
    result away from the consensus.
    """

    image_points = _points2d(contour)
    if len(image_points) < 5:
        return None

    measurements: list[SingleMarkerMeasurement] = []
    for marker_id, homography in sorted(marker_homographies.items()):
        try:
            measurement = _measure_with_one_marker(
                image_points, marker_id, homography
            )
        except (ValueError, cv2.error):
            continue
        if measurement is None:
            continue
        if not (minimum_length_mm <= measurement.length_mm <= maximum_length_mm):
            continue
        if not (minimum_width_mm <= measurement.width_mm <= maximum_width_mm):
            continue
        measurements.append(measurement)

    if not measurements:
        return None

    lengths = np.array([item.length_mm for item in measurements], dtype=np.float64)
    widths = np.array([item.width_mm for item in measurements], dtype=np.float64)
    length_median = float(np.median(lengths))
    width_median = float(np.median(widths))
    marker_spread = float(1.4826 * np.median(np.abs(lengths - length_median)))
    representative = min(
        measurements, key=lambda item: abs(item.length_mm - length_median)
    )

    return FootMeasurement(
        length_mm=length_median,
        width_mm=width_median,
        marker_spread_mm=marker_spread,
        marker_count=len(measurements),
        image_endpoints=representative.image_endpoints.copy(),
        image_centroid=image_points.mean(axis=0),
        per_marker_lengths={item.marker_id: item.length_mm for item in measurements},
    )


def marker_pixels_per_mm(
    reference_corners: Mapping[int, np.ndarray], marker_size_mm: float
) -> float:
    side_lengths: list[float] = []
    for corners in reference_corners.values():
        points = _points2d(corners)
        if points.shape != (4, 2):
            continue
        side_lengths.extend(
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        )
    if not side_lengths or marker_size_mm <= 0:
        raise CalibrationError("Cannot estimate marker pixel scale")
    return float(np.median(side_lengths) / marker_size_mm)


def infer_divider(reference_corners: Mapping[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Infer the line through the middle marker of the upper/lower rows."""

    centers = np.array(
        [_points2d(corners).mean(axis=0) for corners in reference_corners.values()],
        dtype=np.float32,
    )
    if len(centers) < 4 or len(centers) % 2 != 0:
        raise CalibrationError("An even number of distributed markers is required")

    y_order = np.argsort(centers[:, 1])
    row_size = len(centers) // 2
    top_row = centers[y_order[:row_size]]
    bottom_row = centers[y_order[row_size:]]
    if len(top_row) < 2 or len(bottom_row) < 2:
        raise CalibrationError("Could not infer the two marker rows")

    top_row = top_row[np.argsort(top_row[:, 0])]
    bottom_row = bottom_row[np.argsort(bottom_row[:, 0])]
    marker_sides = []
    for corners in reference_corners.values():
        points = _points2d(corners)
        marker_sides.extend(
            np.linalg.norm(points[(index + 1) % 4] - points[index])
            for index in range(4)
        )
    typical_marker_side = float(np.median(marker_sides))
    row_gap = float(np.median(bottom_row[:, 1]) - np.median(top_row[:, 1]))
    row_tilt = float(max(np.ptp(top_row[:, 1]), np.ptp(bottom_row[:, 1])))
    if row_gap < max(typical_marker_side, row_tilt * 1.2):
        raise CalibrationError("Markers do not form two separated horizontal rows")
    if (
        np.ptp(top_row[:, 0]) < typical_marker_side * 2.0
        or np.ptp(bottom_row[:, 0]) < typical_marker_side * 2.0
    ):
        raise CalibrationError("Markers are not spread across both foot bays")
    def row_middle(row: np.ndarray) -> np.ndarray:
        middle = len(row) // 2
        if len(row) % 2:
            return row[middle]
        return (row[middle - 1] + row[middle]) * 0.5

    top_middle = row_middle(top_row)
    bottom_middle = row_middle(bottom_row)
    if np.linalg.norm(bottom_middle - top_middle) < 10.0:
        raise CalibrationError("Marker rows are too close to infer two foot bays")
    return top_middle.astype(np.float32), bottom_middle.astype(np.float32)


def build_locked_calibration(
    background_bgr: np.ndarray,
    reference_corners: Mapping[int, np.ndarray],
    marker_size_mm: float = MARKER_SIZE_MM,
) -> LockedCalibration:
    if background_bgr.ndim != 3 or background_bgr.shape[2] != 3:
        raise CalibrationError("Background must be a BGR image")
    if len(reference_corners) < 4:
        raise CalibrationError("At least four markers are required for calibration")

    normalized_corners = {
        int(marker_id): _points2d(corners).reshape(4, 2)
        for marker_id, corners in reference_corners.items()
    }
    homographies = {
        marker_id: build_image_to_marker_homography(corners, marker_size_mm)
        for marker_id, corners in normalized_corners.items()
    }
    all_corners = np.concatenate(list(normalized_corners.values()), axis=0)
    roi_polygon = cv2.convexHull(all_corners.astype(np.float32)).reshape(-1, 2)
    divider_top, divider_bottom = infer_divider(normalized_corners)
    height, width = background_bgr.shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(
        roi_mask,
        [np.rint(roi_polygon).astype(np.int32)],
        255,
    )
    y_coordinates, x_coordinates = np.indices((height, width), dtype=np.float32)
    divider_direction = divider_bottom - divider_top
    divider_cross = (
        divider_direction[0] * (y_coordinates - divider_top[1])
        - divider_direction[1] * (x_coordinates - divider_top[0])
    )
    image_left_bay_mask = np.where(
        (divider_cross >= 0) & (roi_mask > 0), 255, 0
    ).astype(np.uint8)
    image_right_bay_mask = np.where(
        (divider_cross < 0) & (roi_mask > 0), 255, 0
    ).astype(np.uint8)

    return LockedCalibration(
        background_bgr=background_bgr.copy(),
        background_lab=cv2.cvtColor(background_bgr, cv2.COLOR_BGR2LAB).astype(np.int16),
        marker_homographies=homographies,
        reference_corners=normalized_corners,
        roi_polygon=roi_polygon,
        roi_mask=roi_mask,
        image_left_bay_mask=image_left_bay_mask,
        image_right_bay_mask=image_right_bay_mask,
        divider_top=divider_top,
        divider_bottom=divider_bottom,
        pixels_per_mm=marker_pixels_per_mm(normalized_corners, marker_size_mm),
        marker_size_mm=float(marker_size_mm),
        frame_size=(width, height),
    )


def compare_marker_positions(
    reference_corners: Mapping[int, np.ndarray],
    current_corners: Mapping[int, np.ndarray],
) -> MotionCheck:
    common_ids = sorted(set(reference_corners).intersection(current_corners))
    per_marker: dict[int, float] = {}
    all_errors: list[float] = []
    for marker_id in common_ids:
        reference = _points2d(reference_corners[marker_id])
        current = _points2d(current_corners[marker_id])
        if reference.shape != (4, 2) or current.shape != (4, 2):
            continue
        corner_errors = np.linalg.norm(current - reference, axis=1)
        per_marker[marker_id] = float(np.sqrt(np.mean(corner_errors**2)))
        all_errors.extend(float(value) for value in corner_errors)

    if not all_errors:
        return MotionCheck(0, float("inf"), float("inf"), {})
    return MotionCheck(
        common_marker_count=len(per_marker),
        median_error_px=float(np.median(list(per_marker.values()))),
        max_error_px=float(np.max(all_errors)),
        per_marker_error_px=per_marker,
    )


def _odd_kernel_size(value: float, minimum: int = 3, maximum: int = 31) -> int:
    size = int(round(value))
    size = max(minimum, min(maximum, size))
    return size if size % 2 == 1 else size + 1


def make_foreground_mask(
    frame_bgr: np.ndarray,
    calibration: LockedCalibration,
    *,
    threshold: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary foreground mask, colour-difference score)."""

    if frame_bgr.shape != calibration.background_bgr.shape:
        raise ValueError("Current frame size differs from the calibration frame")

    current_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    delta = current_lab - calibration.background_lab

    roi_values = delta[:, :, 0][calibration.roi_mask > 0]
    if roi_values.size:
        # Remove a global exposure shift. Feet occupy less than half of the board.
        delta[:, :, 0] -= int(np.median(roi_values))

    score = np.max(np.abs(delta), axis=2).clip(0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (5, 5), 0)
    _, mask = cv2.threshold(score, int(threshold), 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, calibration.roi_mask)

    opening_size = _odd_kernel_size(calibration.pixels_per_mm * 1.0)
    closing_size = _odd_kernel_size(calibration.pixels_per_mm * 3.0)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
    return mask, score


def side_of_divider(
    point: np.ndarray, divider_top: np.ndarray, divider_bottom: np.ndarray
) -> str:
    point = np.asarray(point, dtype=np.float64).reshape(2)
    top = np.asarray(divider_top, dtype=np.float64).reshape(2)
    bottom = np.asarray(divider_bottom, dtype=np.float64).reshape(2)
    direction = bottom - top
    relative = point - top
    cross = direction[0] * relative[1] - direction[1] * relative[0]
    return "image_left" if cross >= 0 else "image_right"


def extract_foot_contours(
    mask: np.ndarray,
    calibration: LockedCalibration,
    *,
    minimum_area_mm2: float = 3_000.0,
    maximum_area_mm2: float = 60_000.0,
) -> dict[str, np.ndarray]:
    minimum_area_px = minimum_area_mm2 * calibration.pixels_per_mm**2
    maximum_area_px = maximum_area_mm2 * calibration.pixels_per_mm**2
    output: dict[str, np.ndarray] = {}
    bay_masks = {
        "image_left": calibration.image_left_bay_mask,
        "image_right": calibration.image_right_bay_mask,
    }

    # Split first, then find components. A thin shadow/foreground bridge between
    # the feet must not turn both feet into one connected component.
    for side, bay_mask in bay_masks.items():
        side_mask = cv2.bitwise_and(mask, bay_mask)
        contours, _ = cv2.findContours(
            side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if minimum_area_px <= area <= maximum_area_px:
                candidates.append((area, contour))
        if candidates:
            output[side] = max(candidates, key=lambda item: item[0])[1]
    return output


def contour_touches_roi(
    contour: np.ndarray, roi_polygon: np.ndarray, margin_px: float = 1.0
) -> bool:
    points = _points2d(contour)
    polygon = _points2d(roi_polygon)
    step = max(1, len(points) // 300)
    for point in points[::step]:
        distance = cv2.pointPolygonTest(
            polygon, (float(point[0]), float(point[1])), True
        )
        if distance <= margin_px:
            return True
    return False


class CalibrationCollector:
    """Average an empty board and collect stable corners for all expected IDs."""

    def __init__(
        self,
        expected_ids: Sequence[int],
        target_frames: int = 30,
        minimum_seen_fraction: float = 0.7,
        maximum_corner_jitter_px: float = 2.5,
    ) -> None:
        if target_frames < 1:
            raise ValueError("target_frames must be positive")
        self.expected_ids = tuple(dict.fromkeys(int(value) for value in expected_ids))
        self.target_frames = int(target_frames)
        self.minimum_seen_fraction = float(minimum_seen_fraction)
        self.maximum_corner_jitter_px = float(maximum_corner_jitter_px)
        self.frame_count = 0
        self._image_sum: np.ndarray | None = None
        self._corner_samples: dict[int, list[np.ndarray]] = {
            marker_id: [] for marker_id in self.expected_ids
        }

    @property
    def complete(self) -> bool:
        return self.frame_count >= self.target_frames

    def add_frame(
        self, frame_bgr: np.ndarray, detected_corners: Mapping[int, np.ndarray]
    ) -> None:
        if self.complete:
            return
        frame_float = frame_bgr.astype(np.float32)
        if self._image_sum is None:
            self._image_sum = np.zeros_like(frame_float, dtype=np.float32)
        if self._image_sum.shape != frame_float.shape:
            raise CalibrationError("Frame size changed while collecting calibration")
        self._image_sum += frame_float
        self.frame_count += 1

        for marker_id in self.expected_ids:
            corners = detected_corners.get(marker_id)
            if corners is None:
                continue
            points = _points2d(corners)
            if points.shape == (4, 2):
                self._corner_samples[marker_id].append(points.copy())

    def finalize(self, marker_size_mm: float = MARKER_SIZE_MM) -> LockedCalibration:
        if not self.complete or self._image_sum is None:
            raise CalibrationError("Calibration frame collection is not complete")
        minimum_observations = max(
            1, int(np.ceil(self.target_frames * self.minimum_seen_fraction))
        )
        missing = [
            marker_id
            for marker_id, samples in self._corner_samples.items()
            if len(samples) < minimum_observations
        ]
        if missing:
            missing_text = ", ".join(str(value) for value in missing)
            raise CalibrationError(
                f"Markers not detected consistently during calibration: {missing_text}"
            )

        unstable: list[int] = []
        for marker_id, samples in self._corner_samples.items():
            stacked = np.stack(samples, axis=0)
            median_corners = np.median(stacked, axis=0)
            corner_errors = np.linalg.norm(stacked - median_corners, axis=2)
            jitter_95 = float(np.percentile(corner_errors, 95))
            if jitter_95 > self.maximum_corner_jitter_px:
                unstable.append(marker_id)
        if unstable:
            unstable_text = ", ".join(str(value) for value in unstable)
            raise CalibrationError(
                f"Board/camera moved while calibrating (marker IDs: {unstable_text})"
            )

        background = np.clip(
            self._image_sum / float(self.frame_count), 0, 255
        ).astype(np.uint8)
        reference_corners = {
            marker_id: np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)
            for marker_id, samples in self._corner_samples.items()
        }
        return build_locked_calibration(
            background, reference_corners, marker_size_mm=marker_size_mm
        )
