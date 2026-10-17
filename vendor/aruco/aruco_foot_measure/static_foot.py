from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np

from .pencil import (
    WORLD_POINTS_MM,
    BoardHomography,
    PencilMeasurementError,
    RectifiedBoard,
    marker_world_corners,
)


# The names describe positions in the perspective-corrected board image.  They
# intentionally do not claim anatomical left/right because a mirrored camera
# can swap those labels.  The centre marker column at x=181 mm is the ownership
# boundary, so one physical foot cannot be returned by both bays.
FOOT_BAY_ROIS_MM: dict[str, tuple[float, float, float, float]] = {
    "image_left": (5.0, 5.0, 181.0, 307.0),
    "image_right": (181.0, 5.0, 365.0, 307.0),
}

# Kept for all existing single-foot callers and regression photographs.
LEGACY_SINGLE_FOOT_ROI_MM = (45.0, 45.0, 205.0, 307.0)

# Compact layout used by photographs where the heel-side part of each foot is
# intentionally allowed to extend beyond the second marker row.  The supplied
# reference photograph resolves to about 168--170 mm between marker centres;
# 170 mm is the nominal repeatable board dimension and can be overridden by the
# CLI after measuring the physical board.
MARKER_OFFSET_ROW_SPACING_MM = 170.0
MARKER_OFFSET_FIXED_LENGTH_MM = 110.0
# No empirical person-specific correction is applied by default.  Metric
# rectification supplies the visible length directly; callers may still pass an
# explicit scale when they have calibrated their own fixed camera/board setup.
MARKER_OFFSET_VISIBLE_LENGTH_SCALE = 1.0


def make_marker_offset_world_points(
    row_spacing_mm: float = MARKER_OFFSET_ROW_SPACING_MM,
    column_spacing_mm: float | None = None,
) -> dict[int, tuple[float, float]]:
    if not 100.0 <= row_spacing_mm <= 250.0:
        raise PencilMeasurementError(
            "Marker-row spacing must be between 100 and 250 mm"
        )
    if column_spacing_mm is not None and not 50.0 <= column_spacing_mm <= 250.0:
        raise PencilMeasurementError(
            "Marker-column spacing must be between 50 and 250 mm"
        )
    top_y = WORLD_POINTS_MM[0][1]
    bottom_y = top_y + float(row_spacing_mm)
    if column_spacing_mm is None:
        x_by_column = {
            column: WORLD_POINTS_MM[column][0] for column in range(3)
        }
    else:
        # Keep the centre marker column at its established world coordinate.
        # FOOT_BAY_ROIS_MM uses this x coordinate as the left/right ownership
        # boundary, while only metric distances (not the arbitrary board
        # origin) affect the measurement.
        centre_x = float(WORLD_POINTS_MM[1][0])
        spacing = float(column_spacing_mm)
        x_by_column = {
            0: centre_x - spacing,
            1: centre_x,
            2: centre_x + spacing,
        }
    return {
        marker_id: (
            x_by_column[marker_id % 3],
            top_y if marker_id < 3 else bottom_y,
        )
        for marker_id in sorted(WORLD_POINTS_MM)
    }


def estimate_marker_offset_row_spacing(
    corners_by_id: Mapping[int, np.ndarray],
    marker_ids: Sequence[int],
    *,
    marker_size_mm: float = 30.0,
    column_spacing_mm: float | None = None,
    minimum_spacing_mm: float = 130.0,
    maximum_spacing_mm: float = 210.0,
) -> tuple[float, float]:
    """Infer compact marker-row spacing from known marker corners.

    The old board uses a 302 mm row gap, while the new fixed-offset photograph
    uses a compact gap.  Searching the gap keeps those two physical layouts
    separate and uses the declared physical marker size as the metric reference.
    Returns ``(spacing_mm, image_reprojection_rmse_px)``.
    """

    selected_ids = [marker_id for marker_id in marker_ids if marker_id in corners_by_id]
    if len(selected_ids) < 3:
        raise PencilMeasurementError(
            "Need at least 3 local markers to estimate compact row spacing"
        )
    if not ({marker_id < 3 for marker_id in selected_ids} == {False, True}):
        raise PencilMeasurementError(
            "Compact row-spacing markers must cover both marker rows"
        )
    reference_points = make_marker_offset_world_points(
        MARKER_OFFSET_ROW_SPACING_MM,
        column_spacing_mm=column_spacing_mm,
    )
    x_values = {reference_points[marker_id][0] for marker_id in selected_ids}
    if len(x_values) < 2:
        raise PencilMeasurementError(
            "Compact row-spacing markers must cover two marker columns"
        )
    if not np.isfinite(marker_size_mm) or marker_size_mm <= 0:
        raise PencilMeasurementError("Marker size must be positive")
    if not 100.0 <= minimum_spacing_mm < maximum_spacing_mm <= 250.0:
        raise PencilMeasurementError("Invalid compact row-spacing search range")

    image_points = np.concatenate(
        [
            np.asarray(corners_by_id[marker_id], dtype=np.float32).reshape(4, 2)
            for marker_id in selected_ids
        ],
        axis=0,
    )

    def score(spacing_mm: float) -> float:
        world_points_by_id = make_marker_offset_world_points(
            spacing_mm,
            column_spacing_mm=column_spacing_mm,
        )
        world_corners = np.concatenate(
            [
                marker_world_corners(
                    world_points_by_id[marker_id], marker_size_mm
                )
                for marker_id in selected_ids
            ],
            axis=0,
        ).astype(np.float32)
        homography, _ = cv2.findHomography(world_corners, image_points, 0)
        if homography is None:
            return float("inf")
        projected = cv2.perspectiveTransform(
            world_corners.reshape(1, -1, 2), homography
        ).reshape(-1, 2)
        return float(
            np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
        )

    coarse = np.arange(
        minimum_spacing_mm, maximum_spacing_mm + 0.25, 0.5, dtype=np.float64
    )
    coarse_scores = np.array([score(float(value)) for value in coarse])
    best_coarse = float(coarse[int(np.argmin(coarse_scores))])
    refine_left = max(minimum_spacing_mm, best_coarse - 1.0)
    refine_right = min(maximum_spacing_mm, best_coarse + 1.0)
    refined = np.arange(refine_left, refine_right + 0.025, 0.05)
    refined_scores = np.array([score(float(value)) for value in refined])
    best_index = int(np.argmin(refined_scores))
    best_spacing = float(refined[best_index])
    best_rmse = float(refined_scores[best_index])
    if not np.isfinite(best_rmse):
        raise PencilMeasurementError("Could not estimate compact marker-row spacing")
    return best_spacing, best_rmse


@dataclass(frozen=True)
class FootThresholdEstimate:
    threshold: float
    length_mm: float
    width_mm: float
    toe_rectified_px: np.ndarray
    heel_rectified_px: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class StaticFootMeasurement:
    length_mm: float
    width_mm: float
    threshold_spread_mm: float
    toe_world_mm: np.ndarray
    heel_world_mm: np.ndarray
    toe_rectified_px: np.ndarray
    heel_rectified_px: np.ndarray
    mask: np.ndarray
    per_threshold_lengths_mm: Mapping[float, float]
    measurement_method: str = "inferred_heel"
    segmentation_method: str = "lab_colour_distance"
    segmentation_confidence: float | None = None
    segmentation_model: str | None = None
    segmentation_mask_area_mm2: float | None = None
    start_line_world_mm: np.ndarray | None = None
    start_line_rectified_px: np.ndarray | None = None
    start_line_angle_degrees: float | None = None
    start_line_support_mm: float | None = None
    start_line_residual_p95_mm: float | None = None
    visible_length_mm: float | None = None
    raw_visible_length_mm: float | None = None
    visible_length_scale: float | None = None
    fixed_offset_mm: float | None = None
    reference_edge_world_mm: np.ndarray | None = None
    reference_point_world_mm: np.ndarray | None = None
    reference_point_rectified_px: np.ndarray | None = None


@dataclass(frozen=True)
class DetectedStartLine:
    point_rectified_px: np.ndarray
    direction_rectified_px: np.ndarray
    normal_to_foot_rectified_px: np.ndarray
    endpoints_rectified_px: np.ndarray
    support_mm: float
    angle_degrees: float
    residual_p95_mm: float


def _odd(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def _roi_world_to_px(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    if len(roi_world_mm) != 4:
        raise PencilMeasurementError("A foot ROI must have four coordinates")
    left_mm, top_mm, right_mm, bottom_mm = (
        float(value) for value in roi_world_mm
    )
    if not left_mm < right_mm or not top_mm < bottom_mm:
        raise PencilMeasurementError("A foot ROI has invalid bounds")
    origin_x, origin_y = rectified.world_origin_mm
    scale = rectified.pixels_per_mm
    height, width = rectified.image_bgr.shape[:2]
    left = int(round((left_mm - origin_x) * scale))
    top = int(round((top_mm - origin_y) * scale))
    right = int(round((right_mm - origin_x) * scale))
    bottom = int(round((bottom_mm - origin_y) * scale))
    left = max(0, min(width - 1, left))
    right = max(0, min(width - 1, right))
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height - 1, bottom))
    if right - left < max(2, int(round(50.0 * scale))):
        raise PencilMeasurementError("A foot ROI is too narrow")
    if bottom - top < max(2, int(round(100.0 * scale))):
        raise PencilMeasurementError("A foot ROI is too short")
    return left, top, right, bottom


def _roi_mask(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bounds_px = _roi_world_to_px(rectified, roi_world_mm)
    mask = np.zeros(rectified.image_bgr.shape[:2], dtype=np.uint8)
    left, top, right, bottom = bounds_px
    cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    return mask, bounds_px


def _row_background_lab(
    lab_image: np.ndarray, pixels_per_mm: float
) -> np.ndarray:
    width = lab_image.shape[1]
    left = min(width - 2, int(round(220.0 * pixels_per_mm)))
    right = min(width, int(round(310.0 * pixels_per_mm)))
    if right - left < 10:
        raise PencilMeasurementError("Not enough empty board area to estimate background")
    return np.median(lab_image[:, left:right, :], axis=1)


def _largest_filled_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise PencilMeasurementError("Sock/foot foreground was not detected")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise PencilMeasurementError("Sock/foot contour was not detected")
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def _smooth_boundary(values: np.ndarray, sigma_px: float) -> np.ndarray:
    radius = max(3, int(round(sigma_px * 2.5)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / max(sigma_px, 1.0)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_red_start_line(rectified: RectifiedBoard) -> DetectedStartLine:
    """Detect the red heel reference line in a rectified board image.

    A long visible line is preferred.  When the heel hides most of the stroke,
    a short segment is accepted only if it is highly saturated, thin and close
    to horizontal.  The latter case is intentionally strict because the fitted
    line has to be extrapolated underneath the foot.
    """

    image = rectified.image_bgr
    pixels_per_mm = rectified.pixels_per_mm
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    search_roi = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(
        search_roi,
        (int(round(80.0 * pixels_per_mm)), int(round(270.0 * pixels_per_mm))),
        (int(round(360.0 * pixels_per_mm)), int(round(315.0 * pixels_per_mm))),
        255,
        -1,
    )
    close_width = _odd(pixels_per_mm * 2.0)
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_width, 3)
    )

    selected: tuple[np.ndarray, int, float] | None = None
    # (minimum saturation, minimum visible support, minimum aspect ratio,
    #  maximum thickness, minimum centre y).  The first pass retains the
    # original long-line rule; the second is the occluded-line fallback.
    attempts = (
        (50, 80.0, 8.0, None, None),
        (70, 8.0, 6.0, 3.0, 285.0),
    )
    for (
        minimum_saturation,
        minimum_support_mm,
        minimum_aspect_ratio,
        maximum_thickness_mm,
        minimum_center_y_mm,
    ) in attempts:
        red = (
            ((hue <= 12) | (hue >= 170))
            & (saturation >= minimum_saturation)
            & (value >= 60)
        )
        red_mask = np.where(red & (search_roi > 0), 255, 0).astype(np.uint8)
        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE, closing_kernel
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask)
        candidates: list[tuple[int, int]] = []
        for label in range(1, count):
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            center_y = float(stats[label, cv2.CC_STAT_TOP] + height * 0.5)
            if width < minimum_support_mm * pixels_per_mm:
                continue
            if height <= 0 or width / height < minimum_aspect_ratio:
                continue
            if (
                maximum_thickness_mm is not None
                and height > maximum_thickness_mm * pixels_per_mm
            ):
                continue
            if (
                minimum_center_y_mm is not None
                and center_y < minimum_center_y_mm * pixels_per_mm
            ):
                continue
            candidates.append((width * area, label))
        if candidates:
            _, selected_label = max(candidates)
            selected = (labels, selected_label, minimum_support_mm)
            break

    if selected is None:
        raise PencilMeasurementError("A long red start line was not detected")

    labels, selected_label, minimum_support_mm = selected
    y_coordinates, x_coordinates = np.where(labels == selected_label)
    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(
        points, cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction *= -1.0
    point = np.array([float(x0), float(y0)], dtype=np.float64)
    normal = np.array([direction[1], -direction[0]], dtype=np.float64)
    offsets = points.astype(np.float64) - point
    projections = offsets[:, 0] * direction[0] + offsets[:, 1] * direction[1]
    residuals = np.abs(offsets[:, 0] * normal[0] + offsets[:, 1] * normal[1])
    support_mm = float(np.ptp(projections) / pixels_per_mm)
    residual_p95_mm = float(np.percentile(residuals, 95) / pixels_per_mm)
    angle_degrees = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if abs(angle_degrees) > 10.0:
        raise PencilMeasurementError(
            f"Detected red line is not horizontal enough ({angle_degrees:.1f} deg)"
        )
    if support_mm < minimum_support_mm or residual_p95_mm > 2.5:
        raise PencilMeasurementError("Detected red pixels do not form a stable line")
    endpoints = np.vstack(
        (
            point + direction * float(projections.min()),
            point + direction * float(projections.max()),
        )
    ).astype(np.float32)
    return DetectedStartLine(
        point_rectified_px=point.astype(np.float32),
        direction_rectified_px=direction.astype(np.float32),
        normal_to_foot_rectified_px=normal.astype(np.float32),
        endpoints_rectified_px=endpoints,
        support_mm=support_mm,
        angle_degrees=angle_degrees,
        residual_p95_mm=residual_p95_mm,
    )


def _estimate_from_start_line_mask(
    mask: np.ndarray,
    rectified: RectifiedBoard,
    start_line: DetectedStartLine,
    threshold: float,
) -> FootThresholdEstimate:
    pixels_per_mm = rectified.pixels_per_mm
    y_coordinates, x_coordinates = np.where(mask > 0)
    if len(x_coordinates) < pixels_per_mm**2 * 3_000:
        raise PencilMeasurementError("Detected foot region is too small")

    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    relative = points - start_line.point_rectified_px
    signed_distances = (
        relative[:, 0] * start_line.normal_to_foot_rectified_px[0]
        + relative[:, 1] * start_line.normal_to_foot_rectified_px[1]
    )
    maximum_distance = float(signed_distances.max())
    if maximum_distance <= 0:
        raise PencilMeasurementError("Foot is not above the detected start line")

    toe_band = signed_distances >= maximum_distance - pixels_per_mm
    toe = np.median(points[toe_band], axis=0).astype(np.float32)
    toe_relative = toe - start_line.point_rectified_px
    toe_distance = float(
        toe_relative[0] * start_line.normal_to_foot_rectified_px[0]
        + toe_relative[1] * start_line.normal_to_foot_rectified_px[1]
    )
    toe += start_line.normal_to_foot_rectified_px * (
        maximum_distance - toe_distance
    )
    heel = toe - start_line.normal_to_foot_rectified_px * maximum_distance
    length_mm = maximum_distance / pixels_per_mm

    widths: list[float] = []
    for row in np.unique(y_coordinates):
        row_x = x_coordinates[y_coordinates == row]
        if len(row_x) >= 2:
            widths.append(float(row_x.max() - row_x.min()))
    if not widths:
        raise PencilMeasurementError("Foot width could not be estimated")
    width_mm = float(np.percentile(widths, 98) / pixels_per_mm)
    if not 180.0 <= length_mm <= 320.0:
        raise PencilMeasurementError(f"Implausible foot length: {length_mm:.1f} mm")
    if not 50.0 <= width_mm <= 140.0:
        raise PencilMeasurementError(f"Implausible foot width: {width_mm:.1f} mm")
    return FootThresholdEstimate(
        threshold=float(threshold),
        length_mm=length_mm,
        width_mm=width_mm,
        toe_rectified_px=toe,
        heel_rectified_px=heel.astype(np.float32),
        mask=mask,
    )


def _longest_stable_estimate_window(
    estimates: Sequence[FootThresholdEstimate], maximum_spread_mm: float
) -> list[FootThresholdEstimate]:
    ordered = sorted(estimates, key=lambda item: item.threshold)
    best: list[FootThresholdEstimate] = []
    for start in range(len(ordered)):
        for end in range(start + 3, len(ordered) + 1):
            window = ordered[start:end]
            lengths = [item.length_mm for item in window]
            if max(lengths) - min(lengths) > maximum_spread_mm:
                continue
            if len(window) > len(best) or (
                len(window) == len(best)
                and sum(item.threshold for item in window)
                > sum(item.threshold for item in best)
            ):
                best = window
    return best


def _estimate_from_mask(
    mask: np.ndarray, rectified: RectifiedBoard, threshold: float
) -> FootThresholdEstimate:
    pixels_per_mm = rectified.pixels_per_mm
    y_coordinates, x_coordinates = np.where(mask > 0)
    if len(x_coordinates) < pixels_per_mm**2 * 3_000:
        raise PencilMeasurementError("Detected foot region is too small")

    toe_y = int(y_coordinates.min())
    toe_band = x_coordinates[y_coordinates <= toe_y + max(2, int(pixels_per_mm))]
    toe_x = float(np.median(toe_band))

    first_y = toe_y
    last_y = int(y_coordinates.max())
    row_count = last_y - first_y + 1
    left_boundary = np.empty(row_count, dtype=np.float64)
    right_boundary = np.empty(row_count, dtype=np.float64)
    row_centers = np.empty(row_count, dtype=np.float64)
    for offset, row in enumerate(range(first_y, last_y + 1)):
        row_x = np.flatnonzero(mask[row] > 0)
        if not len(row_x):
            left_boundary[offset] = np.nan
            right_boundary[offset] = np.nan
            row_centers[offset] = np.nan
            continue
        left_boundary[offset] = float(row_x.min())
        right_boundary[offset] = float(row_x.max())
        row_centers[offset] = float((row_x.min() + row_x.max()) * 0.5)

    valid = np.isfinite(row_centers)
    if valid.sum() < pixels_per_mm * 100:
        raise PencilMeasurementError("Foot contour is not tall enough")
    indices = np.arange(row_count)
    for values in (left_boundary, right_boundary, row_centers):
        missing = ~np.isfinite(values)
        values[missing] = np.interp(indices[missing], indices[~missing], values[~missing])

    # The lower leg enters the board at the bottom and bends to one side. The
    # visible heel is on the opposite side. Determine that side from the shift
    # of the component centre, then locate the strongest heel-to-ankle turn.
    span = last_y - toe_y
    mid_start = int(row_count * 0.45)
    mid_end = int(row_count * 0.65)
    bottom_start = int(row_count * 0.88)
    mid_center = float(np.median(row_centers[mid_start:mid_end]))
    bottom_center = float(np.median(row_centers[bottom_start:]))
    leg_bends_left = bottom_center < mid_center
    boundary = right_boundary if leg_bends_left else left_boundary
    smoothed = _smooth_boundary(boundary, sigma_px=pixels_per_mm * 2.4)
    derivative = np.gradient(smoothed)

    search_start_y = int(round(toe_y + span * 0.78))
    search_end_y = min(
        last_y - int(round(pixels_per_mm * 2.0)),
        rectified.measurement_roi_px[3] - int(round(pixels_per_mm * 2.0)),
    )
    search_start = max(0, search_start_y - first_y)
    search_end = min(row_count, search_end_y - first_y + 1)
    if search_end - search_start < int(pixels_per_mm * 10):
        raise PencilMeasurementError("Heel search region is too short")
    local_derivative = derivative[search_start:search_end]
    if leg_bends_left:
        heel_offset = search_start + int(np.argmin(local_derivative))
    else:
        heel_offset = search_start + int(np.argmax(local_derivative))
    heel_y = float(first_y + heel_offset)
    heel_x = float(smoothed[heel_offset])

    toe = np.array([toe_x, float(toe_y)], dtype=np.float32)
    heel = np.array([heel_x, heel_y], dtype=np.float32)
    length_mm = float(np.linalg.norm(heel - toe) / pixels_per_mm)
    widths = right_boundary - left_boundary
    foot_end_offset = min(row_count, heel_offset + 1)
    width_mm = float(
        np.percentile(widths[:foot_end_offset], 98) / pixels_per_mm
    )
    if not 180.0 <= length_mm <= 320.0:
        raise PencilMeasurementError(f"Implausible foot length: {length_mm:.1f} mm")
    if not 50.0 <= width_mm <= 140.0:
        raise PencilMeasurementError(f"Implausible foot width: {width_mm:.1f} mm")

    foot_only_mask = mask.copy()
    foot_only_mask[int(round(heel_y)) + 1 :] = 0
    return FootThresholdEstimate(
        threshold=float(threshold),
        length_mm=length_mm,
        width_mm=width_mm,
        toe_rectified_px=toe,
        heel_rectified_px=heel,
        mask=foot_only_mask,
    )


def measure_socked_foot(
    rectified: RectifiedBoard,
    *,
    thresholds: Sequence[float] = (11.0, 13.0, 15.0, 18.0),
) -> StaticFootMeasurement:
    """Measure a single socked foot in the left bay of the supplied board.

    The leg is connected to the foot, so the heel is inferred from the strongest
    turn of the exposed heel boundary rather than using the component's longest
    dimension. Several Lab colour-distance thresholds are combined to expose
    sensitivity to the pale sock/white board segmentation.
    """

    image = rectified.image_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = _row_background_lab(lab, rectified.pixels_per_mm)
    colour_distance = np.sqrt(np.sum((lab - background[:, None, :]) ** 2, axis=2))
    roi = np.zeros(image.shape[:2], dtype=np.uint8)
    left = int(round(45.0 * rectified.pixels_per_mm))
    right = int(round(205.0 * rectified.pixels_per_mm))
    top = int(round(45.0 * rectified.pixels_per_mm))
    bottom = int(round(307.0 * rectified.pixels_per_mm))
    cv2.rectangle(roi, (left, top), (right, bottom), 255, -1)

    opening_size = _odd(rectified.pixels_per_mm * 1.0)
    closing_size = _odd(rectified.pixels_per_mm * 3.0)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    estimates: list[FootThresholdEstimate] = []
    for threshold in thresholds:
        raw = np.where((colour_distance > threshold) & (roi > 0), 255, 0).astype(
            np.uint8
        )
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, opening_kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, closing_kernel)
        try:
            component = _largest_filled_component(raw)
            estimates.append(_estimate_from_mask(component, rectified, threshold))
        except PencilMeasurementError:
            continue
    if len(estimates) < 3:
        raise PencilMeasurementError(
            f"Only {len(estimates)} stable foot segmentations; need at least 3"
        )

    lengths = np.array([item.length_mm for item in estimates], dtype=np.float64)
    median_length = float(np.median(lengths))
    representative = min(estimates, key=lambda item: abs(item.length_mm - median_length))
    spread = float(np.ptp(lengths))
    if spread > 8.0:
        raise PencilMeasurementError(
            f"Foot segmentation is unstable across thresholds ({spread:.1f} mm)"
        )
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    toe_world = representative.toe_rectified_px / rectified.pixels_per_mm + origin
    heel_world = representative.heel_rectified_px / rectified.pixels_per_mm + origin
    return StaticFootMeasurement(
        length_mm=median_length,
        width_mm=float(np.median([item.width_mm for item in estimates])),
        threshold_spread_mm=spread,
        toe_world_mm=toe_world,
        heel_world_mm=heel_world,
        toe_rectified_px=representative.toe_rectified_px,
        heel_rectified_px=representative.heel_rectified_px,
        mask=representative.mask,
        per_threshold_lengths_mm={item.threshold: item.length_mm for item in estimates},
        measurement_method="inferred_heel",
    )


def _measure_foot_from_start_line_response(
    rectified: RectifiedBoard,
    response: np.ndarray,
    *,
    thresholds: Sequence[float],
    stable_spread_mm: float,
    segmentation_method: str,
    roi_world_mm: tuple[float, float, float, float] = LEGACY_SINGLE_FOOT_ROI_MM,
    start_line: DetectedStartLine | None = None,
) -> StaticFootMeasurement:
    if stable_spread_mm <= 0:
        raise PencilMeasurementError("stable_spread_mm must be positive")
    if start_line is None:
        start_line = detect_red_start_line(rectified)
    image = rectified.image_bgr
    pixels_per_mm = rectified.pixels_per_mm
    if response.shape != image.shape[:2]:
        raise PencilMeasurementError("Segmentation response has the wrong shape")

    roi, _ = _roi_mask(rectified, roi_world_mm)
    y_grid, x_grid = np.indices(image.shape[:2], dtype=np.float32)
    distance_to_line = (
        (x_grid - start_line.point_rectified_px[0])
        * start_line.normal_to_foot_rectified_px[0]
        + (y_grid - start_line.point_rectified_px[1])
        * start_line.normal_to_foot_rectified_px[1]
    )
    # Removing a narrow band around the line prevents the red stroke from
    # joining the foot component to the otherwise empty right side of the board.
    roi[distance_to_line <= 3.0 * pixels_per_mm] = 0

    opening_size = _odd(pixels_per_mm * 1.0)
    closing_size = _odd(pixels_per_mm * 3.0)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    estimates: list[FootThresholdEstimate] = []
    for threshold in thresholds:
        raw = np.where((response > threshold) & (roi > 0), 255, 0).astype(
            np.uint8
        )
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, opening_kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, closing_kernel)
        try:
            component = _largest_filled_component(raw)
            estimates.append(
                _estimate_from_start_line_mask(
                    component, rectified, start_line, threshold
                )
            )
        except PencilMeasurementError:
            continue
    stable_estimates = _longest_stable_estimate_window(
        estimates, stable_spread_mm
    )
    if len(stable_estimates) < 3:
        raise PencilMeasurementError(
            "Could not find a stable toe boundary across colour thresholds"
        )

    lengths = np.array(
        [item.length_mm for item in stable_estimates], dtype=np.float64
    )
    median_length = float(np.median(lengths))
    representative = min(
        stable_estimates, key=lambda item: abs(item.length_mm - median_length)
    )
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    toe_world = representative.toe_rectified_px / pixels_per_mm + origin
    heel_world = representative.heel_rectified_px / pixels_per_mm + origin
    line_world = start_line.endpoints_rectified_px / pixels_per_mm + origin
    return StaticFootMeasurement(
        length_mm=median_length,
        width_mm=float(
            np.median([item.width_mm for item in stable_estimates])
        ),
        threshold_spread_mm=float(np.ptp(lengths)),
        toe_world_mm=toe_world,
        heel_world_mm=heel_world,
        toe_rectified_px=representative.toe_rectified_px,
        heel_rectified_px=representative.heel_rectified_px,
        mask=representative.mask,
        per_threshold_lengths_mm={
            item.threshold: item.length_mm for item in stable_estimates
        },
        measurement_method="red_start_line",
        segmentation_method=segmentation_method,
        start_line_world_mm=line_world,
        start_line_rectified_px=start_line.endpoints_rectified_px,
        start_line_angle_degrees=start_line.angle_degrees,
        start_line_support_mm=start_line.support_mm,
        start_line_residual_p95_mm=start_line.residual_p95_mm,
    )


def measure_socked_foot_from_start_line(
    rectified: RectifiedBoard,
    *,
    thresholds: Sequence[float] = (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0),
    stable_spread_mm: float = 2.0,
    roi_world_mm: tuple[float, float, float, float] = LEGACY_SINGLE_FOOT_ROI_MM,
    start_line: DetectedStartLine | None = None,
) -> StaticFootMeasurement:
    """Measure a pale sock using a detected red heel reference line.

    Low colour-distance thresholds often include the pale shadow in front of a
    white sock. The longest stable consecutive threshold window is therefore
    used instead of treating the shadow boundary as the toe.
    """

    image = rectified.image_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = _row_background_lab(lab, rectified.pixels_per_mm)
    colour_distance = np.sqrt(
        np.sum((lab - background[:, None, :]) ** 2, axis=2)
    )
    return _measure_foot_from_start_line_response(
        rectified,
        colour_distance,
        thresholds=thresholds,
        stable_spread_mm=stable_spread_mm,
        segmentation_method="lab_colour_distance",
        roi_world_mm=roi_world_mm,
        start_line=start_line,
    )


def measure_bare_foot_from_start_line(
    rectified: RectifiedBoard,
    *,
    saturation_thresholds: Sequence[float] = (
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
        30.0,
        32.0,
        35.0,
    ),
    stable_spread_mm: float = 2.0,
    roi_world_mm: tuple[float, float, float, float] = LEGACY_SINGLE_FOOT_ROI_MM,
    start_line: DetectedStartLine | None = None,
) -> StaticFootMeasurement:
    """Measure bare skin against the white board using HSV saturation."""

    hsv = cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    return _measure_foot_from_start_line_response(
        rectified,
        saturation,
        thresholds=saturation_thresholds,
        stable_spread_mm=stable_spread_mm,
        segmentation_method="hsv_saturation",
        roi_world_mm=roi_world_mm,
        start_line=start_line,
    )


def measure_foot_from_start_line(rectified: RectifiedBoard) -> StaticFootMeasurement:
    """Automatically choose pale-sock or bare-skin start-line segmentation."""

    sock_measurement: StaticFootMeasurement | None = None
    bare_measurement: StaticFootMeasurement | None = None
    sock_error: PencilMeasurementError | None = None
    bare_error: PencilMeasurementError | None = None
    try:
        sock_measurement = measure_socked_foot_from_start_line(rectified)
    except PencilMeasurementError as error:
        sock_error = error
    try:
        bare_measurement = measure_bare_foot_from_start_line(rectified)
    except PencilMeasurementError as error:
        bare_error = error

    if sock_measurement is None and bare_measurement is None:
        raise PencilMeasurementError(
            "Start-line foot segmentation failed for both sock and bare-foot "
            f"modes (sock: {sock_error}; bare: {bare_error})"
        ) from bare_error
    if sock_measurement is None:
        assert bare_measurement is not None
        return bare_measurement
    if bare_measurement is None:
        return sock_measurement

    def mask_solidity(measurement: StaticFootMeasurement) -> float:
        contours, _ = cv2.findContours(
            measurement.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            return 0.0
        return float(cv2.contourArea(contour) / hull_area)

    # A wrong colour model can still produce stable toe coordinates while
    # attaching thin board/shadow streaks to the foot.  Prefer the cleaner
    # silhouette when the solidity difference is material; otherwise retain
    # the threshold-stability criterion.
    sock_solidity = mask_solidity(sock_measurement)
    bare_solidity = mask_solidity(bare_measurement)
    if abs(sock_solidity - bare_solidity) >= 0.03:
        return (
            sock_measurement
            if sock_solidity > bare_solidity
            else bare_measurement
        )
    return min(
        (sock_measurement, bare_measurement),
        key=lambda item: item.threshold_spread_mm,
    )


def _board_background_lab(rectified: RectifiedBoard) -> np.ndarray:
    """Estimate row-wise white-board colour without assuming either bay is empty."""

    image = rectified.image_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    usable = (saturation <= 50) & (value >= 135)
    rows = np.full((image.shape[0], 3), np.nan, dtype=np.float32)
    minimum_samples = max(20, int(round(image.shape[1] * 0.12)))
    for row in range(image.shape[0]):
        pixels = lab[row, usable[row]]
        if len(pixels) >= minimum_samples:
            rows[row] = np.median(pixels, axis=0)
    valid_rows = np.flatnonzero(np.isfinite(rows[:, 0]))
    if len(valid_rows) < image.shape[0] * 0.25:
        raise PencilMeasurementError(
            "Not enough visible white board to estimate the background"
        )
    all_rows = np.arange(image.shape[0])
    for channel in range(3):
        rows[:, channel] = np.interp(
            all_rows, valid_rows, rows[valid_rows, channel]
        )
    return rows


def _measure_foot_from_inferred_response(
    rectified: RectifiedBoard,
    response: np.ndarray,
    *,
    thresholds: Sequence[float],
    stable_spread_mm: float,
    segmentation_method: str,
    roi_world_mm: tuple[float, float, float, float],
) -> StaticFootMeasurement:
    if stable_spread_mm <= 0:
        raise PencilMeasurementError("stable_spread_mm must be positive")
    if response.shape != rectified.image_bgr.shape[:2]:
        raise PencilMeasurementError("Segmentation response has the wrong shape")

    roi, bounds_px = _roi_mask(rectified, roi_world_mm)
    local_rectified = replace(rectified, measurement_roi_px=bounds_px)
    pixels_per_mm = rectified.pixels_per_mm
    opening_size = _odd(pixels_per_mm * 1.0)
    closing_size = _odd(pixels_per_mm * 3.0)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )

    estimates: list[FootThresholdEstimate] = []
    for threshold in thresholds:
        raw = np.where((response > threshold) & (roi > 0), 255, 0).astype(
            np.uint8
        )
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, opening_kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, closing_kernel)
        try:
            component = _largest_filled_component(raw)
            estimates.append(
                _estimate_from_mask(component, local_rectified, threshold)
            )
        except PencilMeasurementError:
            continue

    stable_estimates = _longest_stable_estimate_window(
        estimates, stable_spread_mm
    )
    if len(stable_estimates) < 3:
        raise PencilMeasurementError(
            f"Only {len(estimates)} plausible segmentations and no stable "
            "three-threshold window"
        )
    lengths = np.array(
        [item.length_mm for item in stable_estimates], dtype=np.float64
    )
    median_length = float(np.median(lengths))
    representative = min(
        stable_estimates, key=lambda item: abs(item.length_mm - median_length)
    )
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    toe_world = representative.toe_rectified_px / pixels_per_mm + origin
    heel_world = representative.heel_rectified_px / pixels_per_mm + origin
    return StaticFootMeasurement(
        length_mm=median_length,
        width_mm=float(
            np.median([item.width_mm for item in stable_estimates])
        ),
        threshold_spread_mm=float(np.ptp(lengths)),
        toe_world_mm=toe_world,
        heel_world_mm=heel_world,
        toe_rectified_px=representative.toe_rectified_px,
        heel_rectified_px=representative.heel_rectified_px,
        mask=representative.mask,
        per_threshold_lengths_mm={
            item.threshold: item.length_mm for item in stable_estimates
        },
        measurement_method="inferred_heel",
        segmentation_method=segmentation_method,
    )


def measure_bare_foot_in_roi(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
    *,
    saturation_thresholds: Sequence[float] = (
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
        30.0,
        32.0,
        35.0,
        40.0,
        45.0,
    ),
    stable_spread_mm: float = 8.0,
) -> StaticFootMeasurement:
    """Measure one bare foot inside an explicit board bay without a red line."""

    hsv = cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    return _measure_foot_from_inferred_response(
        rectified,
        saturation,
        thresholds=saturation_thresholds,
        stable_spread_mm=stable_spread_mm,
        segmentation_method="hsv_saturation_roi",
        roi_world_mm=roi_world_mm,
    )


def measure_socked_foot_in_roi(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
    *,
    thresholds: Sequence[float] = (
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
    ),
    stable_spread_mm: float = 8.0,
) -> StaticFootMeasurement:
    """Measure one socked foot using board-wide, foot-independent background."""

    lab = cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = _board_background_lab(rectified)
    colour_distance = np.sqrt(
        np.sum((lab - background[:, None, :]) ** 2, axis=2)
    )
    return _measure_foot_from_inferred_response(
        rectified,
        colour_distance,
        thresholds=thresholds,
        stable_spread_mm=stable_spread_mm,
        segmentation_method="lab_colour_distance_board",
        roi_world_mm=roi_world_mm,
    )


def measure_foot_from_start_line_in_roi(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
    start_line: DetectedStartLine,
) -> StaticFootMeasurement:
    """Measure one bay against a start line shared by all feet in the frame."""

    bare_error: PencilMeasurementError | None = None
    try:
        return measure_bare_foot_from_start_line(
            rectified,
            roi_world_mm=roi_world_mm,
            start_line=start_line,
        )
    except PencilMeasurementError as error:
        bare_error = error

    lab = cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = _board_background_lab(rectified)
    colour_distance = np.sqrt(
        np.sum((lab - background[:, None, :]) ** 2, axis=2)
    )
    try:
        return _measure_foot_from_start_line_response(
            rectified,
            colour_distance,
            thresholds=(10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0),
            stable_spread_mm=2.0,
            segmentation_method="lab_colour_distance_board",
            roi_world_mm=roi_world_mm,
            start_line=start_line,
        )
    except PencilMeasurementError as sock_error:
        raise PencilMeasurementError(
            "Foot segmentation failed in both bare and sock modes "
            f"(bare: {bare_error}; sock: {sock_error})"
        ) from sock_error


def measure_feet_in_bays(
    rectified: RectifiedBoard,
    *,
    sides: Sequence[str] = ("image_left", "image_right"),
    mode: str = "auto",
) -> dict[str, StaticFootMeasurement]:
    """Measure requested board bays independently from one rectified frame.

    ``image_left`` and ``image_right`` are locations after any requested mirror
    correction.  Every requested bay must contain a valid foot; otherwise the
    call fails instead of silently returning a one-foot result for a two-foot
    request.
    """

    requested = tuple(dict.fromkeys(sides))
    if not requested:
        raise PencilMeasurementError("At least one foot bay must be requested")
    unknown = [side for side in requested if side not in FOOT_BAY_ROIS_MM]
    if unknown:
        raise PencilMeasurementError("Unknown foot bay(s): " + ", ".join(unknown))
    if mode not in {"auto", "start-line", "inferred-heel"}:
        raise PencilMeasurementError(f"Unknown foot measurement mode: {mode}")

    start_line: DetectedStartLine | None = None
    line_error: PencilMeasurementError | None = None
    if mode != "inferred-heel":
        try:
            start_line = detect_red_start_line(rectified)
        except PencilMeasurementError as error:
            line_error = error
            if mode == "start-line":
                raise

    measurements: dict[str, StaticFootMeasurement] = {}
    failures: dict[str, str] = {}
    for side in requested:
        roi_world_mm = FOOT_BAY_ROIS_MM[side]
        if start_line is not None:
            try:
                measurements[side] = measure_foot_from_start_line_in_roi(
                    rectified, roi_world_mm, start_line
                )
                continue
            except PencilMeasurementError as error:
                if mode == "start-line":
                    failures[side] = str(error)
                    continue

        bare_error: PencilMeasurementError | None = None
        try:
            measurements[side] = measure_bare_foot_in_roi(
                rectified, roi_world_mm
            )
            continue
        except PencilMeasurementError as error:
            bare_error = error
        try:
            measurements[side] = measure_socked_foot_in_roi(
                rectified, roi_world_mm
            )
        except PencilMeasurementError as sock_error:
            detail = f"bare: {bare_error}; sock: {sock_error}"
            if line_error is not None:
                detail = f"start line: {line_error}; {detail}"
            failures[side] = detail

    if failures:
        details = "; ".join(
            f"{side} ({reason})" for side, reason in failures.items()
        )
        raise PencilMeasurementError(
            f"Could not measure every requested foot bay: {details}"
        )
    return measurements


def _estimate_from_marker_edge_mask(
    mask: np.ndarray,
    rectified: RectifiedBoard,
    *,
    reference_edge_y_mm: float,
    fixed_offset_mm: float,
    visible_length_scale: float,
    threshold: float,
) -> FootThresholdEstimate:
    """Estimate total length while deliberately ignoring the hidden heel area."""

    pixels_per_mm = rectified.pixels_per_mm
    origin_x, origin_y = rectified.world_origin_mm
    reference_y_px = (reference_edge_y_mm - origin_y) * pixels_per_mm
    y_coordinates, x_coordinates = np.where(mask > 0)
    if len(x_coordinates) < pixels_per_mm**2 * 2_000:
        raise PencilMeasurementError("Detected visible foot region is too small")
    if float(y_coordinates.max()) < reference_y_px - 7.0 * pixels_per_mm:
        raise PencilMeasurementError(
            "Foot does not reach the heel-side marker reference edge"
        )

    toe_y = int(y_coordinates.min())
    toe_band_height = max(2, int(round(pixels_per_mm)))
    toe_band = x_coordinates[y_coordinates <= toe_y + toe_band_height]
    if len(toe_band) < max(3, int(round(pixels_per_mm))):
        raise PencilMeasurementError("Toe boundary is too small or noisy")
    toe_x = float(np.median(toe_band))
    raw_visible_length_mm = float(reference_y_px - toe_y) / pixels_per_mm
    visible_length_mm = raw_visible_length_mm * visible_length_scale
    total_length_mm = visible_length_mm + fixed_offset_mm

    widths: list[float] = []
    for row in np.unique(y_coordinates):
        row_x = x_coordinates[y_coordinates == row]
        if len(row_x) >= 2:
            widths.append(float(row_x.max() - row_x.min()))
    if not widths:
        raise PencilMeasurementError("Visible foot width could not be estimated")
    width_mm = float(np.percentile(widths, 98) / pixels_per_mm)
    if not 35.0 <= raw_visible_length_mm <= 180.0:
        raise PencilMeasurementError(
            f"Implausible visible foot length: {raw_visible_length_mm:.1f} mm"
        )
    if not 180.0 <= total_length_mm <= 320.0:
        raise PencilMeasurementError(
            f"Implausible total foot length: {total_length_mm:.1f} mm"
        )
    if not 50.0 <= width_mm <= 140.0:
        raise PencilMeasurementError(f"Implausible foot width: {width_mm:.1f} mm")

    toe = np.array([toe_x, float(toe_y)], dtype=np.float32)
    virtual_heel = np.array(
        [toe_x, float(toe_y) + total_length_mm * pixels_per_mm],
        dtype=np.float32,
    )
    visible_mask = mask.copy()
    visible_mask[int(np.ceil(reference_y_px)) :] = 0
    return FootThresholdEstimate(
        threshold=float(threshold),
        length_mm=total_length_mm,
        width_mm=width_mm,
        toe_rectified_px=toe,
        heel_rectified_px=virtual_heel,
        mask=visible_mask,
    )


def measure_foot_from_marker_edge(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
    *,
    reference_edge_y_mm: float,
    fixed_offset_mm: float = MARKER_OFFSET_FIXED_LENGTH_MM,
    visible_length_scale: float = MARKER_OFFSET_VISIBLE_LENGTH_SCALE,
    saturation_thresholds: Sequence[float] = (
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
        30.0,
        32.0,
        35.0,
        40.0,
        45.0,
        50.0,
    ),
    stable_spread_mm: float = 3.0,
) -> StaticFootMeasurement:
    """Measure toe-to-marker length and add a known heel-side offset.

    Only pixels on the toe-facing side of ``reference_edge_y_mm`` contribute to
    segmentation.  Anything behind the black marker's outer edge is therefore
    ignored, and ``fixed_offset_mm`` is added exactly once to the visible part.
    """

    if fixed_offset_mm <= 0:
        raise PencilMeasurementError("Fixed heel-side offset must be positive")
    if not 0.5 <= visible_length_scale <= 1.5:
        raise PencilMeasurementError(
            "Visible-length calibration factor must be between 0.5 and 1.5"
        )
    if stable_spread_mm <= 0:
        raise PencilMeasurementError("stable_spread_mm must be positive")
    left_mm, top_mm, right_mm, _ = roi_world_mm
    measurement_roi_mm = (
        left_mm,
        max(20.0, top_mm),
        right_mm,
        reference_edge_y_mm,
    )
    roi, bounds_px = _roi_mask(rectified, measurement_roi_mm)
    pixels_per_mm = rectified.pixels_per_mm
    reference_y_px = int(round(
        (reference_edge_y_mm - rectified.world_origin_mm[1]) * pixels_per_mm
    ))
    # A 3 mm guard band prevents marker-edge compression colours and any skin
    # behind the declared boundary from joining the visible-foot component.
    roi[max(0, reference_y_px - int(round(3.0 * pixels_per_mm))) :] = 0

    hsv = cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(
        hsv[:, :, 1].astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0
    )
    value = hsv[:, :, 2]
    saturation[value < 45] = 0.0
    opening_size = _odd(pixels_per_mm * 1.0)
    closing_size = _odd(pixels_per_mm * 3.0)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )

    estimates: list[FootThresholdEstimate] = []
    for threshold in saturation_thresholds:
        raw = np.where((saturation > threshold) & (roi > 0), 255, 0).astype(
            np.uint8
        )
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, opening_kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, closing_kernel)
        try:
            component = _largest_filled_component(raw)
            estimates.append(
                _estimate_from_marker_edge_mask(
                    component,
                    rectified,
                    reference_edge_y_mm=reference_edge_y_mm,
                    fixed_offset_mm=fixed_offset_mm,
                    visible_length_scale=visible_length_scale,
                    threshold=threshold,
                )
            )
        except PencilMeasurementError:
            continue

    stable_estimates = _longest_stable_estimate_window(
        estimates, stable_spread_mm
    )
    if len(stable_estimates) < 3:
        raise PencilMeasurementError(
            "Could not find a stable toe boundary for the marker-offset method"
        )
    return _finalize_marker_edge_measurement(
        rectified,
        stable_estimates,
        reference_edge_y_mm=reference_edge_y_mm,
        fixed_offset_mm=fixed_offset_mm,
        visible_length_scale=visible_length_scale,
        segmentation_method="hsv_saturation_marker_roi",
    )


def _finalize_marker_edge_measurement(
    rectified: RectifiedBoard,
    estimates: Sequence[FootThresholdEstimate],
    *,
    reference_edge_y_mm: float,
    fixed_offset_mm: float,
    visible_length_scale: float,
    segmentation_method: str,
    segmentation_confidence: float | None = None,
    segmentation_model: str | None = None,
) -> StaticFootMeasurement:
    """Convert one or more validated masks into a metric foot measurement."""

    if not estimates:
        raise PencilMeasurementError("No valid marker-edge foot estimates")
    pixels_per_mm = rectified.pixels_per_mm
    lengths = np.array([item.length_mm for item in estimates], dtype=np.float64)
    median_length = float(np.median(lengths))
    representative = min(
        estimates, key=lambda item: abs(item.length_mm - median_length)
    )
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    toe_world = representative.toe_rectified_px / pixels_per_mm + origin
    heel_world = representative.heel_rectified_px / pixels_per_mm + origin
    reference_point_px = np.array(
        [
            representative.toe_rectified_px[0],
            (reference_edge_y_mm - origin[1]) * pixels_per_mm,
        ],
        dtype=np.float32,
    )
    reference_point_world = reference_point_px / pixels_per_mm + origin
    board_left_mm = float(origin[0])
    board_right_mm = float(
        origin[0] + rectified.image_bgr.shape[1] / pixels_per_mm
    )
    reference_edge_world = np.array(
        [
            [board_left_mm, reference_edge_y_mm],
            [board_right_mm, reference_edge_y_mm],
        ],
        dtype=np.float32,
    )
    return StaticFootMeasurement(
        length_mm=median_length,
        width_mm=float(np.median([item.width_mm for item in estimates])),
        threshold_spread_mm=float(np.ptp(lengths)),
        toe_world_mm=toe_world,
        heel_world_mm=heel_world,
        toe_rectified_px=representative.toe_rectified_px,
        heel_rectified_px=representative.heel_rectified_px,
        mask=representative.mask,
        per_threshold_lengths_mm={
            item.threshold: item.length_mm for item in estimates
        },
        measurement_method="marker_edge_offset",
        segmentation_method=segmentation_method,
        segmentation_confidence=segmentation_confidence,
        segmentation_model=segmentation_model,
        segmentation_mask_area_mm2=(
            float(np.count_nonzero(representative.mask)) / pixels_per_mm**2
        ),
        visible_length_mm=median_length - fixed_offset_mm,
        raw_visible_length_mm=(
            (median_length - fixed_offset_mm) / visible_length_scale
        ),
        visible_length_scale=float(visible_length_scale),
        fixed_offset_mm=float(fixed_offset_mm),
        reference_edge_world_mm=reference_edge_world,
        reference_point_world_mm=reference_point_world,
        reference_point_rectified_px=reference_point_px,
    )


def measure_foot_from_marker_edge_mask(
    rectified: RectifiedBoard,
    roi_world_mm: tuple[float, float, float, float],
    foot_mask: np.ndarray,
    *,
    reference_edge_y_mm: float,
    fixed_offset_mm: float = MARKER_OFFSET_FIXED_LENGTH_MM,
    visible_length_scale: float = MARKER_OFFSET_VISIBLE_LENGTH_SCALE,
    mask_threshold: float = 0.5,
    segmentation_method: str = "external_instance_segmentation",
    segmentation_confidence: float | None = None,
    segmentation_model: str | None = None,
) -> StaticFootMeasurement:
    """Measure a supplied foot mask without colour morphology or contour fill."""

    if foot_mask.shape != rectified.image_bgr.shape[:2]:
        raise PencilMeasurementError("External foot mask shape is inconsistent")
    if fixed_offset_mm <= 0:
        raise PencilMeasurementError("Fixed heel-side offset must be positive")
    if not 0.5 <= visible_length_scale <= 1.5:
        raise PencilMeasurementError(
            "Visible-length calibration factor must be between 0.5 and 1.5"
        )
    left_mm, top_mm, right_mm, _ = roi_world_mm
    measurement_roi_mm = (
        left_mm,
        max(20.0, top_mm),
        right_mm,
        reference_edge_y_mm,
    )
    roi, _ = _roi_mask(rectified, measurement_roi_mm)
    binary = np.where((foot_mask > 0) & (roi > 0), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        raise PencilMeasurementError("External foot mask is empty in its bay")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    estimate = _estimate_from_marker_edge_mask(
        component,
        rectified,
        reference_edge_y_mm=reference_edge_y_mm,
        fixed_offset_mm=fixed_offset_mm,
        visible_length_scale=visible_length_scale,
        threshold=mask_threshold,
    )
    return _finalize_marker_edge_measurement(
        rectified,
        [estimate],
        reference_edge_y_mm=reference_edge_y_mm,
        fixed_offset_mm=fixed_offset_mm,
        visible_length_scale=visible_length_scale,
        segmentation_method=segmentation_method,
        segmentation_confidence=segmentation_confidence,
        segmentation_model=segmentation_model,
    )


def measure_feet_from_marker_edge_masks(
    rectified: RectifiedBoard,
    masks_by_side: Mapping[str, np.ndarray],
    sides: Sequence[str],
    reference_edge_y_mm: float,
    *,
    fixed_offset_mm: float = MARKER_OFFSET_FIXED_LENGTH_MM,
    visible_length_scale: float = MARKER_OFFSET_VISIBLE_LENGTH_SCALE,
    mask_threshold: float = 0.5,
    confidence_by_side: Mapping[str, float] | None = None,
    segmentation_method: str = "external_instance_segmentation",
    segmentation_model: str | None = None,
) -> dict[str, StaticFootMeasurement]:
    """Measure requested bays from externally supplied instance masks."""

    requested = tuple(dict.fromkeys(sides))
    unknown = [side for side in requested if side not in FOOT_BAY_ROIS_MM]
    if unknown:
        raise PencilMeasurementError("Unknown foot bay(s): " + ", ".join(unknown))
    missing = [side for side in requested if side not in masks_by_side]
    if missing:
        raise PencilMeasurementError(
            "External segmentation is missing foot bay(s): " + ", ".join(missing)
        )
    measurements: dict[str, StaticFootMeasurement] = {}
    failures: dict[str, str] = {}
    for side in requested:
        try:
            measurements[side] = measure_foot_from_marker_edge_mask(
                rectified,
                FOOT_BAY_ROIS_MM[side],
                masks_by_side[side],
                reference_edge_y_mm=reference_edge_y_mm,
                fixed_offset_mm=fixed_offset_mm,
                visible_length_scale=visible_length_scale,
                mask_threshold=mask_threshold,
                segmentation_method=segmentation_method,
                segmentation_confidence=(
                    None
                    if confidence_by_side is None
                    else confidence_by_side.get(side)
                ),
                segmentation_model=segmentation_model,
            )
        except PencilMeasurementError as error:
            failures[side] = str(error)
    if failures:
        details = "; ".join(
            f"{side} ({reason})" for side, reason in failures.items()
        )
        raise PencilMeasurementError(
            "Could not measure every external foot mask: " + details
        )
    return measurements


def measure_feet_from_marker_edge(
    rectified: RectifiedBoard,
    sides: Sequence[str] = ("image_left", "image_right"),
    reference_edge_y_mm: float | None = None,
    *,
    fixed_offset_mm: float = MARKER_OFFSET_FIXED_LENGTH_MM,
    visible_length_scale: float = MARKER_OFFSET_VISIBLE_LENGTH_SCALE,
) -> dict[str, StaticFootMeasurement]:
    """Measure all requested bays using one black-marker outer edge."""

    requested = tuple(dict.fromkeys(sides))
    if not requested:
        raise PencilMeasurementError("At least one foot bay must be requested")
    if reference_edge_y_mm is None:
        raise PencilMeasurementError("Marker reference-edge y coordinate is required")
    unknown = [side for side in requested if side not in FOOT_BAY_ROIS_MM]
    if unknown:
        raise PencilMeasurementError("Unknown foot bay(s): " + ", ".join(unknown))
    measurements: dict[str, StaticFootMeasurement] = {}
    failures: dict[str, str] = {}
    for side in requested:
        try:
            measurements[side] = measure_foot_from_marker_edge(
                rectified,
                FOOT_BAY_ROIS_MM[side],
                reference_edge_y_mm=reference_edge_y_mm,
                fixed_offset_mm=fixed_offset_mm,
                visible_length_scale=visible_length_scale,
            )
        except PencilMeasurementError as error:
            failures[side] = str(error)
    if failures:
        details = "; ".join(
            f"{side} ({reason})" for side, reason in failures.items()
        )
        raise PencilMeasurementError(
            f"Could not measure every requested marker-offset foot: {details}"
        )
    return measurements


def annotate_static_foot_original(
    image_bgr: np.ndarray,
    corners_by_id: Mapping[int, np.ndarray],
    calibration: BoardHomography,
    measurement: StaticFootMeasurement,
) -> np.ndarray:
    annotated = image_bgr.copy()
    inlier_ids = set(calibration.inlier_marker_ids)
    for marker_id, corners in corners_by_id.items():
        points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        colour = (40, 210, 40) if marker_id in inlier_ids else (30, 30, 230)
        cv2.polylines(annotated, [points], True, colour, 5, cv2.LINE_AA)
    if measurement.start_line_world_mm is not None:
        image_line = cv2.perspectiveTransform(
            np.asarray(measurement.start_line_world_mm, dtype=np.float32).reshape(
                1, -1, 2
            ),
            calibration.world_to_image,
        ).reshape(-1, 2)
        image_line = np.rint(image_line).astype(int)
        cv2.line(
            annotated,
            tuple(image_line[0]),
            tuple(image_line[1]),
            (255, 180, 0),
            6,
            cv2.LINE_AA,
        )
    world_endpoints = np.float32([measurement.toe_world_mm, measurement.heel_world_mm])
    image_endpoints = cv2.perspectiveTransform(
        world_endpoints.reshape(1, -1, 2), calibration.world_to_image
    ).reshape(-1, 2)
    endpoints = np.rint(image_endpoints).astype(int)
    cv2.line(
        annotated,
        tuple(endpoints[0]),
        tuple(endpoints[1]),
        (0, 0, 255),
        8,
        cv2.LINE_AA,
    )
    for endpoint in endpoints:
        cv2.circle(annotated, tuple(endpoint), 18, (0, 0, 255), -1, cv2.LINE_AA)
    label = endpoints.mean(axis=0).astype(int) + np.array([35, 0])
    method_label = (
        "start line"
        if measurement.measurement_method == "red_start_line"
        else "estimated heel"
    )
    cv2.putText(
        annotated,
        f"Foot: {measurement.length_mm / 10.0:.1f} cm ({method_label})",
        tuple(label),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return annotated


def annotate_static_foot_rectified(
    rectified: RectifiedBoard, measurement: StaticFootMeasurement
) -> np.ndarray:
    annotated = rectified.image_bgr.copy()

    # A metric grid makes the perspective-corrected plane visually explicit.
    grid = annotated.copy()
    grid_step = max(1, int(round(50.0 * rectified.pixels_per_mm)))
    for x in range(0, annotated.shape[1], grid_step):
        cv2.line(grid, (x, 0), (x, annotated.shape[0] - 1), (90, 90, 90), 1)
    for y in range(0, annotated.shape[0], grid_step):
        cv2.line(grid, (0, y), (annotated.shape[1] - 1, y), (90, 90, 90), 1)
    cv2.addWeighted(grid, 0.18, annotated, 0.82, 0, annotated)

    overlay = annotated.copy()
    overlay[measurement.mask > 0] = (0, 210, 255)
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    if measurement.start_line_rectified_px is not None:
        line = np.rint(measurement.start_line_rectified_px).astype(int)
        direction = line[1].astype(np.float64) - line[0].astype(np.float64)
        if abs(direction[0]) > 1e-6:
            roi_left, _, roi_right, _ = rectified.measurement_roi_px
            extended: list[tuple[int, int]] = []
            for x in (roi_left, roi_right):
                scale = (x - line[0, 0]) / direction[0]
                y = line[0, 1] + scale * direction[1]
                extended.append((int(round(x)), int(round(y))))
            cv2.line(
                annotated,
                extended[0],
                extended[1],
                (255, 150, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.line(
            annotated,
            tuple(line[0]),
            tuple(line[1]),
            (255, 255, 0),
            5,
            cv2.LINE_AA,
        )
    toe = np.rint(measurement.toe_rectified_px).astype(int)
    heel = np.rint(measurement.heel_rectified_px).astype(int)
    cv2.line(annotated, tuple(toe), tuple(heel), (0, 0, 255), 6, cv2.LINE_AA)
    cv2.circle(annotated, tuple(toe), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(annotated, tuple(heel), 12, (0, 0, 255), -1, cv2.LINE_AA)
    label = ((toe + heel) // 2) + np.array([25, 0])
    cv2.putText(
        annotated,
        f"{measurement.length_mm / 10.0:.1f} cm",
        tuple(label),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    legend = [("Top-down plane | grid: 50 mm", (24, 38), 0.75)]
    if measurement.start_line_support_mm is not None:
        partial_label = (
            " (partial; extrapolated)"
            if measurement.start_line_support_mm < 80.0
            else ""
        )
        legend.append(
            (
                f"Visible start line: {measurement.start_line_support_mm:.1f} mm"
                f"{partial_label}",
                (24, 70),
                0.65,
            )
        )
    for text, origin, scale in legend:
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    return annotated


_FOOT_ANNOTATION_COLOURS: dict[str, tuple[int, int, int]] = {
    "image_left": (30, 30, 235),
    "image_right": (235, 120, 20),
}


def combined_foot_mask(
    measurements: Mapping[str, StaticFootMeasurement],
) -> np.ndarray:
    """Return one binary mask containing every successfully measured foot."""

    if not measurements:
        raise PencilMeasurementError("No foot measurements to combine")
    masks = [measurement.mask for measurement in measurements.values()]
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise PencilMeasurementError("Foot masks have inconsistent shapes")
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        combined[mask > 0] = 255
    return combined


def annotate_static_feet_original(
    image_bgr: np.ndarray,
    corners_by_id: Mapping[int, np.ndarray],
    calibration: BoardHomography,
    measurements: Mapping[str, StaticFootMeasurement],
) -> np.ndarray:
    """Annotate multiple independently measured board bays in camera space."""

    if not measurements:
        raise PencilMeasurementError("No foot measurements to annotate")
    annotated = image_bgr.copy()
    inlier_ids = set(calibration.inlier_marker_ids)
    for marker_id, corners in corners_by_id.items():
        points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        colour = (40, 210, 40) if marker_id in inlier_ids else (30, 30, 230)
        cv2.polylines(annotated, [points], True, colour, 5, cv2.LINE_AA)

    first = next(iter(measurements.values()))
    if first.start_line_world_mm is not None:
        line = cv2.perspectiveTransform(
            np.asarray(first.start_line_world_mm, dtype=np.float32).reshape(
                1, -1, 2
            ),
            calibration.world_to_image,
        ).reshape(-1, 2)
        line = np.rint(line).astype(int)
        cv2.line(
            annotated,
            tuple(line[0]),
            tuple(line[1]),
            (255, 200, 0),
            6,
            cv2.LINE_AA,
        )

    for side, measurement in measurements.items():
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        world_endpoints = np.float32(
            [measurement.toe_world_mm, measurement.heel_world_mm]
        )
        image_endpoints = cv2.perspectiveTransform(
            world_endpoints.reshape(1, -1, 2), calibration.world_to_image
        ).reshape(-1, 2)
        endpoints = np.rint(image_endpoints).astype(int)
        cv2.line(
            annotated,
            tuple(endpoints[0]),
            tuple(endpoints[1]),
            colour,
            8,
            cv2.LINE_AA,
        )
        for endpoint in endpoints:
            cv2.circle(annotated, tuple(endpoint), 18, colour, -1, cv2.LINE_AA)
        label = endpoints.mean(axis=0).astype(int) + np.array([25, 0])
        display_side = side.replace("image_", "").upper()
        cv2.putText(
            annotated,
            f"{display_side}: {measurement.length_mm / 10.0:.1f} cm",
            tuple(label),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colour,
            4,
            cv2.LINE_AA,
        )
    return annotated


def annotate_static_feet_rectified(
    rectified: RectifiedBoard,
    measurements: Mapping[str, StaticFootMeasurement],
) -> np.ndarray:
    """Visualize multiple measurements on the perspective-corrected plane."""

    if not measurements:
        raise PencilMeasurementError("No foot measurements to annotate")
    annotated = rectified.image_bgr.copy()
    grid = annotated.copy()
    grid_step = max(1, int(round(50.0 * rectified.pixels_per_mm)))
    for x in range(0, annotated.shape[1], grid_step):
        cv2.line(grid, (x, 0), (x, annotated.shape[0] - 1), (90, 90, 90), 1)
    for y in range(0, annotated.shape[0], grid_step):
        cv2.line(grid, (0, y), (annotated.shape[1] - 1, y), (90, 90, 90), 1)
    cv2.addWeighted(grid, 0.18, annotated, 0.82, 0, annotated)

    for side, measurement in measurements.items():
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        overlay = annotated.copy()
        overlay[measurement.mask > 0] = colour
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    divider_x = int(round(
        (181.0 - rectified.world_origin_mm[0]) * rectified.pixels_per_mm
    ))
    cv2.line(
        annotated,
        (divider_x, 0),
        (divider_x, annotated.shape[0] - 1),
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    first = next(iter(measurements.values()))
    if first.start_line_rectified_px is not None:
        line = np.rint(first.start_line_rectified_px).astype(int)
        direction = line[1].astype(np.float64) - line[0].astype(np.float64)
        if abs(direction[0]) > 1e-6:
            extended: list[tuple[int, int]] = []
            for x in (0, annotated.shape[1] - 1):
                scale = (x - line[0, 0]) / direction[0]
                y = line[0, 1] + scale * direction[1]
                extended.append((x, int(round(y))))
            cv2.line(
                annotated,
                extended[0],
                extended[1],
                (255, 170, 0),
                3,
                cv2.LINE_AA,
            )
        cv2.line(
            annotated,
            tuple(line[0]),
            tuple(line[1]),
            (255, 255, 0),
            5,
            cv2.LINE_AA,
        )

    for side, measurement in measurements.items():
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        toe = np.rint(measurement.toe_rectified_px).astype(int)
        heel = np.rint(measurement.heel_rectified_px).astype(int)
        cv2.line(
            annotated, tuple(toe), tuple(heel), colour, 6, cv2.LINE_AA
        )
        cv2.circle(annotated, tuple(toe), 12, colour, -1, cv2.LINE_AA)
        cv2.circle(annotated, tuple(heel), 12, colour, -1, cv2.LINE_AA)
        label = ((toe + heel) // 2) + np.array([18, 0])
        display_side = side.replace("image_", "").upper()
        cv2.putText(
            annotated,
            f"{display_side} {measurement.length_mm / 10.0:.1f} cm",
            tuple(label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            3,
            cv2.LINE_AA,
        )

    for text, origin, scale in (
        ("Top-down plane | grid: 50 mm", (24, 38), 0.75),
        ("LEFT/RIGHT = corrected image bays", (24, 70), 0.65),
    ):
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    return annotated


def compose_marker_offset_rectified(
    rectified_by_side: Mapping[str, RectifiedBoard],
    measurements: Mapping[str, StaticFootMeasurement],
) -> tuple[RectifiedBoard, dict[str, StaticFootMeasurement]]:
    """Stitch independently calibrated left/right bay planes for visualization."""

    if not rectified_by_side or set(rectified_by_side) != set(measurements):
        raise PencilMeasurementError(
            "Rectified marker-offset bays and measurements do not match"
        )
    first = next(iter(rectified_by_side.values()))
    pixels_per_mm = first.pixels_per_mm
    if any(
        abs(board.pixels_per_mm - pixels_per_mm) > 1e-6
        for board in rectified_by_side.values()
    ):
        raise PencilMeasurementError("Marker-offset bay scales do not match")
    height = max(board.image_bgr.shape[0] for board in rectified_by_side.values())
    width = max(board.image_bgr.shape[1] for board in rectified_by_side.values())
    composite_image = np.full((height, width, 3), 235, dtype=np.uint8)
    # Start with a full plane so a one-bay request still produces a useful view.
    base = first.image_bgr
    composite_image[: base.shape[0], : base.shape[1]] = base
    divider = int(round(
        (181.0 - first.world_origin_mm[0]) * pixels_per_mm
    ))
    for side, board in rectified_by_side.items():
        source = board.image_bgr
        if side == "image_left":
            composite_image[: source.shape[0], :divider] = source[:, :divider]
        elif side == "image_right":
            composite_image[: source.shape[0], divider:source.shape[1]] = source[
                :, divider:
            ]

    composite = RectifiedBoard(
        image_bgr=composite_image,
        image_to_rectified=first.image_to_rectified,
        world_origin_mm=first.world_origin_mm,
        pixels_per_mm=pixels_per_mm,
        measurement_roi_px=(0, 0, width - 1, height - 1),
    )
    padded_measurements: dict[str, StaticFootMeasurement] = {}
    for side, measurement in measurements.items():
        padded_mask = np.zeros((height, width), dtype=np.uint8)
        mask_height, mask_width = measurement.mask.shape
        padded_mask[:mask_height, :mask_width] = measurement.mask
        padded_measurements[side] = replace(measurement, mask=padded_mask)
    return composite, padded_measurements


def _put_outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw readable diagnostic text over light or dark photographs."""

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (15, 15, 15),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def annotate_marker_offset_feet_original(
    image_bgr: np.ndarray,
    corners_by_id: Mapping[int, np.ndarray],
    calibrations: Mapping[str, BoardHomography],
    measurements: Mapping[str, StaticFootMeasurement],
    rectified: RectifiedBoard,
    *,
    per_marker_validation_rmse_mm: Mapping[int, float] | None = None,
    global_world_rmse_mm: float | None = None,
    lens_correction_text: str = "Lens undistortion: NOT APPLIED",
    measurement_valid: bool = True,
    invalid_reasons: Sequence[str] = (),
) -> np.ndarray:
    """Show masks, marker fit quality and measured toe-to-marker segments."""

    if set(calibrations) != set(measurements):
        raise PencilMeasurementError(
            "Marker-offset calibrations and measurements do not match"
        )
    annotated = image_bgr.copy()
    rectified_to_image = np.linalg.inv(rectified.image_to_rectified)

    # Project the selected foot masks back to the input orientation.  This makes
    # the actual HSV or YOLO segmentation visible instead of suggesting that
    # the coloured centre line alone was used as the foot boundary.
    for side, measurement in measurements.items():
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        projected_mask = cv2.warpPerspective(
            measurement.mask,
            rectified_to_image,
            (annotated.shape[1], annotated.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlay = annotated.copy()
        overlay[projected_mask > 0] = colour
        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
        contours, _ = cv2.findContours(
            projected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, colour, 3, cv2.LINE_AA)

    inlier_ids = {
        marker_id
        for calibration in calibrations.values()
        for marker_id in calibration.inlier_marker_ids
    }
    calibrated_ids = {
        marker_id
        for calibration in calibrations.values()
        for marker_id in calibration.per_marker_rmse_mm
    }
    for marker_id, corners in corners_by_id.items():
        points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        colour = (40, 210, 40) if marker_id in inlier_ids else (30, 30, 230)
        cv2.polylines(annotated, [points], True, colour, 5, cv2.LINE_AA)
        if marker_id in inlier_ids:
            state = "USED"
        elif marker_id in calibrated_ids:
            state = "REJECT"
        else:
            state = "CHECK"
        error = (
            None
            if per_marker_validation_rmse_mm is None
            else per_marker_validation_rmse_mm.get(marker_id)
        )
        label = f"ID {marker_id} {state}"
        if error is not None:
            label += f" {error:.1f}mm"
        flat_points = points.reshape(-1, 2)
        label_x = max(8, int(np.min(flat_points[:, 0])))
        label_y = min(
            annotated.shape[0] - 12,
            int(np.max(flat_points[:, 1])) + 28,
        )
        _put_outlined_text(
            annotated,
            label,
            (label_x, label_y),
            scale=0.58,
            colour=colour,
            thickness=2,
        )

    for side, measurement in measurements.items():
        if (
            measurement.reference_point_world_mm is None
            or measurement.visible_length_mm is None
            or measurement.fixed_offset_mm is None
        ):
            raise PencilMeasurementError("Incomplete marker-offset measurement")
        calibration = calibrations[side]
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        left_mm, _, right_mm, _ = FOOT_BAY_ROIS_MM[side]
        reference_y = float(measurement.reference_point_world_mm[1])
        edge_world = np.float32(
            [[left_mm, reference_y], [right_mm, reference_y]]
        )
        edge_image = cv2.perspectiveTransform(
            edge_world.reshape(1, -1, 2), calibration.world_to_image
        ).reshape(-1, 2)
        edge_image = np.rint(edge_image).astype(int)
        cv2.line(
            annotated,
            tuple(edge_image[0]),
            tuple(edge_image[1]),
            (255, 200, 0),
            6,
            cv2.LINE_AA,
        )
        segment_world = np.float32(
            [measurement.toe_world_mm, measurement.reference_point_world_mm]
        )
        segment_image = cv2.perspectiveTransform(
            segment_world.reshape(1, -1, 2), calibration.world_to_image
        ).reshape(-1, 2)
        segment_image = np.rint(segment_image).astype(int)
        cv2.line(
            annotated,
            tuple(segment_image[0]),
            tuple(segment_image[1]),
            colour,
            8,
            cv2.LINE_AA,
        )
        for point in segment_image:
            cv2.circle(annotated, tuple(point), 16, colour, -1, cv2.LINE_AA)
        label = segment_image.mean(axis=0).astype(int) + np.array([25, 0])
        display_side = side.replace("image_", "").upper()
        confidence_text = (
            ""
            if measurement.segmentation_confidence is None
            else f" | mask conf {measurement.segmentation_confidence:.3f}"
        )
        _put_outlined_text(
            annotated,
            (
                (
                    f"BAY {display_side}: "
                    if measurement_valid
                    else f"BAY {display_side}: INVALID / DIAGNOSTIC "
                )
                + f"{measurement.length_mm / 10.0:.1f} cm "
                f"({measurement.visible_length_mm / 10.0:.1f}+"
                f"{measurement.fixed_offset_mm / 10.0:.1f}){confidence_text}"
            ),
            tuple(label),
            scale=0.82,
            colour=colour,
            thickness=3,
        )

    segmentation_methods = {
        measurement.segmentation_method for measurement in measurements.values()
    }
    uses_yolo = any(method.startswith("yolo") for method in segmentation_methods)
    mask_label = "YOLO instance foot mask" if uses_yolo else "HSV foot mask"
    panel_lines = [
        (
            "MEASUREMENT: VALID"
            if measurement_valid
            else "MEASUREMENT: INVALID - DIAGNOSTIC LENGTHS ONLY"
        ),
        "GREEN: marker used for global plane | RED: check/rejected",
        f"COLOUR TINT + OUTLINE: selected {mask_label}",
        "CYAN: outer marker edge | line: visible toe-to-edge length",
        lens_correction_text,
    ]
    if not measurement_valid and invalid_reasons:
        panel_lines.append("Invalid: " + ", ".join(invalid_reasons))
    if global_world_rmse_mm is not None:
        quality = "WARNING" if global_world_rmse_mm > 3.0 else "OK"
        panel_lines.append(
            f"All-marker plane RMSE: {global_world_rmse_mm:.2f} mm [{quality}]"
        )
    panel_x = max(16, int(round(0.025 * annotated.shape[1])))
    panel_y = max(120, int(round(0.24 * annotated.shape[0])))
    panel_width = min(900, annotated.shape[1] - panel_x - 12)
    panel_height = 18 + 34 * len(panel_lines)
    panel = annotated.copy()
    cv2.rectangle(
        panel,
        (panel_x - 10, panel_y - 28),
        (panel_x + panel_width, panel_y - 28 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(panel, 0.56, annotated, 0.44, 0, annotated)
    for index, text in enumerate(panel_lines):
        line_colour = (
            (40, 80, 255)
            if "[WARNING]" in text or "INVALID" in text
            else (245, 245, 245)
        )
        _put_outlined_text(
            annotated,
            text,
            (panel_x, panel_y + index * 34),
            scale=0.58,
            colour=line_colour,
            thickness=2,
        )
    return annotated


def annotate_marker_offset_feet_rectified(
    rectified: RectifiedBoard,
    measurements: Mapping[str, StaticFootMeasurement],
    *,
    lens_correction_text: str = "Lens undistortion: NOT APPLIED",
    measurement_valid: bool = True,
    invalid_reasons: Sequence[str] = (),
) -> np.ndarray:
    """Show the single global metric plane and calibrated heel addition."""

    if not measurements:
        raise PencilMeasurementError("No marker-offset feet to annotate")
    annotated = rectified.image_bgr.copy()
    grid = annotated.copy()
    grid_step = max(1, int(round(50.0 * rectified.pixels_per_mm)))
    for x in range(0, annotated.shape[1], grid_step):
        cv2.line(grid, (x, 0), (x, annotated.shape[0] - 1), (90, 90, 90), 1)
    for y in range(0, annotated.shape[0], grid_step):
        cv2.line(grid, (0, y), (annotated.shape[1] - 1, y), (90, 90, 90), 1)
    cv2.addWeighted(grid, 0.18, annotated, 0.82, 0, annotated)

    for side, measurement in measurements.items():
        if (
            measurement.reference_point_rectified_px is None
            or measurement.visible_length_mm is None
            or measurement.fixed_offset_mm is None
        ):
            raise PencilMeasurementError("Incomplete marker-offset measurement")
        colour = _FOOT_ANNOTATION_COLOURS.get(side, (0, 0, 255))
        overlay = annotated.copy()
        overlay[measurement.mask > 0] = colour
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        toe = np.rint(measurement.toe_rectified_px).astype(int)
        reference = np.rint(measurement.reference_point_rectified_px).astype(int)
        left_mm, _, right_mm, _ = FOOT_BAY_ROIS_MM[side]
        edge_y = int(reference[1])
        edge_left = int(round(
            (left_mm - rectified.world_origin_mm[0]) * rectified.pixels_per_mm
        ))
        edge_right = int(round(
            (right_mm - rectified.world_origin_mm[0]) * rectified.pixels_per_mm
        ))
        cv2.line(
            annotated,
            (edge_left, edge_y),
            (edge_right, edge_y),
            (255, 200, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            annotated, tuple(toe), tuple(reference), colour, 6, cv2.LINE_AA
        )
        cv2.circle(annotated, tuple(toe), 12, colour, -1, cv2.LINE_AA)
        cv2.circle(annotated, tuple(reference), 12, colour, -1, cv2.LINE_AA)
        label = ((toe + reference) // 2) + np.array([15, 0])
        display_side = side.replace("image_", "").upper()
        confidence_text = (
            ""
            if measurement.segmentation_confidence is None
            else f" | conf {measurement.segmentation_confidence:.3f}"
        )
        cv2.putText(
            annotated,
            (
                (
                    f"BAY {display_side} "
                    if measurement_valid
                    else f"INVALID DIAGNOSTIC BAY {display_side} "
                )
                + f"{measurement.length_mm / 10.0:.1f} cm "
                f"= {measurement.visible_length_mm / 10.0:.1f} + "
                f"{measurement.fixed_offset_mm / 10.0:.1f}{confidence_text}"
            ),
            tuple(label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            3,
            cv2.LINE_AA,
        )

    methods = {measurement.segmentation_method for measurement in measurements.values()}
    segmentation_text = (
        "Mask: YOLO11 instance segmentation"
        if any(method.startswith("yolo") for method in methods)
        else "Mask: HSV saturation ensemble"
    )
    status_text = (
        "MEASUREMENT VALID"
        if measurement_valid
        else "MEASUREMENT INVALID | DIAGNOSTIC LENGTHS ONLY"
    )
    header_lines = [
        (status_text, (24, 38), 0.72),
        ("Global marker plane | grid: 50 mm", (24, 70), 0.62),
        ("Cyan: black marker reference edge", (24, 102), 0.62),
        (segmentation_text, (24, 134), 0.62),
        (lens_correction_text, (24, 166), 0.58),
    ]
    if not measurement_valid and invalid_reasons:
        header_lines.append(
            ("Invalid: " + ", ".join(invalid_reasons), (24, 198), 0.50)
        )
    for text, origin, scale in header_lines:
        text_colour = (
            (30, 30, 230)
            if "INVALID" in text or "Invalid:" in text
            else (25, 25, 25)
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_colour,
            2,
            cv2.LINE_AA,
        )
    return annotated
