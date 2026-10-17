from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


# Coordinates of the black 30 mm marker centres on the board.
# Treating these as marker top-left points would only translate every world
# coordinate by the same amount, so measured lengths would be unchanged.
WORLD_POINTS_MM: dict[int, tuple[float, float]] = {
    0: (25.0, 25.0),
    1: (181.0, 25.0),
    2: (345.0, 25.0),
    3: (25.0, 327.0),
    4: (181.0, 327.0),
    5: (345.0, 327.0),
}


class PencilMeasurementError(RuntimeError):
    """Raised when calibration or pencil segmentation is not trustworthy."""


@dataclass(frozen=True)
class BoardHomography:
    world_to_image: np.ndarray
    image_to_world: np.ndarray
    inlier_marker_ids: tuple[int, ...]
    rejected_marker_ids: tuple[int, ...]
    image_rmse_px: float
    world_rmse_mm: float
    world_max_error_mm: float
    per_marker_rmse_mm: Mapping[int, float]
    ransac_threshold_px: float


@dataclass(frozen=True)
class RectifiedBoard:
    image_bgr: np.ndarray
    image_to_rectified: np.ndarray
    world_origin_mm: tuple[float, float]
    pixels_per_mm: float
    measurement_roi_px: tuple[int, int, int, int]


@dataclass(frozen=True)
class PencilMeasurement:
    length_mm: float
    width_mm: float
    endpoints_world_mm: np.ndarray
    endpoints_rectified_px: np.ndarray
    contour_rectified_px: np.ndarray
    mask: np.ndarray
    saturation_threshold: int


def read_image(path: Path) -> np.ndarray:
    """Read non-ASCII paths on every OpenCV-supported platform."""

    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise PencilMeasurementError(f"Could not read image: {path}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise PencilMeasurementError(f"Could not decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise PencilMeasurementError(f"Could not encode output image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def marker_world_corners(
    center_mm: tuple[float, float], marker_size_mm: float
) -> np.ndarray:
    center = np.asarray(center_mm, dtype=np.float32)
    half = np.float32(marker_size_mm * 0.5)
    return center + np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float32,
    )


def _marker_side_pixels(corners_by_id: Mapping[int, np.ndarray]) -> float:
    sides: list[float] = []
    for corners in corners_by_id.values():
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        sides.extend(
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        )
    if not sides:
        raise PencilMeasurementError("No marker sides available")
    return float(np.median(sides))


def estimate_board_homography(
    corners_by_id: Mapping[int, np.ndarray],
    *,
    world_points_mm: Mapping[int, tuple[float, float]] = WORLD_POINTS_MM,
    marker_size_mm: float = 30.0,
    ransac_threshold_px: float | None = None,
    minimum_inlier_markers: int = 4,
    use_all_detected_markers: bool = False,
) -> BoardHomography:
    """Estimate one board-wide world->image homography from known markers.

    RANSAC is intentionally fitted in the world->image direction, so its
    threshold is expressed in observable image pixels. Inliers are approved as
    whole markers rather than mixing arbitrary corners from displaced markers.
    When ``use_all_detected_markers`` is true, one least-squares homography is
    fitted to every decoded marker corner without dropping the centre column.
    """

    detected_ids = sorted(set(corners_by_id).intersection(world_points_mm))
    if len(detected_ids) < minimum_inlier_markers:
        raise PencilMeasurementError(
            f"Need at least {minimum_inlier_markers} known markers; "
            f"detected {len(detected_ids)}"
        )
    if marker_size_mm <= 0:
        raise PencilMeasurementError("marker_size_mm must be positive")

    world_blocks: list[np.ndarray] = []
    image_blocks: list[np.ndarray] = []
    for marker_id in detected_ids:
        image_corners = np.asarray(corners_by_id[marker_id], dtype=np.float32).reshape(4, 2)
        world_corners = marker_world_corners(
            world_points_mm[marker_id], marker_size_mm
        )
        world_blocks.append(world_corners)
        image_blocks.append(image_corners)
    world_points = np.concatenate(world_blocks, axis=0)
    image_points = np.concatenate(image_blocks, axis=0)

    if ransac_threshold_px is None:
        # About 5 px for the supplied 4000x3000 photograph.
        known_corners = {marker_id: corners_by_id[marker_id] for marker_id in detected_ids}
        ransac_threshold_px = max(2.0, _marker_side_pixels(known_corners) * 0.03)
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise PencilMeasurementError("RANSAC threshold must be positive")

    if use_all_detected_markers:
        inlier_ids = tuple(detected_ids)
        selected_indices = np.arange(len(world_points), dtype=int)
    else:
        cv2.setRNGSeed(0)
        initial_homography, inlier_mask = cv2.findHomography(
            world_points,
            image_points,
            cv2.RANSAC,
            float(ransac_threshold_px),
            maxIters=10_000,
            confidence=0.999,
        )
        if initial_homography is None or inlier_mask is None:
            raise PencilMeasurementError("Could not estimate a board homography")

        flat_mask = inlier_mask.reshape(-1)
        inlier_ids = tuple(
            marker_id
            for index, marker_id in enumerate(detected_ids)
            if int(flat_mask[index * 4 : (index + 1) * 4].sum()) >= 3
        )
        if len(inlier_ids) < minimum_inlier_markers:
            raise PencilMeasurementError(
                f"Only {len(inlier_ids)} geometrically consistent markers; "
                f"need {minimum_inlier_markers}"
            )

        # Guard against accepting markers clustered in one small board region.
        all_centers = np.float32(
            [world_points_mm[value] for value in detected_ids]
        )
        inlier_centers = np.float32(
            [world_points_mm[value] for value in inlier_ids]
        )
        all_area = abs(cv2.contourArea(cv2.convexHull(all_centers)))
        inlier_area = abs(cv2.contourArea(cv2.convexHull(inlier_centers)))
        if all_area > 0 and inlier_area / all_area < 0.5:
            raise PencilMeasurementError(
                "Inlier markers do not cover enough of the board"
            )

        selected_indices = np.concatenate(
            [
                np.arange(index * 4, index * 4 + 4)
                for index, marker_id in enumerate(detected_ids)
                if marker_id in inlier_ids
            ]
        )
    world_to_image, _ = cv2.findHomography(
        world_points[selected_indices], image_points[selected_indices], 0
    )
    if world_to_image is None or abs(np.linalg.det(world_to_image)) < 1e-12:
        raise PencilMeasurementError("Refined board homography is singular")
    image_to_world = np.linalg.inv(world_to_image)
    world_to_image /= world_to_image[2, 2]
    image_to_world /= image_to_world[2, 2]

    predicted_image = cv2.perspectiveTransform(
        world_points.reshape(1, -1, 2), world_to_image
    ).reshape(-1, 2)
    image_errors = np.linalg.norm(predicted_image - image_points, axis=1)
    predicted_world = cv2.perspectiveTransform(
        image_points.reshape(1, -1, 2), image_to_world
    ).reshape(-1, 2)
    world_errors = np.linalg.norm(predicted_world - world_points, axis=1)

    per_marker_rmse = {
        marker_id: float(
            np.sqrt(np.mean(world_errors[index * 4 : (index + 1) * 4] ** 2))
        )
        for index, marker_id in enumerate(detected_ids)
    }
    inlier_corner_indices = selected_indices.astype(int)
    rejected_ids = tuple(value for value in detected_ids if value not in inlier_ids)
    return BoardHomography(
        world_to_image=world_to_image,
        image_to_world=image_to_world,
        inlier_marker_ids=inlier_ids,
        rejected_marker_ids=rejected_ids,
        image_rmse_px=float(
            np.sqrt(np.mean(image_errors[inlier_corner_indices] ** 2))
        ),
        world_rmse_mm=float(
            np.sqrt(np.mean(world_errors[inlier_corner_indices] ** 2))
        ),
        world_max_error_mm=float(np.max(world_errors[inlier_corner_indices])),
        per_marker_rmse_mm=per_marker_rmse,
        ransac_threshold_px=float(ransac_threshold_px),
    )


def rectify_board(
    image_bgr: np.ndarray,
    calibration: BoardHomography,
    *,
    world_points_mm: Mapping[int, tuple[float, float]] = WORLD_POINTS_MM,
    marker_size_mm: float = 30.0,
    pixels_per_mm: float = 5.0,
    object_padding_mm: float = 5.0,
) -> RectifiedBoard:
    if pixels_per_mm <= 0:
        raise PencilMeasurementError("pixels_per_mm must be positive")
    centers = np.float32(list(world_points_mm.values()))
    unique_x = np.unique(centers[:, 0])
    unique_y = np.unique(centers[:, 1])
    if len(unique_x) < 2 or len(unique_y) < 2:
        raise PencilMeasurementError("Marker coordinates must form rows and columns")

    # The supplied coordinates use a board origin outside the first marker.
    world_min_x = 0.0
    world_min_y = 0.0
    world_max_x = float(unique_x.max() + unique_x.min())
    world_max_y = float(unique_y.max() + unique_y.min())
    width_px = int(np.ceil((world_max_x - world_min_x) * pixels_per_mm))
    height_px = int(np.ceil((world_max_y - world_min_y) * pixels_per_mm))
    world_to_rectified = np.array(
        [
            [pixels_per_mm, 0.0, -world_min_x * pixels_per_mm],
            [0.0, pixels_per_mm, -world_min_y * pixels_per_mm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_to_rectified = world_to_rectified @ calibration.image_to_world
    rectified = cv2.warpPerspective(
        image_bgr,
        image_to_rectified,
        (width_px, height_px),
        flags=cv2.INTER_LINEAR,
    )

    half_marker = marker_size_mm * 0.5
    roi_left_mm = float(unique_x.min() + half_marker + object_padding_mm)
    roi_right_mm = float(unique_x.max() - half_marker - object_padding_mm)
    roi_top_mm = float(unique_y.min() + half_marker + object_padding_mm)
    roi_bottom_mm = float(unique_y.max() - half_marker - object_padding_mm)
    roi_px = (
        int(round((roi_left_mm - world_min_x) * pixels_per_mm)),
        int(round((roi_top_mm - world_min_y) * pixels_per_mm)),
        int(round((roi_right_mm - world_min_x) * pixels_per_mm)),
        int(round((roi_bottom_mm - world_min_y) * pixels_per_mm)),
    )
    return RectifiedBoard(
        image_bgr=rectified,
        image_to_rectified=image_to_rectified,
        world_origin_mm=(world_min_x, world_min_y),
        pixels_per_mm=float(pixels_per_mm),
        measurement_roi_px=roi_px,
    )


def _odd_kernel_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def measure_coloured_pencil(
    rectified: RectifiedBoard,
    *,
    saturation_threshold: int = 45,
    minimum_value: int = 40,
    minimum_length_mm: float = 80.0,
    maximum_length_mm: float = 250.0,
    minimum_width_mm: float = 2.0,
    maximum_width_mm: float = 20.0,
) -> PencilMeasurement:
    """Segment the orange/red pencil and measure its PCA-axis support length."""

    if not 0 <= saturation_threshold <= 255 or not 0 <= minimum_value <= 255:
        raise PencilMeasurementError("HSV thresholds must be between 0 and 255")
    image = rectified.image_bgr
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    left, top, right, bottom = rectified.measurement_roi_px
    cv2.rectangle(roi_mask, (left, top), (right, bottom), 255, -1)
    mask = np.where(
        (hsv[:, :, 1] >= saturation_threshold)
        & (hsv[:, :, 2] >= minimum_value)
        & (roi_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    opening_size = _odd_kernel_size(rectified.pixels_per_mm * 1.0)
    closing_size = _odd_kernel_size(rectified.pixels_per_mm * 1.4)
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[
        tuple[float, float, float, int, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for label in range(1, component_count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < rectified.pixels_per_mm**2 * 20.0:
            continue
        y_coordinates, x_coordinates = np.where(labels == label)
        points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)
        major_projection = (
            points[:, 0] * major_axis[0] + points[:, 1] * major_axis[1]
        )
        minor_projection = (
            points[:, 0] * minor_axis[0] + points[:, 1] * minor_axis[1]
        )
        length_mm = float(np.ptp(major_projection) / rectified.pixels_per_mm)
        width_mm = float(np.ptp(minor_projection) / rectified.pixels_per_mm)
        aspect_ratio = length_mm / max(width_mm, 1e-6)
        if not (minimum_length_mm <= length_mm <= maximum_length_mm):
            continue
        if not (minimum_width_mm <= width_mm <= maximum_width_mm):
            continue
        if aspect_ratio < 8.0:
            continue
        if (
            x_coordinates.min() <= left
            or x_coordinates.max() >= right
            or y_coordinates.min() <= top
            or y_coordinates.max() >= bottom
        ):
            continue
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        score = length_mm * aspect_ratio
        candidates.append(
            (
                score,
                length_mm,
                width_mm,
                label,
                major_axis,
                minor_axis,
                contour,
            )
        )

    if not candidates:
        raise PencilMeasurementError(
            "Could not find one long, narrow coloured object inside the marker area"
        )
    _, length_mm, width_mm, selected_label, major_axis, minor_axis, contour = max(
        candidates, key=lambda item: item[0]
    )
    y_coordinates, x_coordinates = np.where(labels == selected_label)
    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
    major_projection = points[:, 0] * major_axis[0] + points[:, 1] * major_axis[1]
    minor_projection = points[:, 0] * minor_axis[0] + points[:, 1] * minor_axis[1]
    minor_center = float(np.median(minor_projection))
    endpoints_px = np.array(
        [
            major_axis * float(major_projection.min()) + minor_axis * minor_center,
            major_axis * float(major_projection.max()) + minor_axis * minor_center,
        ],
        dtype=np.float32,
    )
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    endpoints_world = endpoints_px / rectified.pixels_per_mm + origin
    selected_mask = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    return PencilMeasurement(
        length_mm=length_mm,
        width_mm=width_mm,
        endpoints_world_mm=endpoints_world,
        endpoints_rectified_px=endpoints_px,
        contour_rectified_px=contour,
        mask=selected_mask,
        saturation_threshold=int(saturation_threshold),
    )


def annotate_original_image(
    image_bgr: np.ndarray,
    corners_by_id: Mapping[int, np.ndarray],
    calibration: BoardHomography,
    measurement: PencilMeasurement,
    rectified: RectifiedBoard,
) -> np.ndarray:
    annotated = image_bgr.copy()
    inlier_set = set(calibration.inlier_marker_ids)
    for marker_id, corners in corners_by_id.items():
        points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        colour = (40, 210, 40) if marker_id in inlier_set else (30, 30, 230)
        cv2.polylines(annotated, [points], True, colour, 5, cv2.LINE_AA)
        anchor = tuple(points[0, 0])
        cv2.putText(
            annotated,
            f"ID {marker_id}",
            (anchor[0], anchor[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colour,
            3,
            cv2.LINE_AA,
        )

    contour_rectified = measurement.contour_rectified_px.reshape(-1, 2).astype(np.float32)
    contour_world = (
        contour_rectified / rectified.pixels_per_mm
        + np.asarray(rectified.world_origin_mm, dtype=np.float32)
    )
    contour_image = cv2.perspectiveTransform(
        contour_world.reshape(1, -1, 2), calibration.world_to_image
    ).reshape(-1, 1, 2)
    cv2.polylines(
        annotated,
        [np.rint(contour_image).astype(np.int32)],
        True,
        (0, 220, 255),
        5,
        cv2.LINE_AA,
    )
    endpoints_image = cv2.perspectiveTransform(
        measurement.endpoints_world_mm.reshape(1, -1, 2),
        calibration.world_to_image,
    ).reshape(-1, 2)
    endpoints_int = np.rint(endpoints_image).astype(int)
    cv2.line(
        annotated,
        tuple(endpoints_int[0]),
        tuple(endpoints_int[1]),
        (0, 0, 255),
        7,
        cv2.LINE_AA,
    )
    for endpoint in endpoints_int:
        cv2.circle(annotated, tuple(endpoint), 16, (0, 0, 255), -1, cv2.LINE_AA)
    label_position = tuple(endpoints_int.mean(axis=0).astype(int) + np.array([30, -30]))
    cv2.putText(
        annotated,
        f"Pencil: {measurement.length_mm:.1f} mm",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return annotated


def annotate_rectified_image(
    rectified: RectifiedBoard, measurement: PencilMeasurement
) -> np.ndarray:
    annotated = rectified.image_bgr.copy()
    cv2.rectangle(
        annotated,
        (rectified.measurement_roi_px[0], rectified.measurement_roi_px[1]),
        (rectified.measurement_roi_px[2], rectified.measurement_roi_px[3]),
        (255, 170, 0),
        3,
    )
    cv2.drawContours(
        annotated,
        [measurement.contour_rectified_px.astype(np.int32)],
        -1,
        (0, 220, 255),
        4,
        cv2.LINE_AA,
    )
    endpoints = np.rint(measurement.endpoints_rectified_px).astype(int)
    cv2.line(
        annotated,
        tuple(endpoints[0]),
        tuple(endpoints[1]),
        (0, 0, 255),
        5,
        cv2.LINE_AA,
    )
    for endpoint in endpoints:
        cv2.circle(annotated, tuple(endpoint), 10, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"{measurement.length_mm:.1f} mm",
        tuple(endpoints.mean(axis=0).astype(int) + np.array([20, -20])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return annotated
