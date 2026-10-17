from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import cv2
import numpy as np


# Source contract:
# https://github.com/1st-Guardians/ChArUco/tree/
# 36b4db11682b160941ff66b2761907c04f0ad4f1/two_marker_foot_length
REQUIRED_MARKER_IDS = (1, 4)
IMAGE_SIDES = ("image_left", "image_right")


class TwoMarkerGeometryError(RuntimeError):
    """Raised when IDs 1/4 cannot support a trustworthy metric result."""


@dataclass(frozen=True)
class RigGeometry:
    """Physical dimensions of the ID 1/4 centre-column measurement rig."""

    marker_size_mm: float = 20.0
    marker_edge_gap_mm: float = 175.0
    heel_offset_mm: float = 114.0
    board_width_mm: float = 365.0
    marker_center_x_mm: float = 181.0
    top_margin_mm: float = 15.0
    bottom_margin_mm: float = 10.0
    side_margin_mm: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.marker_size_mm,
            self.marker_edge_gap_mm,
            self.heel_offset_mm,
            self.board_width_mm,
            self.marker_center_x_mm,
            self.top_margin_mm,
            self.bottom_margin_mm,
            self.side_margin_mm,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise TwoMarkerGeometryError("All two-marker rig dimensions must be finite")
        if self.marker_size_mm <= 0 or self.marker_edge_gap_mm <= 0:
            raise TwoMarkerGeometryError("Marker size and edge gap must be positive")
        if self.heel_offset_mm <= 0 or self.board_width_mm <= 0:
            raise TwoMarkerGeometryError("Board and heel dimensions must be positive")
        if self.top_margin_mm < 0 or self.bottom_margin_mm < 0:
            raise TwoMarkerGeometryError("Board margins cannot be negative")
        half = self.marker_size_mm * 0.5
        if not half < self.marker_center_x_mm < self.board_width_mm - half:
            raise TwoMarkerGeometryError("Marker centre must lie inside the board")
        if not 0 <= self.side_margin_mm < min(
            self.marker_center_x_mm,
            self.board_width_mm - self.marker_center_x_mm,
        ):
            raise TwoMarkerGeometryError("Side margin leaves no foot bay")

    @property
    def marker_center_distance_mm(self) -> float:
        # The declared 175 mm is black-edge to black-edge, not centre spacing.
        return float(self.marker_edge_gap_mm + self.marker_size_mm)

    @property
    def id1_center_y_mm(self) -> float:
        return float(self.top_margin_mm + self.marker_size_mm * 0.5)

    @property
    def id4_center_y_mm(self) -> float:
        return float(self.id1_center_y_mm + self.marker_center_distance_mm)

    @property
    def reference_edge_y_mm(self) -> float:
        """Heel-facing black outer edge of ID 4."""

        return float(self.id4_center_y_mm + self.marker_size_mm * 0.5)

    @property
    def heel_line_y_mm(self) -> float:
        return float(self.reference_edge_y_mm + self.heel_offset_mm)

    @property
    def board_height_mm(self) -> float:
        return float(self.heel_line_y_mm + self.bottom_margin_mm)

    @property
    def world_points_mm(self) -> dict[int, tuple[float, float]]:
        return {
            1: (float(self.marker_center_x_mm), self.id1_center_y_mm),
            4: (float(self.marker_center_x_mm), self.id4_center_y_mm),
        }

    @property
    def bay_rois_mm(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "image_left": (
                float(self.side_margin_mm),
                0.0,
                float(self.marker_center_x_mm),
                self.reference_edge_y_mm,
            ),
            "image_right": (
                float(self.marker_center_x_mm),
                0.0,
                float(self.board_width_mm - self.side_margin_mm),
                self.reference_edge_y_mm,
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_ids": list(REQUIRED_MARKER_IDS),
            "marker_size_mm": float(self.marker_size_mm),
            "marker_gap_semantics": "black_edge_to_edge",
            "marker_edge_gap_mm": float(self.marker_edge_gap_mm),
            "marker_center_distance_mm": self.marker_center_distance_mm,
            "id1_center_mm": list(self.world_points_mm[1]),
            "id4_center_mm": list(self.world_points_mm[4]),
            "id4_heel_facing_outer_edge_y_mm": self.reference_edge_y_mm,
            "heel_offset_from_id4_outer_edge_mm": float(self.heel_offset_mm),
            "heel_line_y_mm": self.heel_line_y_mm,
            "board_width_mm": float(self.board_width_mm),
            "board_height_mm": self.board_height_mm,
            "marker_center_x_mm": float(self.marker_center_x_mm),
            "bay_rois_mm": {
                key: list(value) for key, value in self.bay_rois_mm.items()
            },
            "length_formula": (
                "foot_length_mm = (id4_outer_edge_y_mm - toe_y_mm) "
                "+ heel_offset_mm"
            ),
        }


@dataclass(frozen=True)
class HomographyFit:
    world_to_image: np.ndarray = field(repr=False)
    image_to_world: np.ndarray = field(repr=False)
    image_rmse_px: float
    image_max_error_px: float
    world_fit_rmse_mm: float
    world_fit_max_error_mm: float
    leave_one_corner_out_p95_mm: float
    per_marker_world_fit_rmse_mm: Mapping[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "all_8_corners_least_squares",
            "image_fit_rmse_px": float(self.image_rmse_px),
            "image_fit_max_error_px": float(self.image_max_error_px),
            "world_fit_rmse_mm": float(self.world_fit_rmse_mm),
            "world_fit_max_error_mm": float(self.world_fit_max_error_mm),
            "leave_one_corner_out_p95_mm": float(
                self.leave_one_corner_out_p95_mm
            ),
            "per_marker_world_fit_rmse_mm": {
                str(key): float(value)
                for key, value in self.per_marker_world_fit_rmse_mm.items()
            },
            "world_to_image": self.world_to_image.astype(float).tolist(),
            "image_to_world": self.image_to_world.astype(float).tolist(),
            "interpretation": (
                "fit residual on the same marker corners; not final foot-length error"
            ),
        }


@dataclass(frozen=True)
class FootLengthMeasurement:
    side: str
    foot_length_mm: float
    visible_to_reference_mm: float
    fixed_heel_offset_mm: float
    toe_world_mm: tuple[float, float]
    reference_world_mm: tuple[float, float]
    heel_world_mm: tuple[float, float]
    ball_width_mm: float
    ball_line_world_mm: tuple[tuple[float, float], tuple[float, float]]
    ball_region_fraction_from_toe: tuple[float, float]
    visible_mask: np.ndarray = field(repr=False, compare=False)
    mask_area_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "side_semantics": "position_in_oriented_image_not_anatomical_side",
            "foot_length_mm": round(float(self.foot_length_mm), 3),
            "foot_length_cm": round(float(self.foot_length_mm) / 10.0, 3),
            "visible_to_id4_outer_edge_mm": round(
                float(self.visible_to_reference_mm), 3
            ),
            "fixed_heel_offset_mm": round(float(self.fixed_heel_offset_mm), 3),
            "formula_check_mm": round(
                float(self.visible_to_reference_mm + self.fixed_heel_offset_mm), 3
            ),
            "toe_world_mm": [round(value, 3) for value in self.toe_world_mm],
            "reference_world_mm": [
                round(value, 3) for value in self.reference_world_mm
            ],
            "virtual_heel_world_mm": [
                round(value, 3) for value in self.heel_world_mm
            ],
            "ball_width_mm": round(float(self.ball_width_mm), 3),
            "ball_width_cm": round(float(self.ball_width_mm) / 10.0, 3),
            "ball_line_world_mm": [
                [round(value, 3) for value in point]
                for point in self.ball_line_world_mm
            ],
            "ball_axis": "perpendicular_to_id1_id4_axis",
            "ball_region_fraction_from_toe": list(
                self.ball_region_fraction_from_toe
            ),
            "ball_width_method": (
                "90th_percentile_horizontal_mask_chord_in_forefoot_band"
            ),
            "mask_area_mm2": round(float(self.mask_area_mm2), 3),
            "heel_is_virtual": True,
        }


def marker_world_corners(
    center_mm: tuple[float, float], marker_size_mm: float
) -> np.ndarray:
    center = np.asarray(center_mm, dtype=np.float64)
    half = float(marker_size_mm) * 0.5
    return center + np.asarray(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float64,
    )


def correspondence_arrays(
    corners_by_id: Mapping[int, np.ndarray], geometry: RigGeometry
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        marker_id
        for marker_id in REQUIRED_MARKER_IDS
        if marker_id not in corners_by_id
    ]
    if missing:
        raise TwoMarkerGeometryError(
            "Required ArUco marker(s) missing: " + ", ".join(map(str, missing))
        )
    world = np.concatenate(
        [
            marker_world_corners(
                geometry.world_points_mm[marker_id], geometry.marker_size_mm
            )
            for marker_id in REQUIRED_MARKER_IDS
        ],
        axis=0,
    )
    image = np.concatenate(
        [
            np.asarray(corners_by_id[marker_id], dtype=np.float64).reshape(4, 2)
            for marker_id in REQUIRED_MARKER_IDS
        ],
        axis=0,
    )
    if image.shape != (8, 2) or not np.isfinite(image).all():
        raise TwoMarkerGeometryError("Marker corners must be finite 4x2 arrays")
    return world, image


def marker_image_geometry(
    corners_by_id: Mapping[int, np.ndarray],
    geometry: RigGeometry,
    *,
    minimum_marker_side_px: float = 30.0,
) -> dict[str, float]:
    if not math.isfinite(minimum_marker_side_px) or minimum_marker_side_px <= 0:
        raise TwoMarkerGeometryError("Minimum marker side must be positive")
    _, image = correspondence_arrays(corners_by_id, geometry)
    side_lengths: list[float] = []
    areas: list[float] = []
    centres: list[np.ndarray] = []
    for index, marker_id in enumerate(REQUIRED_MARKER_IDS):
        corners = image[index * 4 : index * 4 + 4].astype(np.float32)
        if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
            raise TwoMarkerGeometryError(f"Marker ID {marker_id} is not convex")
        area = abs(float(cv2.contourArea(corners)))
        if area <= 1.0:
            raise TwoMarkerGeometryError(f"Marker ID {marker_id} has zero area")
        areas.append(area)
        centres.append(corners.mean(axis=0).astype(np.float64))
        side_lengths.extend(
            float(np.linalg.norm(corners[(edge + 1) % 4] - corners[edge]))
            for edge in range(4)
        )
    minimum_side = float(np.min(side_lengths))
    median_side = float(np.median(side_lengths))
    if minimum_side < minimum_marker_side_px:
        raise TwoMarkerGeometryError(
            f"Marker is too small ({minimum_side:.1f}px < "
            f"{minimum_marker_side_px:.1f}px)"
        )
    if centres[1][1] <= centres[0][1]:
        raise TwoMarkerGeometryError(
            "Canonical order is invalid: ID 1 must be above ID 4"
        )
    centre_distance_px = float(np.linalg.norm(centres[1] - centres[0]))
    observed_ratio = centre_distance_px / median_side
    expected_ratio = geometry.marker_center_distance_mm / geometry.marker_size_mm
    ratio_scale = observed_ratio / expected_ratio
    if not 0.50 <= ratio_scale <= 2.00:
        raise TwoMarkerGeometryError(
            "Marker centre/side ratio is inconsistent with the declared rig"
        )
    area_ratio = max(areas) / min(areas)
    if area_ratio > 4.0:
        raise TwoMarkerGeometryError(
            "Marker scale changes too strongly across the two-marker span"
        )
    return {
        "minimum_marker_side_px": minimum_side,
        "median_marker_side_px": median_side,
        "marker_center_distance_px": centre_distance_px,
        "observed_center_to_side_ratio": observed_ratio,
        "expected_center_to_side_ratio": expected_ratio,
        "ratio_scale": ratio_scale,
        "marker_area_ratio": float(area_ratio),
    }


def _perspective(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(1, -1, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)


def fit_two_marker_homography(
    corners_by_id: Mapping[int, np.ndarray], geometry: RigGeometry
) -> HomographyFit:
    world, image = correspondence_arrays(corners_by_id, geometry)
    homography, _ = cv2.findHomography(world, image, 0)
    if homography is None or not np.isfinite(homography).all():
        raise TwoMarkerGeometryError("Could not fit the two-marker Homography")
    scale = float(homography[2, 2])
    if abs(scale) < 1e-12:
        raise TwoMarkerGeometryError("Homography has an invalid scale")
    homography = np.asarray(homography, dtype=np.float64) / scale
    determinant = float(np.linalg.det(homography))
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise TwoMarkerGeometryError("Two-marker Homography is singular")
    inverse = np.linalg.inv(homography)
    inverse /= inverse[2, 2]

    image_errors = np.linalg.norm(_perspective(world, homography) - image, axis=1)
    world_errors = np.linalg.norm(_perspective(image, inverse) - world, axis=1)
    held_out_errors: list[float] = []
    for held_out in range(len(world)):
        keep = np.arange(len(world)) != held_out
        candidate, _ = cv2.findHomography(world[keep], image[keep], 0)
        if candidate is None or abs(float(np.linalg.det(candidate))) < 1e-12:
            held_out_errors.append(float("inf"))
            continue
        candidate_inverse = np.linalg.inv(candidate)
        predicted = _perspective(
            image[held_out : held_out + 1], candidate_inverse
        )[0]
        held_out_errors.append(float(np.linalg.norm(predicted - world[held_out])))
    if not np.isfinite(held_out_errors).all():
        raise TwoMarkerGeometryError(
            "Homography is unstable when one corner is withheld"
        )

    fit = HomographyFit(
        world_to_image=homography,
        image_to_world=inverse,
        image_rmse_px=float(np.sqrt(np.mean(image_errors**2))),
        image_max_error_px=float(np.max(image_errors)),
        world_fit_rmse_mm=float(np.sqrt(np.mean(world_errors**2))),
        world_fit_max_error_mm=float(np.max(world_errors)),
        leave_one_corner_out_p95_mm=float(np.percentile(held_out_errors, 95)),
        per_marker_world_fit_rmse_mm={
            marker_id: float(
                np.sqrt(np.mean(world_errors[index * 4 : index * 4 + 4] ** 2))
            )
            for index, marker_id in enumerate(REQUIRED_MARKER_IDS)
        },
    )
    validate_homography_coverage(fit, geometry)
    return fit


def validate_homography_coverage(
    fit: HomographyFit, geometry: RigGeometry
) -> None:
    bounds = np.asarray(
        [
            [0.0, 0.0],
            [geometry.board_width_mm, 0.0],
            [geometry.board_width_mm, geometry.board_height_mm],
            [0.0, geometry.board_height_mm],
            [geometry.marker_center_x_mm, geometry.reference_edge_y_mm],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([bounds, np.ones(len(bounds))])
    denominators = (homogeneous @ fit.world_to_image[2, :].reshape(3, 1)).reshape(
        -1
    )
    if not np.isfinite(denominators).all():
        raise TwoMarkerGeometryError("Homography is non-finite over the board")
    if np.any(np.abs(denominators) < 1e-3):
        raise TwoMarkerGeometryError("Homography horizon intersects the board")
    signs = np.sign(denominators)
    if np.any(signs != signs[0]):
        raise TwoMarkerGeometryError("Homography folds across the measurement area")


def rectification_transform(
    fit: HomographyFit,
    geometry: RigGeometry,
    pixels_per_mm: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise TwoMarkerGeometryError("pixels_per_mm must be positive")
    world_to_rectified = np.asarray(
        [
            [pixels_per_mm, 0.0, 0.0],
            [0.0, pixels_per_mm, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_to_rectified = world_to_rectified @ fit.image_to_world
    width = int(math.ceil(geometry.board_width_mm * pixels_per_mm))
    height = int(math.ceil(geometry.board_height_mm * pixels_per_mm))
    return image_to_rectified, (width, height)


def roi_pixels(
    geometry: RigGeometry,
    side: str,
    pixels_per_mm: float,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    if side not in geometry.bay_rois_mm:
        raise TwoMarkerGeometryError(f"Unknown image side: {side}")
    left, top, right, bottom = geometry.bay_rois_mm[side]
    height, width = shape
    return (
        max(0, int(math.floor(left * pixels_per_mm))),
        max(0, int(math.floor(top * pixels_per_mm))),
        min(width, int(math.ceil(right * pixels_per_mm))),
        min(height, int(math.ceil(bottom * pixels_per_mm))),
    )


def measure_foot_mask(
    mask: np.ndarray,
    geometry: RigGeometry,
    *,
    side: str,
    pixels_per_mm: float,
    minimum_area_mm2: float = 2_000.0,
    maximum_reference_gap_mm: float = 7.0,
    minimum_foot_length_mm: float = 180.0,
    maximum_foot_length_mm: float = 320.0,
    ball_region_start_fraction: float = 0.32,
    ball_region_end_fraction: float = 0.36,
    minimum_ball_width_mm: float = 50.0,
    maximum_ball_width_mm: float = 140.0,
) -> FootLengthMeasurement:
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise TwoMarkerGeometryError("Foot mask must be a non-empty 2D array")
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise TwoMarkerGeometryError("pixels_per_mm must be positive")
    if minimum_area_mm2 <= 0 or maximum_reference_gap_mm < 0:
        raise TwoMarkerGeometryError("Mask quality limits are invalid")
    if not 0 < minimum_foot_length_mm < maximum_foot_length_mm:
        raise TwoMarkerGeometryError("Foot-length limits are invalid")
    if not 0 < ball_region_start_fraction < ball_region_end_fraction < 1:
        raise TwoMarkerGeometryError("Ball-width forefoot fractions are invalid")
    if not 0 < minimum_ball_width_mm < maximum_ball_width_mm:
        raise TwoMarkerGeometryError("Ball-width limits are invalid")

    binary = np.where(array > 0, 255, 0).astype(np.uint8)
    left, top, right, bottom = roi_pixels(
        geometry, side, pixels_per_mm, binary.shape
    )
    clipped = np.zeros_like(binary)
    clipped[top:bottom, left:right] = binary[top:bottom, left:right]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (clipped > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise TwoMarkerGeometryError(f"No foot mask found in {side}")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    area_mm2 = float(np.count_nonzero(component) / (pixels_per_mm**2))
    if area_mm2 < minimum_area_mm2:
        raise TwoMarkerGeometryError(
            f"Foot mask in {side} is too small ({area_mm2:.0f} mm^2)"
        )

    y_coordinates, x_coordinates = np.where(component > 0)
    reference_y_px = geometry.reference_edge_y_mm * pixels_per_mm
    if float(y_coordinates.max()) < (
        reference_y_px - maximum_reference_gap_mm * pixels_per_mm
    ):
        raise TwoMarkerGeometryError(
            f"Foot mask in {side} does not reach the ID 4 reference edge"
        )
    toe_y_px = int(y_coordinates.min())
    toe_band_height_px = max(2, int(round(pixels_per_mm)))
    toe_band_x = x_coordinates[y_coordinates <= toe_y_px + toe_band_height_px]
    if len(toe_band_x) < max(3, int(round(pixels_per_mm))):
        raise TwoMarkerGeometryError(f"Toe boundary in {side} is too noisy")
    toe_x_px = float(np.median(toe_band_x))
    toe_world = (toe_x_px / pixels_per_mm, float(toe_y_px) / pixels_per_mm)
    visible_mm = float(geometry.reference_edge_y_mm - toe_world[1])
    total_mm = float(visible_mm + geometry.heel_offset_mm)
    if not minimum_foot_length_mm <= total_mm <= maximum_foot_length_mm:
        raise TwoMarkerGeometryError(
            f"Implausible foot length in {side}: {total_mm:.1f} mm"
        )

    ball_top_px = int(
        round(toe_y_px + ball_region_start_fraction * total_mm * pixels_per_mm)
    )
    ball_bottom_px = int(
        round(toe_y_px + ball_region_end_fraction * total_mm * pixels_per_mm)
    )
    ball_rows: list[tuple[int, int, int, float]] = []
    for row in range(
        max(0, ball_top_px), min(component.shape[0], ball_bottom_px + 1)
    ):
        row_x = np.where(component[row] > 0)[0]
        if row_x.size < 2:
            continue
        row_left = int(row_x.min())
        row_right = int(row_x.max())
        width_mm = float(row_right - row_left) / pixels_per_mm
        ball_rows.append((row, row_left, row_right, width_mm))
    if len(ball_rows) < max(3, int(round(pixels_per_mm))):
        raise TwoMarkerGeometryError(
            f"Could not find a stable horizontal ball-width band in {side}"
        )
    stable_width = float(np.percentile([item[3] for item in ball_rows], 90))
    selected_ball_row = min(
        ball_rows,
        key=lambda item: (
            abs(item[3] - stable_width),
            abs(item[0] - (ball_top_px + ball_bottom_px) / 2),
        ),
    )
    ball_width_mm = float(selected_ball_row[3])
    if not minimum_ball_width_mm <= ball_width_mm <= maximum_ball_width_mm:
        raise TwoMarkerGeometryError(
            f"Implausible ball width in {side}: {ball_width_mm:.1f} mm"
        )
    ball_y_mm = float(selected_ball_row[0]) / pixels_per_mm
    ball_line_world = (
        (float(selected_ball_row[1]) / pixels_per_mm, ball_y_mm),
        (float(selected_ball_row[2]) / pixels_per_mm, ball_y_mm),
    )
    visible = component.copy()
    visible[int(math.ceil(reference_y_px)) :] = 0
    return FootLengthMeasurement(
        side=side,
        foot_length_mm=total_mm,
        visible_to_reference_mm=visible_mm,
        fixed_heel_offset_mm=float(geometry.heel_offset_mm),
        toe_world_mm=toe_world,
        reference_world_mm=(toe_world[0], geometry.reference_edge_y_mm),
        heel_world_mm=(toe_world[0], geometry.heel_line_y_mm),
        ball_width_mm=ball_width_mm,
        ball_line_world_mm=ball_line_world,
        ball_region_fraction_from_toe=(
            float(ball_region_start_fraction),
            float(ball_region_end_fraction),
        ),
        visible_mask=visible,
        mask_area_mm2=area_mm2,
    )
