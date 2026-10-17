from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .pencil import PencilMeasurementError, RectifiedBoard
from .static_foot import StaticFootMeasurement


_SIDE_COLOURS: dict[str, tuple[int, int, int]] = {
    "image_left": (50, 70, 245),
    "image_right": (245, 120, 35),
}


@dataclass(frozen=True)
class FootBallWidthMeasurement:
    """A stable forefoot cross-section measured in the metric board plane."""

    width_mm: float
    endpoints_rectified_px: np.ndarray
    endpoints_world_mm: np.ndarray
    center_rectified_px: np.ndarray
    center_world_mm: np.ndarray
    distance_from_toe_mm: float
    fraction_from_toe: float
    fraction_from_heel: float
    search_start_fraction_from_toe: float
    search_end_fraction_from_toe: float
    target_fraction_from_toe: float
    axis_angle_degrees: float
    candidate_count: int
    candidate_width_spread_mm: float
    method: str = "paper_parallel_toe_cleared_mtp_zone_width"


def measure_foot_ball_width(
    rectified: RectifiedBoard,
    foot: StaticFootMeasurement,
    *,
    search_start_fraction_from_toe: float = 0.32,
    search_end_fraction_from_toe: float = 0.38,
    target_fraction_from_toe: float = 0.35,
    slice_band_mm: float = 5.0,
    slice_step_mm: float = 0.5,
) -> FootBallWidthMeasurement:
    """Estimate ball width without allowing toes or the midfoot to win.

    This is a mask-based forefoot breadth estimate, not a ball-girth estimate.
    Every candidate line is parallel to the rectified paper's horizontal axis,
    regardless of how the foot itself is rotated.  The default 32--38% zone
    starts behind the digit silhouette and approximates the first-to-fifth
    metatarsophalangeal (MTP) joint region.  It does not claim to localise the
    joints themselves because the segmentation model supplies only a mask.
    """

    start = float(search_start_fraction_from_toe)
    end = float(search_end_fraction_from_toe)
    target_fraction = float(target_fraction_from_toe)
    if not 0.05 <= start < end <= 0.55:
        raise PencilMeasurementError(
            "Ball-width search fractions must satisfy 0.05 <= start < end <= 0.55"
        )
    if not start <= target_fraction <= end:
        raise PencilMeasurementError(
            "Ball-width target fraction must be inside the configured search zone"
        )
    if not np.isfinite(slice_band_mm) or slice_band_mm <= 0:
        raise PencilMeasurementError("Ball-width slice band must be positive")
    if not np.isfinite(slice_step_mm) or slice_step_mm <= 0:
        raise PencilMeasurementError("Ball-width slice step must be positive")
    if foot.length_mm <= 0:
        raise PencilMeasurementError("Foot length must be positive")
    if foot.mask.shape != rectified.image_bgr.shape[:2]:
        raise PencilMeasurementError("Foot mask and rectified image do not match")

    pixels_per_mm = float(rectified.pixels_per_mm)
    toe = np.asarray(foot.toe_rectified_px, dtype=np.float64)
    # The rectified board uses +X along the paper width and +Y from the toe
    # marker row toward the heel-side row.  Candidate segments therefore keep
    # a constant Y coordinate and are physically parallel to the paper edge.

    y_coordinates, x_coordinates = np.where(foot.mask > 0)
    if len(x_coordinates) < 100:
        raise PencilMeasurementError("Foot mask is too small for ball width")

    zone_start_mm = foot.length_mm * start
    zone_end_mm = foot.length_mm * end
    visible_reach_mm = float(np.max(y_coordinates) - toe[1]) / pixels_per_mm
    if zone_end_mm > visible_reach_mm + slice_band_mm:
        raise PencilMeasurementError(
            "The visible foot mask does not reach the configured ball-width zone"
        )

    half_band_mm = slice_band_mm * 0.5
    centers = np.arange(
        zone_start_mm + half_band_mm,
        zone_end_mm - half_band_mm + slice_step_mm * 0.25,
        slice_step_mm,
        dtype=np.float64,
    )
    # Build an exact horizontal width profile.  The longest consecutive run in
    # each row prevents separated toes or isolated mask fragments from being
    # joined into an artificial width.
    row_profiles: dict[int, tuple[float, int, int]] = {}
    for row in np.unique(y_coordinates):
        values = np.sort(x_coordinates[y_coordinates == row])
        if len(values) < 2:
            continue
        split_indices = np.where(np.diff(values) > 1)[0] + 1
        runs = np.split(values, split_indices)
        longest = max(runs, key=lambda run: (int(run[-1] - run[0]), len(run)))
        if len(longest) < 2:
            continue
        left_x = int(longest[0])
        right_x = int(longest[-1])
        row_profiles[int(row)] = (
            float(right_x - left_x) / pixels_per_mm,
            left_x,
            right_x,
        )

    candidates: list[tuple[float, float, int, int, int, float]] = []
    minimum_band_rows = max(3, int(round(pixels_per_mm * slice_band_mm * 0.5)))
    for center_mm in centers:
        target_y = float(toe[1] + center_mm * pixels_per_mm)
        band_rows = [
            (row, profile)
            for row, profile in row_profiles.items()
            if abs(float(row) - target_y) <= half_band_mm * pixels_per_mm
        ]
        if len(band_rows) < minimum_band_rows:
            continue
        band_widths = np.asarray(
            [profile[0] for _, profile in band_rows], dtype=np.float64
        )
        # Median across a 5 mm neighbourhood suppresses one-row segmentation
        # spikes while retaining a genuinely broad metatarsal plateau.
        stable_width = float(np.median(band_widths))
        representative_row, representative_profile = min(
            band_rows,
            key=lambda item: (
                abs(item[1][0] - stable_width),
                abs(float(item[0]) - target_y),
            ),
        )
        raw_width, left_x, right_x = representative_profile
        if 40.0 <= stable_width <= 150.0 and 40.0 <= raw_width <= 150.0:
            candidates.append(
                (
                    float(center_mm),
                    stable_width,
                    representative_row,
                    left_x,
                    right_x,
                    raw_width,
                )
            )
    if len(candidates) < 5:
        raise PencilMeasurementError(
            "Too few stable forefoot cross-sections for ball width"
        )

    stable_widths = np.asarray([item[1] for item in candidates], dtype=np.float64)
    target_distance_mm = foot.length_mm * target_fraction
    # A maximum-width rule is still vulnerable to the distal toe/phalange
    # silhouette when the profile decreases throughout the MTP zone.  Select
    # the stable row nearest the conservative 35% anatomical proxy instead.
    selected = min(
        candidates,
        key=lambda item: (
            abs((float(item[2]) - toe[1]) / pixels_per_mm - target_distance_mm),
            abs(item[5] - item[1]),
            -item[1],
        ),
    )
    _, _, selected_row, left_x, right_x, width_mm = selected
    if not 50.0 <= width_mm <= 140.0:
        raise PencilMeasurementError(
            f"Implausible ball width: {width_mm:.1f} mm"
        )

    endpoints_px = np.asarray(
        [[left_x, selected_row], [right_x, selected_row]], dtype=np.float32
    )
    center_px = endpoints_px.mean(axis=0).astype(np.float64)
    origin = np.asarray(rectified.world_origin_mm, dtype=np.float32)
    endpoints_world = endpoints_px / pixels_per_mm + origin
    center_world = center_px.astype(np.float32) / pixels_per_mm + origin
    distance_from_toe_mm = float(selected_row - toe[1]) / pixels_per_mm
    fraction_from_toe = float(distance_from_toe_mm / foot.length_mm)
    angle_degrees = 0.0
    return FootBallWidthMeasurement(
        width_mm=float(width_mm),
        endpoints_rectified_px=endpoints_px,
        endpoints_world_mm=endpoints_world.astype(np.float32),
        center_rectified_px=center_px.astype(np.float32),
        center_world_mm=center_world.astype(np.float32),
        distance_from_toe_mm=distance_from_toe_mm,
        fraction_from_toe=fraction_from_toe,
        fraction_from_heel=float(1.0 - fraction_from_toe),
        search_start_fraction_from_toe=start,
        search_end_fraction_from_toe=end,
        target_fraction_from_toe=target_fraction,
        axis_angle_degrees=angle_degrees,
        candidate_count=len(candidates),
        candidate_width_spread_mm=float(
            np.percentile(stable_widths, 95.0)
            - np.percentile(stable_widths, 50.0)
        ),
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    colour: tuple[int, int, int],
    scale: float = 0.72,
    thickness: int = 2,
) -> None:
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


def _draw_status_panel(
    image: np.ndarray,
    *,
    measurement_valid: bool,
    invalid_reasons: Sequence[str],
) -> None:
    lines = [
        (
            "BALL WIDTH: VALID"
            if measurement_valid
            else "BALL WIDTH: DIAGNOSTIC - MARKER GEOMETRY WARNING"
        ),
        "LINE: toe-cleared MTP-zone proxy, parallel to paper",
        "MASK: YOLO foot instance segmentation",
    ]
    if invalid_reasons:
        lines.append("Warning: " + ", ".join(invalid_reasons))
    panel = image.copy()
    height = 38 + 34 * len(lines)
    cv2.rectangle(panel, (12, 12), (min(image.shape[1] - 12, 960), height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.58, image, 0.42, 0, image)
    for index, line in enumerate(lines):
        _put_text(
            image,
            line,
            (26, 45 + index * 34),
            colour=(40, 80, 255) if (index == 0 and not measurement_valid) else (245, 245, 245),
            scale=0.62,
            thickness=2,
        )


def annotate_ball_widths_rectified(
    rectified: RectifiedBoard,
    feet: Mapping[str, StaticFootMeasurement],
    widths: Mapping[str, FootBallWidthMeasurement],
    *,
    measurement_valid: bool,
    invalid_reasons: Sequence[str] = (),
) -> np.ndarray:
    annotated = rectified.image_bgr.copy()
    for side, ball_width in widths.items():
        foot = feet[side]
        colour = _SIDE_COLOURS.get(side, (0, 220, 255))
        overlay = annotated.copy()
        overlay[foot.mask > 0] = colour
        cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
        endpoints = np.rint(ball_width.endpoints_rectified_px).astype(int)
        cv2.line(annotated, tuple(endpoints[0]), tuple(endpoints[1]), colour, 8, cv2.LINE_AA)
        for point in endpoints:
            cv2.circle(annotated, tuple(point), 12, colour, -1, cv2.LINE_AA)
        center = np.rint(ball_width.center_rectified_px).astype(int)
        display_side = side.replace("image_", "").upper()
        _put_text(
            annotated,
            f"{display_side} BALL WIDTH {ball_width.width_mm / 10.0:.1f} cm",
            (int(center[0]) + 18, int(center[1]) - 12),
            colour=colour,
        )
    _draw_status_panel(
        annotated,
        measurement_valid=measurement_valid,
        invalid_reasons=invalid_reasons,
    )
    return annotated


def annotate_ball_widths_original(
    image_bgr: np.ndarray,
    rectified: RectifiedBoard,
    feet: Mapping[str, StaticFootMeasurement],
    widths: Mapping[str, FootBallWidthMeasurement],
    *,
    measurement_valid: bool,
    invalid_reasons: Sequence[str] = (),
) -> np.ndarray:
    annotated = image_bgr.copy()
    rectified_to_image = np.linalg.inv(rectified.image_to_rectified)
    for side, ball_width in widths.items():
        foot = feet[side]
        colour = _SIDE_COLOURS.get(side, (0, 220, 255))
        projected_mask = cv2.warpPerspective(
            foot.mask,
            rectified_to_image,
            (annotated.shape[1], annotated.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlay = annotated.copy()
        overlay[projected_mask > 0] = colour
        cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
        contours, _ = cv2.findContours(
            projected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, colour, 3, cv2.LINE_AA)
        endpoints = cv2.perspectiveTransform(
            ball_width.endpoints_rectified_px.reshape(1, 2, 2),
            rectified_to_image,
        ).reshape(2, 2)
        endpoints_int = np.rint(endpoints).astype(int)
        cv2.line(
            annotated,
            tuple(endpoints_int[0]),
            tuple(endpoints_int[1]),
            colour,
            8,
            cv2.LINE_AA,
        )
        for point in endpoints_int:
            cv2.circle(annotated, tuple(point), 12, colour, -1, cv2.LINE_AA)
        center = np.rint(endpoints.mean(axis=0)).astype(int)
        display_side = side.replace("image_", "").upper()
        _put_text(
            annotated,
            f"{display_side} BALL WIDTH {ball_width.width_mm / 10.0:.1f} cm",
            (int(center[0]) + 18, int(center[1]) - 12),
            colour=colour,
        )
    _draw_status_panel(
        annotated,
        measurement_valid=measurement_valid,
        invalid_reasons=invalid_reasons,
    )
    return annotated
