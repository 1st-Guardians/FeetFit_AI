from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np


# Contour HVA v6 parameters supplied for the hallux-valgus pipeline.
MIN_MASK_AREA = 5000
MORPH_KERNEL = 7

PROFILE_BINS = 260
PROFILE_SMOOTH_KERNEL = 11

TOE_REGION_RATIO = 0.42
BIG_TOE_STRIP_RATIO = 0.35
HALLUX_AXIS_START_RATIO = 0.18
HALLUX_AXIS_END_RATIO = 0.72

LITTLE_BODY_SEARCH_TOP_RATIO = 0.40
LITTLE_BODY_SEARCH_BOTTOM_RATIO = 0.72
LITTLE_BODY_UPPER_WINDOW_RATIO = 0.07
LITTLE_BODY_LOWER_WINDOW_RATIO = 0.14

LITTLE_BODY_SLOPE_WEIGHT = 0.25
LITTLE_BODY_STABILITY_WEIGHT = 0.30
LITTLE_BODY_POSITION_WEIGHT = 0.45
LITTLE_BODY_SMOOTH_KERNEL = 21

METATARSAL_SEARCH_TOP_RATIO = 0.20
METATARSAL_SEARCH_BOTTOM_RATIO = 0.68

MEDIAL_REFERENCE_START_W = 0.08
MEDIAL_REFERENCE_END_W = 0.62

RANSAC_ITERATIONS = 400
RANSAC_DISTANCE_RATIO = 0.012

MAX_INTERSECTION_DISTANCE_W = 0.90

HALLUX_LINE_LENGTH_W = 0.385
REFERENCE_LINE_LENGTH_W = 0.525

FOOT_LINE_COLOR = (26, 26, 26, 255)
FOOT_FILL_COLOR = (255, 255, 255, 255)
AXIS_COLOR = (255, 136, 0, 255)

CONTOUR_THICKNESS = 3
LINE_THICKNESS = 5
INTERSECTION_RADIUS = 5
CROP_PADDING = 10


def normalize(vector: np.ndarray | list[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm < 1e-10:
        return np.array([0.0, 0.0], dtype=np.float64)
    return value / norm


def angle_between_lines(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    unit_1 = normalize(vector_1)
    unit_2 = normalize(vector_2)
    dot = float(np.clip(abs(np.dot(unit_1, unit_2)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def foot_side_to_medial_side(foot_side: str) -> str:
    if foot_side == "left":
        return "right"
    if foot_side == "right":
        return "left"
    raise ValueError(f"Unsupported foot side: {foot_side}")


def get_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    return max(contours, key=cv2.contourArea) if contours else None


def clean_foot_mask(mask: np.ndarray) -> np.ndarray | None:
    """Apply the supplied morphology and retain one foot component."""

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("Foot mask must be a two-dimensional array.")
    binary = (value > 0).astype(np.uint8) * 255
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contour = get_largest_contour(binary)
    if contour is None or cv2.contourArea(contour) < MIN_MASK_AREA:
        return None
    cleaned = np.zeros_like(binary)
    cv2.drawContours(cleaned, [contour], -1, 255, -1)
    return cleaned


def contour_to_points(contour: np.ndarray) -> np.ndarray:
    return contour.reshape(-1, 2).astype(np.float64)


def get_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def smooth_1d(values: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if len(value) < 5:
        return value.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, len(value))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return value.copy()
    kernel = np.ones(kernel_size, dtype=np.float64)
    kernel /= kernel.sum()
    padding = kernel_size // 2
    padded = np.pad(value, (padding, padding), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def affine_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([value, np.ones((len(value), 1))])
    return (matrix @ homogeneous.T).T


def rotate_mask_expand(mask: np.ndarray, angle_degree: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated, cv2.invertAffineTransform(matrix)


def find_bottom_cut_edge(mask: np.ndarray) -> float | None:
    bbox = get_mask_bbox(mask)
    contour = get_largest_contour(mask)
    if bbox is None or contour is None:
        return None
    x1, y1, x2, y2 = bbox
    points = contour_to_points(contour)
    foot_height = y2 - y1 + 1
    foot_width = x2 - x1 + 1
    band = max(3, int(foot_height * 0.05))
    bottom_points = points[points[:, 1] >= y2 - band]
    if len(bottom_points) < 10:
        return None
    span = bottom_points[:, 0].max() - bottom_points[:, 0].min()
    if span < foot_width * 0.35:
        return None
    vx, vy, _, _ = cv2.fitLine(
        bottom_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    )
    direction = normalize([float(vx.item()), float(vy.item())])
    return math.degrees(math.atan2(direction[1], direction[0]))


def normalize_orientation(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = find_bottom_cut_edge(mask)
    if angle is not None:
        return rotate_mask_expand(mask, angle)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return mask.copy(), identity


def get_side_profiles(
    mask: np.ndarray,
    bins: int = PROFILE_BINS,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None, None
    _, y1, _, y2 = bbox
    sample_count = min(bins, y2 - y1 + 1)
    y_values = np.linspace(y1, y2, sample_count)
    left_points: list[list[float]] = []
    right_points: list[list[float]] = []
    half_bin = max(1, int((y2 - y1 + 1) / sample_count))
    for y_float in y_values:
        center_y = int(round(y_float))
        top = max(y1, center_y - half_bin)
        bottom = min(y2, center_y + half_bin)
        _, xs = np.where(mask[top : bottom + 1] > 0)
        if len(xs) == 0:
            continue
        left_points.append([float(xs.min()), float(center_y)])
        right_points.append([float(xs.max()), float(center_y)])
    if len(left_points) < 10 or len(right_points) < 10:
        return None, None
    return (
        np.asarray(left_points, dtype=np.float64),
        np.asarray(right_points, dtype=np.float64),
    )


def smooth_profile(profile: np.ndarray | None) -> np.ndarray | None:
    if profile is None:
        return None
    result = profile.copy()
    result[:, 0] = smooth_1d(result[:, 0], PROFILE_SMOOTH_KERNEL)
    return result


def get_big_toe_region(mask: np.ndarray, medial_side: str) -> dict[str, Any] | None:
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    foot_width = x2 - x1 + 1
    foot_height = y2 - y1 + 1
    toe_y2 = int(y1 + foot_height * TOE_REGION_RATIO)
    strip_width = int(foot_width * BIG_TOE_STRIP_RATIO)
    if medial_side == "left":
        roi = mask[y1 : toe_y2 + 1, x1 : min(x2 + 1, x1 + strip_width)]
        x_offset = x1
    else:
        roi_x_start = max(x1, x2 - strip_width + 1)
        roi = mask[y1 : toe_y2 + 1, roi_x_start : x2 + 1]
        x_offset = roi_x_start
    return {
        "roi": roi,
        "x_offset": x_offset,
        "y_offset": y1,
        "bbox_width": foot_width,
    }


def extract_hallux_center_points(mask: np.ndarray, medial_side: str) -> np.ndarray | None:
    region = get_big_toe_region(mask, medial_side)
    if region is None:
        return None
    roi = region["roi"]
    x_offset = int(region["x_offset"])
    y_offset = int(region["y_offset"])
    foot_width = int(region["bbox_width"])
    center_points: list[list[float]] = []
    for local_y in range(roi.shape[0]):
        xs = np.where(roi[local_y] > 0)[0]
        if len(xs) < 2:
            continue
        left_x = float(xs.min())
        right_x = float(xs.max())
        if right_x - left_x < foot_width * 0.06:
            continue
        center_points.append(
            [(left_x + right_x) / 2.0 + x_offset, float(local_y + y_offset)]
        )
    if len(center_points) < 8:
        return None
    points = np.asarray(center_points, dtype=np.float64)
    y_min = points[:, 1].min()
    y_range = points[:, 1].max() - y_min
    start_y = y_min + y_range * HALLUX_AXIS_START_RATIO
    end_y = y_min + y_range * HALLUX_AXIS_END_RATIO
    filtered = points[(points[:, 1] >= start_y) & (points[:, 1] <= end_y)]
    return points if len(filtered) < 5 else filtered


def fit_line_ransac(
    points: np.ndarray,
    scale: float,
    *,
    rng: random.Random | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    value = np.asarray(points, dtype=np.float64)
    if len(value) < 5:
        return None, None
    sampler = rng or random.Random(0)
    threshold = max(1.5, scale * RANSAC_DISTANCE_RATIO)
    best_inliers: np.ndarray | None = None
    best_count = 0
    for _ in range(RANSAC_ITERATIONS):
        index_1, index_2 = sampler.sample(range(len(value)), 2)
        point_1 = value[index_1]
        delta = value[index_2] - point_1
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            continue
        direction = delta / norm
        vectors = value - point_1
        distances = np.abs(
            vectors[:, 0] * direction[1] - vectors[:, 1] * direction[0]
        )
        inliers = distances <= threshold
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 4:
        best_inliers = np.ones(len(value), dtype=bool)
    fit_points = value[best_inliers].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01)
    point = np.array([float(x0.item()), float(y0.item())])
    direction = normalize([float(vx.item()), float(vy.item())])
    if direction[1] < 0:
        direction = -direction
    return point, direction


def line_fit_inlier_ratio(
    points: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
    scale: float,
) -> float:
    value = np.asarray(points, dtype=np.float64)
    if len(value) == 0:
        return 0.0
    unit_direction = normalize(direction)
    vectors = value - np.asarray(point, dtype=np.float64)
    distances = np.abs(
        vectors[:, 0] * unit_direction[1]
        - vectors[:, 1] * unit_direction[0]
    )
    threshold = max(1.5, scale * RANSAC_DISTANCE_RATIO)
    return float(np.mean(distances <= threshold))


def calculate_profile_curvature(profile: np.ndarray) -> np.ndarray | None:
    if profile is None or len(profile) < 10:
        return None
    x_values = smooth_1d(profile[:, 0], PROFILE_SMOOTH_KERNEL)
    y_values = profile[:, 1]
    slope = smooth_1d(np.gradient(x_values, y_values), 9)
    curvature = np.abs(np.gradient(slope, y_values))
    return smooth_1d(curvature, 9)


def find_metatarsal_prominence(
    profile: np.ndarray,
    profile_side: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None
    _, y1, _, y2 = bbox
    foot_height = y2 - y1 + 1
    curvature = calculate_profile_curvature(profile)
    if curvature is None:
        return None
    search_start = y1 + foot_height * METATARSAL_SEARCH_TOP_RATIO
    search_end = y1 + foot_height * METATARSAL_SEARCH_BOTTOM_RATIO
    indices = np.where(
        (profile[:, 1] >= search_start) & (profile[:, 1] <= search_end)
    )[0]
    if len(indices) < 3:
        return None
    candidate_curvature = curvature[indices]
    candidate_x = profile[indices, 0]
    if profile_side == "left":
        prominence = np.max(candidate_x) - candidate_x
    elif profile_side == "right":
        prominence = candidate_x - np.min(candidate_x)
    else:
        return None
    curvature_max = np.max(candidate_curvature)
    prominence_max = np.max(prominence)
    curvature_normalized = (
        candidate_curvature / curvature_max
        if curvature_max > 1e-10
        else np.zeros_like(candidate_curvature)
    )
    prominence_normalized = (
        prominence / prominence_max
        if prominence_max > 1e-10
        else np.zeros_like(prominence)
    )
    score = 0.60 * curvature_normalized + 0.40 * prominence_normalized
    best_index = int(indices[int(np.argmax(score))])
    return profile[best_index].copy()


def find_m1_and_medial_profile(
    mask: np.ndarray,
    medial_side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    left_profile, right_profile = get_side_profiles(mask)
    if left_profile is None or right_profile is None:
        return None
    left_profile = smooth_profile(left_profile)
    right_profile = smooth_profile(right_profile)
    if left_profile is None or right_profile is None:
        return None
    if medial_side == "left":
        medial_profile, profile_side = left_profile, "left"
    else:
        medial_profile, profile_side = right_profile, "right"
    m1 = find_metatarsal_prominence(medial_profile, profile_side, mask)
    return None if m1 is None else (m1, medial_profile)


def extract_medial_reference_points(
    medial_profile: np.ndarray,
    m1: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None
    x1, _, x2, y2 = bbox
    foot_width = x2 - x1 + 1
    start_y = m1[1] + foot_width * MEDIAL_REFERENCE_START_W
    end_y = min(m1[1] + foot_width * MEDIAL_REFERENCE_END_W, float(y2))
    points = medial_profile[
        (medial_profile[:, 1] >= start_y) & (medial_profile[:, 1] <= end_y)
    ]
    if len(points) < 5:
        points = medial_profile[medial_profile[:, 1] > m1[1]]
    return points


def find_little_toe_body_transition(profile: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None
    _, y1, _, y2 = bbox
    foot_height = float(y2 - y1 + 1)
    x_values = smooth_1d(profile[:, 0], LITTLE_BODY_SMOOTH_KERNEL)
    y_values = profile[:, 1].astype(np.float64)
    if len(x_values) < 20:
        return None
    slope = smooth_1d(np.gradient(x_values, y_values), 15)
    search_top = y1 + foot_height * LITTLE_BODY_SEARCH_TOP_RATIO
    search_bottom = y1 + foot_height * LITTLE_BODY_SEARCH_BOTTOM_RATIO
    candidate_indices = np.where(
        (y_values >= search_top) & (y_values <= search_bottom)
    )[0]
    if len(candidate_indices) < 10:
        return None
    upper_window = foot_height * LITTLE_BODY_UPPER_WINDOW_RATIO
    lower_window = foot_height * LITTLE_BODY_LOWER_WINDOW_RATIO
    candidates: list[dict[str, float | int]] = []
    for index in candidate_indices:
        current_y = y_values[index]
        upper_indices = np.where(
            (y_values >= current_y - upper_window) & (y_values < current_y)
        )[0]
        lower_indices = np.where(
            (y_values > current_y) & (y_values <= current_y + lower_window)
        )[0]
        if len(upper_indices) < 4 or len(lower_indices) < 6:
            continue
        upper_slope = float(np.median(slope[upper_indices]))
        lower_slope = float(np.median(slope[lower_indices]))
        slope_change = abs(upper_slope - lower_slope)
        lower_standard_deviation = float(np.std(slope[lower_indices]))
        stability = 1.0 / (1.0 + lower_standard_deviation)
        position = (current_y - search_top) / max(1e-6, search_bottom - search_top)
        candidates.append(
            {
                "index": int(index),
                "slope_change": slope_change,
                "stability": stability,
                "position": position,
            }
        )
    if not candidates:
        return None
    max_change = max(float(item["slope_change"]) for item in candidates)
    max_stability = max(float(item["stability"]) for item in candidates)
    best_score = -1.0
    best_index: int | None = None
    for candidate in candidates:
        slope_change = float(candidate["slope_change"])
        stability = float(candidate["stability"])
        change_normalized = slope_change / max_change if max_change > 1e-10 else 0.0
        stability_normalized = (
            stability / max_stability if max_stability > 1e-10 else 0.0
        )
        score = (
            LITTLE_BODY_SLOPE_WEIGHT * change_normalized
            + LITTLE_BODY_STABILITY_WEIGHT * stability_normalized
            + LITTLE_BODY_POSITION_WEIGHT * float(candidate["position"])
        )
        if score > best_score:
            best_score = score
            best_index = int(candidate["index"])
    if best_index is None:
        return None
    return np.array(
        [x_values[best_index], y_values[best_index]], dtype=np.float64
    )


def find_little_body(mask: np.ndarray, foot_side: str) -> np.ndarray | None:
    left_profile, right_profile = get_side_profiles(mask)
    if left_profile is None or right_profile is None:
        return None
    left_profile = smooth_profile(left_profile)
    right_profile = smooth_profile(right_profile)
    if left_profile is None or right_profile is None:
        return None
    lateral_profile = left_profile if foot_side == "left" else right_profile
    return find_little_toe_body_transition(lateral_profile, mask)


def line_intersection(
    point_1: np.ndarray,
    direction_1: np.ndarray,
    point_2: np.ndarray,
    direction_2: np.ndarray,
) -> np.ndarray | None:
    matrix = np.array(
        [
            [direction_1[0], -direction_2[0]],
            [direction_1[1], -direction_2[1]],
        ],
        dtype=np.float64,
    )
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    distance_1, _ = np.linalg.solve(matrix, point_2 - point_1)
    return point_1 + distance_1 * direction_1


def restore_point(point: np.ndarray, inverse_matrix: np.ndarray) -> np.ndarray:
    return affine_transform_points(np.asarray([point]), inverse_matrix)[0]


def restore_direction(
    point: np.ndarray,
    direction: np.ndarray,
    inverse_matrix: np.ndarray,
) -> np.ndarray:
    test_points = np.array([point, point + direction * 100.0])
    restored = affine_transform_points(test_points, inverse_matrix)
    return normalize(restored[1] - restored[0])


def analyze_foot_mask(mask: np.ndarray, foot_side: str) -> dict[str, Any] | None:
    """Calculate the supplied silhouette-based visible hallux angle for one foot."""

    if foot_side not in {"left", "right"}:
        raise ValueError(f"Unsupported foot side: {foot_side}")
    cleaned_mask = clean_foot_mask(mask)
    if cleaned_mask is None:
        return None
    medial_side = foot_side_to_medial_side(foot_side)
    normalized_mask, inverse_matrix = normalize_orientation(cleaned_mask)
    bbox = get_mask_bbox(normalized_mask)
    if bbox is None:
        return None
    x1, _, x2, _ = bbox
    foot_width = x2 - x1 + 1
    rng = random.Random(0)

    hallux_points = extract_hallux_center_points(normalized_mask, medial_side)
    if hallux_points is None:
        return None
    hallux_point, hallux_direction = fit_line_ransac(
        hallux_points,
        foot_width,
        rng=rng,
    )
    if hallux_point is None or hallux_direction is None:
        return None
    hallux_fit_inlier_ratio = line_fit_inlier_ratio(
        hallux_points,
        hallux_point,
        hallux_direction,
        foot_width,
    )

    medial_result = find_m1_and_medial_profile(normalized_mask, medial_side)
    if medial_result is None:
        return None
    m1, medial_profile = medial_result
    little_body = find_little_body(normalized_mask, foot_side)
    if little_body is None:
        return None
    centerline_direction = normalize(little_body - m1)
    if np.linalg.norm(centerline_direction) < 1e-10:
        return None
    centerline_point = (m1 + little_body) / 2.0

    medial_reference_points = extract_medial_reference_points(
        medial_profile, m1, normalized_mask
    )
    if medial_reference_points is None or len(medial_reference_points) < 5:
        return None
    medial_point, medial_direction = fit_line_ransac(
        medial_reference_points,
        foot_width,
        rng=rng,
    )
    if medial_point is None or medial_direction is None:
        return None
    medial_fit_inlier_ratio = line_fit_inlier_ratio(
        medial_reference_points,
        medial_point,
        medial_direction,
        foot_width,
    )

    intersection = line_intersection(
        hallux_point,
        hallux_direction,
        centerline_point,
        centerline_direction,
    )
    if intersection is None:
        intersection = centerline_point.copy()
    elif np.linalg.norm(intersection - centerline_point) > (
        foot_width * MAX_INTERSECTION_DISTANCE_W
    ):
        intersection = centerline_point.copy()

    angle = angle_between_lines(hallux_direction, medial_direction)
    normalized_contour = get_largest_contour(normalized_mask)
    if normalized_contour is None:
        return None
    contour = affine_transform_points(
        contour_to_points(normalized_contour), inverse_matrix
    )
    hallux_point_original = restore_point(hallux_point, inverse_matrix)
    hallux_direction_original = restore_direction(
        hallux_point, hallux_direction, inverse_matrix
    )
    intersection_original = restore_point(intersection, inverse_matrix)
    reference_point_original = restore_point(intersection, inverse_matrix)
    reference_direction_original = restore_direction(
        intersection, medial_direction, inverse_matrix
    )
    original_foot_width = float(np.max(contour[:, 0]) - np.min(contour[:, 0]))
    return {
        "angle": angle,
        "contour": contour,
        "hallux_point": hallux_point_original,
        "hallux_dir": hallux_direction_original,
        "intersection": intersection_original,
        "reference_point": reference_point_original,
        "reference_dir": reference_direction_original,
        "foot_width": original_foot_width,
        "foot_side": foot_side,
        "m1_normalized": m1,
        "little_body_normalized": little_body,
        "hallux_fit_point_count": int(len(hallux_points)),
        "medial_fit_point_count": int(len(medial_reference_points)),
        "hallux_fit_inlier_ratio": hallux_fit_inlier_ratio,
        "medial_fit_inlier_ratio": medial_fit_inlier_ratio,
        "cleaned_mask": cleaned_mask,
    }


def draw_hallux_line_to_intersection(
    image: np.ndarray,
    intersection: np.ndarray,
    direction: np.ndarray,
    length: float,
) -> None:
    unit_direction = normalize(direction)
    start = tuple(np.round(intersection - unit_direction * length).astype(int))
    end = tuple(np.round(intersection).astype(int))
    cv2.line(
        image,
        start,
        end,
        AXIS_COLOR,
        LINE_THICKNESS,
        cv2.LINE_AA,
    )


def draw_reference_line_from_intersection(
    image: np.ndarray,
    intersection: np.ndarray,
    direction: np.ndarray,
    length: float,
) -> None:
    unit_direction = normalize(direction)
    start = tuple(np.round(intersection).astype(int))
    end = tuple(np.round(intersection + unit_direction * length).astype(int))
    cv2.line(
        image,
        start,
        end,
        AXIS_COLOR,
        LINE_THICKNESS,
        cv2.LINE_AA,
    )


def visualization_crop_bbox(
    image_shape: tuple[int, int] | tuple[int, int, int],
    contour: np.ndarray,
    padding: int = CROP_PADDING,
) -> tuple[int, int, int, int] | None:
    points = np.asarray(contour, dtype=np.float64)
    x_min = int(np.floor(np.min(points[:, 0])))
    y_min = int(np.floor(np.min(points[:, 1])))
    x_max = int(np.ceil(np.max(points[:, 0])))
    y_max = int(np.ceil(np.max(points[:, 1])))
    height, width = image_shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width - 1, x_max + padding)
    y_max = min(height - 1, y_max + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def crop_single_foot(
    image: np.ndarray,
    contour: np.ndarray,
    padding: int = CROP_PADDING,
) -> np.ndarray | None:
    bbox = visualization_crop_bbox(image.shape, contour, padding)
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    return image[y_min : y_max + 1, x_min : x_max + 1].copy()


def visualize_result(
    image_shape: tuple[int, int] | tuple[int, int, int],
    result: dict[str, Any],
) -> np.ndarray:
    """Render the supplied white-filled foot and axes on a transparent background."""

    height, width = image_shape[:2]
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    source_contour = np.asarray(result["contour"], dtype=np.float64)
    contour = np.round(source_contour).astype(np.int32)
    foot_width = float(result["foot_width"])
    intersection = np.asarray(result["intersection"], dtype=np.float64)
    hallux_direction = np.asarray(result["hallux_dir"], dtype=np.float64)
    reference_direction = np.asarray(result["reference_dir"], dtype=np.float64)
    cv2.fillPoly(canvas, [contour], FOOT_FILL_COLOR, cv2.LINE_AA)
    cv2.polylines(
        canvas,
        [contour],
        True,
        FOOT_LINE_COLOR,
        CONTOUR_THICKNESS,
        cv2.LINE_AA,
    )
    draw_hallux_line_to_intersection(
        canvas,
        intersection,
        hallux_direction,
        foot_width * HALLUX_LINE_LENGTH_W,
    )
    draw_reference_line_from_intersection(
        canvas,
        intersection,
        reference_direction,
        foot_width * REFERENCE_LINE_LENGTH_W,
    )
    cv2.circle(
        canvas,
        tuple(np.round(intersection).astype(int)),
        INTERSECTION_RADIUS,
        AXIS_COLOR,
        -1,
        cv2.LINE_AA,
    )
    cropped = crop_single_foot(canvas, source_contour)
    if cropped is None:
        raise ValueError("Could not crop the hallux contour visualization.")
    return cropped
