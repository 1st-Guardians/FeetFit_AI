"""Visual evidence for a partially covered rig marker, never measurement corners.

Two observed adjacent outer corners, visible dictionary-pattern cells and white
quiet space constrain a local completion. The other two corners must lie well
inside the same supplied foot mask. This does not decode a partial marker ID:
the expected partner ID comes from the known two-marker rig.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


PARTIAL_MARKER_DIAGNOSTIC_REASON_CODES = frozenset({
    "PARTIAL_MARKER_NOT_FOUND",
    "PARTIAL_MARKER_CORNERS_INSUFFICIENT",
    "PARTIAL_MARKER_FOOT_OVERLAP_UNCONFIRMED",
    "PARTIAL_MARKER_PATTERN_UNCONFIRMED",
    "PARTIAL_MARKER_INPUT_INVALID",
    "PARTIAL_MARKER_PROCESSING_FAILED",
    "PARTIAL_MARKER_CORNERS_COVERED_BY_FOOT",
})
_DIAGNOSTIC_COUNT_LIMIT = 9999


@dataclass(frozen=True, slots=True)
class PartialMarkerFootOcclusion:
    missing_marker_id: int
    marker_corners: tuple[tuple[float, float], ...]
    visible_corner_count: int
    covered_corner_count: int
    overlap_ratio: float
    pattern_score: float


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _sample(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    coordinates = np.rint(points).astype(int)
    return values[coordinates[:, 1], coordinates[:, 0]]


def _pattern_metrics(gray, feet, quad, bits):
    destination = np.float32([[8, 8], [55, 8], [55, 55], [8, 55]])
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    warped = cv2.warpPerspective(gray, transform, (64, 64), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    covered = cv2.warpPerspective(feet.astype(np.uint8), transform, (64, 64), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    # Keep interpolation or foot-boundary pixels out of the printed-cell test.
    covered = cv2.dilate(covered, np.ones((3, 3), dtype=np.uint8)) > 0
    samples, expected, payload, border, quiet = [], [], [], [], []
    for row in range(8):
        for column in range(8):
            ys, xs = slice(row * 8 + 2, row * 8 + 6), slice(column * 8 + 2, column * 8 + 6)
            if float(np.mean(covered[ys, xs])) > 0.05:
                continue
            value = float(np.median(warped[ys, xs]))
            is_quiet = row in (0, 7) or column in (0, 7)
            bit = 1 if is_quiet else int(bits[row - 1, column - 1])
            index = len(samples)
            samples.append(value)
            expected.append(bit)
            if is_quiet:
                quiet.append(index)
            elif row in (1, 6) or column in (1, 6):
                border.append(index)
            else:
                payload.append(index)
    if len(payload) < 4 or len(border) < 6 or len(quiet) < 6:
        return None
    values = np.asarray(samples)
    labels = np.asarray(expected)
    payload_labels = labels[payload]
    if np.count_nonzero(payload_labels == 0) < 2 or np.count_nonzero(payload_labels == 1) < 2:
        return None
    black = float(np.median(values[labels == 0]))
    white = float(np.median(values[labels == 1]))
    if white - black < 55.0 or black > 95.0 or white < 115.0:
        return None
    threshold = (black + white) / 2.0
    observed = values > threshold
    if not np.array_equal(observed[payload], payload_labels.astype(bool)):
        return None
    if float(np.mean(observed[border] == 0)) < 0.9 or float(np.mean(observed[quiet] == 1)) < 0.9:
        return None
    correlation = float(np.corrcoef(values, labels)[0, 1])
    if not math.isfinite(correlation) or correlation < 0.88:
        return None
    return {
        "score": correlation, "clear_payload_cells": len(payload), "clear_border_cells": len(border),
        "clear_quiet_cells": len(quiet), "black_median": black, "white_median": white,
        "payload_black_cells": int(np.count_nonzero(payload_labels == 0)),
        "payload_white_cells": int(np.count_nonzero(payload_labels == 1)),
    }


def _candidate_edges(contour, anchor_x, anchor_y, anchor_size, gray, feet):
    # A fixed pixel tolerance splits a real straight edge at resampling/JPEG
    # steps in larger images. Scale the geometric tolerance with the decoded
    # anchor; both genuine endpoint corners still need independent quiet space.
    epsilon = max(1.2, 0.025 * anchor_size)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)
    if len(polygon) < 4:
        return
    h, w = gray.shape
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        before = polygon[(index - 1) % len(polygon)]
        after = polygon[(index + 2) % len(polygon)]
        vector = second - first
        length = float(np.linalg.norm(vector))
        if not 0.35 * anchor_size <= length <= 1.35 * anchor_size:
            continue
        direction = vector / length
        preceding, following = before - first, after - second
        if min(np.linalg.norm(preceding), np.linalg.norm(following)) < max(3.0, 0.07 * length):
            continue
        # Both endpoints need a real adjacent outer edge. A cut made by a foot
        # mask does not provide the white quiet-space support checked below.
        if max(abs(float(np.dot(_unit(preceding), direction))),
               abs(float(np.dot(_unit(following), direction)))) > 0.4:
            continue
        vertical = abs(float(np.dot(direction, anchor_y))) >= math.cos(math.radians(22))
        horizontal = abs(float(np.dot(direction, anchor_x))) >= math.cos(math.radians(22))
        if not (vertical or horizontal):
            continue
        along = anchor_y if vertical else anchor_x
        if np.dot(direction, along) < 0:
            first, second = second, first
            direction = -direction
        across = anchor_x if vertical else anchor_y
        for sign in (1, -1):
            inside = across * sign
            positions = first + np.linspace(0.15, 0.85, 7)[:, None] * (second - first)
            black_points = positions + 2.0 * inside
            quiet_points = positions - 3.0 * inside
            # White continuation beyond BOTH observed corner endpoints is
            # essential: one visible corner with a foot-cut edge is unknown.
            endpoint_quiet = np.vstack((first - 3.0 * direction + 2.0 * inside,
                                         second + 3.0 * direction + 2.0 * inside))
            all_points = np.vstack((black_points, quiet_points, endpoint_quiet, first, second))
            if np.any(all_points < 1) or np.any(all_points[:, 0] > w - 2) or np.any(all_points[:, 1] > h - 2):
                continue
            if bool(_sample(feet, all_points).any()):
                continue
            dark_values = _sample(gray, black_points)
            endpoint_values = _sample(gray, endpoint_quiet)
            if not np.all(endpoint_values > 115):
                continue
            light_values = _sample(gray, quiet_points)
            if float(np.mean(dark_values < 95)) < 0.85 or float(np.mean(light_values > 115)) < 0.85:
                continue
            if float(np.median(light_values) - np.median(dark_values)) < 55:
                continue
            yield first, second, vertical, sign, length


def find_partial_marker_foot_occlusion(
    source_bgr: np.ndarray,
    *,
    visible_marker_id: int,
    visible_corners: np.ndarray,
    foot_components: tuple[np.ndarray, ...],
    dictionary_name: str = "DICT_4X4_50",
    diagnostics: dict | None = None,
) -> PartialMarkerFootOcclusion | None:
    """Return only a bounded partial-marker/foot match; uncertainty is None.

    Optional diagnostics contain only a closed reason code and saturated integer
    stage counts. They do not expose image data or exception messages.
    """
    trace = {
        "reason_code": "PARTIAL_MARKER_INPUT_INVALID",
        "dark_component_count": 0,
        "edge_count": 0,
        "covered_candidate_count": 0,
        "pattern_candidate_count": 0,
    }

    def increment(key):
        trace[key] = min(_DIAGNOSTIC_COUNT_LIMIT, trace[key] + 1)

    try:
        if (not isinstance(source_bgr, np.ndarray) or source_bgr.dtype != np.uint8
                or source_bgr.ndim != 3 or source_bgr.shape[2] != 3
                or type(visible_marker_id) is not int or visible_marker_id not in (1, 4)):
            return None
        h, w = source_bgr.shape[:2]
        anchor = np.asarray(visible_corners, dtype=np.float64)
        if anchor.shape != (4, 2) or not np.isfinite(anchor).all():
            return None
        if np.any(anchor < 2) or np.any(anchor[:, 0] > w - 3) or np.any(anchor[:, 1] > h - 3):
            return None
        if not cv2.isContourConvex(anchor.astype(np.float32)) or cv2.contourArea(anchor.astype(np.float32), oriented=True) <= 0:
            return None
        lengths = np.linalg.norm(np.roll(anchor, -1, axis=0) - anchor, axis=1)
        if float(lengths.min()) < 20 or float(lengths.max() / lengths.min()) > 2.0:
            return None
        anchor_size = float(np.median(lengths))
        anchor_x = _unit(anchor[1] + anchor[2] - anchor[0] - anchor[3])
        anchor_y = _unit(anchor[2] + anchor[3] - anchor[0] - anchor[1])
        if abs(float(np.dot(anchor_x, anchor_y))) > 0.4:
            return None
        if not isinstance(dictionary_name, str) or not dictionary_name.startswith("DICT_4X4_"):
            return None
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if type(dictionary_id) is not int:
            return None
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if dictionary.markerSize != 4:
            return None
        missing_id = 4 if visible_marker_id == 1 else 1
        marker = cv2.aruco.generateImageMarker(dictionary, missing_id, 48)
        bits = (marker.reshape(6, 8, 6, 8).mean(axis=(1, 3)) > 127).astype(np.uint8)
        masks = []
        for component in foot_components:
            values = np.asarray(component)
            if values.shape != (h, w) or values.dtype.kind not in "buif" or not np.isfinite(values).all():
                return None
            if np.any(values < 0) or np.any(values > 1):
                return None
            mask = values > 0
            if np.count_nonzero(mask) >= 4.0 * anchor_size ** 2:
                masks.append(mask)
        if not masks:
            trace["reason_code"] = "PARTIAL_MARKER_FOOT_OVERLAP_UNCONFIRMED"
            return None
        anchor_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(anchor_mask, np.rint(anchor).astype(np.int32), 1)
        # A component claiming the already decoded marker is not trustworthy
        # evidence for a different covered marker in this same image.
        masks = [mask for mask in masks if float(np.mean(mask[anchor_mask > 0])) <= 0.10]
        if not masks:
            trace["reason_code"] = "PARTIAL_MARKER_FOOT_OVERLAP_UNCONFIRMED"
            return None
        mask_areas = [int(np.count_nonzero(mask)) for mask in masks]
        feet = np.logical_or.reduce(masks)
        interiors = [cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) for mask in masks]
        gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices((h, w), dtype=np.float32)
        relative_x, relative_y = xx - anchor[:, 0].mean(), yy - anchor[:, 1].mean()
        forward = relative_x * anchor_y[0] + relative_y * anchor_y[1]
        if visible_marker_id == 4:
            forward = -forward
        lateral = relative_x * anchor_x[0] + relative_y * anchor_x[1]
        corridor = ((forward >= 4.0 * anchor_size) & (forward <= 13.0 * anchor_size)
                    & (np.abs(lateral) <= 2.2 * anchor_size))
        dark = ((gray < 80) & ~feet & corridor).astype(np.uint8)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not max(35.0, 0.015 * anchor_size ** 2) <= area <= 1.7 * anchor_size ** 2:
                continue
            increment("dark_component_count")
            for first, second, vertical, sign, length in _candidate_edges(
                contour, anchor_x, anchor_y, anchor_size, gray, feet,
            ):
                increment("edge_count")
                across = anchor_x if vertical else anchor_y
                for angle in (-8.0, -4.0, 0.0, 4.0, 8.0):
                    theta = math.radians(angle)
                    rotation = np.array(((math.cos(theta), -math.sin(theta)), (math.sin(theta), math.cos(theta))))
                    completion_axis = rotation @ across * sign
                    for ratio in np.linspace(0.7, 1.7, 21):
                        width = float(length * ratio)
                        if not 0.35 * anchor_size <= width <= 1.4 * anchor_size:
                            continue
                        displacement = completion_axis * width
                        if vertical:
                            quad = np.array((first, first + displacement, second + displacement, second)) if sign > 0 else np.array((first + displacement, first, second, second + displacement))
                        else:
                            quad = np.array((first, second, second + displacement, first + displacement)) if sign > 0 else np.array((first + displacement, second + displacement, second, first))
                        if np.any(quad < 3) or np.any(quad[:, 0] > w - 4) or np.any(quad[:, 1] > h - 4):
                            continue
                        if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(quad.astype(np.float32), oriented=True) <= 0:
                            continue
                        hidden_corners = np.vstack((first + displacement, second + displacement))
                        corner_patch = (hidden_corners[:, None, :] + np.array(
                            [(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)],
                        )[None, :, :]).reshape(-1, 2)
                        quad_area = float(cv2.contourArea(quad.astype(np.float32)))
                        covering = [i for i, interior in enumerate(interiors)
                                    if mask_areas[i] >= 4.0 * quad_area
                                    and np.all(_sample(interior, corner_patch) >= 3.0)]
                        if not covering:
                            continue
                        polygon_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(polygon_mask, np.rint(quad).astype(np.int32), 1)
                        overlap = max(float(np.mean(masks[i][polygon_mask > 0])) for i in covering)
                        if not 0.25 <= overlap <= 0.85:
                            continue
                        increment("covered_candidate_count")
                        metrics = _pattern_metrics(gray, feet, quad, bits)
                        if metrics is None:
                            continue
                        increment("pattern_candidate_count")
                        candidates.append(PartialMarkerFootOcclusion(
                            missing_id, tuple(tuple(float(value) for value in point) for point in quad),
                            2, 2, overlap, metrics["score"],
                        ))
        if candidates:
            trace["reason_code"] = "PARTIAL_MARKER_CORNERS_COVERED_BY_FOOT"
            return max(candidates, key=lambda candidate: candidate.pattern_score)
        if not trace["dark_component_count"]:
            trace["reason_code"] = "PARTIAL_MARKER_NOT_FOUND"
        elif not trace["edge_count"]:
            trace["reason_code"] = "PARTIAL_MARKER_CORNERS_INSUFFICIENT"
        elif not trace["covered_candidate_count"]:
            trace["reason_code"] = "PARTIAL_MARKER_FOOT_OVERLAP_UNCONFIRMED"
        else:
            trace["reason_code"] = "PARTIAL_MARKER_PATTERN_UNCONFIRMED"
        return None
    except Exception:
        # No image, model, or filesystem data is included in errors or results.
        trace["reason_code"] = "PARTIAL_MARKER_PROCESSING_FAILED"
        return None
    finally:
        if isinstance(diagnostics, dict):
            diagnostics.update(trace)
