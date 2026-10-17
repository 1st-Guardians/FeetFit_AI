"""Deterministic, conservative cleanup for support-clipped Full SAM masks.

The cleanup is intentionally component based.  It never runs a learned model,
never performs a morphological close, and never connects two foreground
regions.  The largest component and other genuinely large anatomical anchors
are retained.  Smaller components are retained only when they are credible
toe contacts in the distal part of the foot.  Everything else is recorded in
the returned component audit before it is removed.

Tiny enclosed holes may optionally be filled.  Hole filling is disabled by
default and can be limited by area, span, and circularity so that an arch
cavity cannot be filled accidentally.  Hole candidates are measured only
after disconnected components have been removed, and candidates containing a
discarded component are rejected in full.  This prevents removed foreground
from being resurrected by the fill step.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import cv2
import numpy as np


FULL_SAM_CLEANUP_FORMAT = "full_sam_component_cleanup_v1"
ToeDirection = Literal["auto", "top", "bottom", "left", "right"]


@dataclass(frozen=True)
class FullSamCleanupConfig:
    """Auditable thresholds for conservative disconnected-component cleanup.

    ``anchor_component_area_ratio`` is deliberately high.  Components below
    ten percent of the Full SAM foreground are not treated as heel/forefoot
    anchors merely because they are larger than a speck.  A separate, much
    lower threshold applies only in the anatomically restricted toe zone.
    """

    anchor_component_area_ratio: float = 0.10
    toe_component_support_area_ratio: float = 0.0001
    distal_zone_fraction: float = 0.22
    max_toe_gap_ratio: float = 0.18
    toe_lateral_margin_ratio: float = 0.15
    min_toe_component_area_pixels: int = 20
    reference_image_size: int = 1024
    toe_direction: ToeDirection = "auto"
    connectivity: int = 8
    max_hole_area_pixels: int = 0
    max_hole_span_pixels: int = 0
    min_hole_circularity: float = 0.0
    small_hole_circularity_exemption_pixels: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.anchor_component_area_ratio) <= 1.0:
            raise ValueError("anchor_component_area_ratio must be in (0, 1]")
        if not 0.0 <= float(self.toe_component_support_area_ratio) < 1.0:
            raise ValueError("toe_component_support_area_ratio must be in [0, 1)")
        if not 0.0 < float(self.distal_zone_fraction) < 0.5:
            raise ValueError("distal_zone_fraction must be in (0, 0.5)")
        if not 0.0 <= float(self.max_toe_gap_ratio) < 1.0:
            raise ValueError("max_toe_gap_ratio must be in [0, 1)")
        if not 0.0 <= float(self.toe_lateral_margin_ratio) < 1.0:
            raise ValueError("toe_lateral_margin_ratio must be in [0, 1)")
        if int(self.min_toe_component_area_pixels) < 0:
            raise ValueError("min_toe_component_area_pixels must be non-negative")
        if int(self.reference_image_size) <= 0:
            raise ValueError("reference_image_size must be positive")
        if self.toe_direction not in ("auto", "top", "bottom", "left", "right"):
            raise ValueError("toe_direction must be auto, top, bottom, left, or right")
        if int(self.connectivity) not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if int(self.max_hole_area_pixels) < 0:
            raise ValueError("max_hole_area_pixels must be non-negative")
        if int(self.max_hole_span_pixels) < 0:
            raise ValueError("max_hole_span_pixels must be non-negative")
        if not 0.0 <= float(self.min_hole_circularity) <= 1.0:
            raise ValueError("min_hole_circularity must be in [0, 1]")
        if int(self.small_hole_circularity_exemption_pixels) < 0:
            raise ValueError(
                "small_hole_circularity_exemption_pixels must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSamCleanupResult:
    """Binary masks and JSON-safe audit produced by :func:`cleanup_full_sam_mask`."""

    original_mask: np.ndarray
    support_clipped_mask: np.ndarray
    cleaned_mask: np.ndarray
    removed_component_mask: np.ndarray
    removed_outside_support_mask: np.ndarray
    filled_hole_mask: np.ndarray
    audit: Mapping[str, Any]


def _binary_mask(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    if np.issubdtype(array.dtype, np.floating) and not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    if array.dtype == np.bool_:
        return array.copy()
    return (array > 0).astype(bool, copy=False).copy()


def _mask_sha256(mask: np.ndarray) -> str:
    """Hash mask geometry and bits, independent of the source image encoding."""

    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(binary.shape, dtype=np.int64).tobytes())
    digest.update(binary.tobytes())
    return digest.hexdigest()


def _component_table(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=int(connectivity),
    )
    if count <= 1:
        return labels, stats[:1], centroids[:1]
    return labels, stats, centroids


def _explicit_axis(direction: ToeDirection) -> np.ndarray | None:
    axes = {
        "top": np.asarray([0.0, -1.0], dtype=np.float64),
        "bottom": np.asarray([0.0, 1.0], dtype=np.float64),
        "left": np.asarray([-1.0, 0.0], dtype=np.float64),
        "right": np.asarray([1.0, 0.0], dtype=np.float64),
    }
    return axes.get(direction)


def _axis_end_width(
    points_xy: np.ndarray,
    axis: np.ndarray,
    *,
    high_end: bool,
) -> float:
    longitudinal = points_xy @ axis
    perpendicular = points_xy @ np.asarray([-axis[1], axis[0]])
    low, high = np.quantile(longitudinal, [0.02, 0.98])
    span = max(float(high - low), 1.0)
    if high_end:
        selected = perpendicular[longitudinal >= high - 0.22 * span]
    else:
        selected = perpendicular[longitudinal <= low + 0.22 * span]
    if selected.size < 2:
        return 0.0
    p05, p95 = np.quantile(selected, [0.05, 0.95])
    return float(max(p95 - p05, 0.0))


def _canonical_axis(
    anchor_mask: np.ndarray,
    toe_direction: ToeDirection,
) -> tuple[np.ndarray, str, float, float]:
    explicit = _explicit_axis(toe_direction)
    if explicit is not None:
        return explicit, toe_direction, 0.0, 0.0

    yy, xx = np.nonzero(anchor_mask)
    points = np.column_stack((xx, yy)).astype(np.float64)
    if points.shape[0] < 2:
        return np.asarray([0.0, -1.0]), "auto_fallback_top", 0.0, 0.0

    centred = points - points.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.asarray([0.0, -1.0]), "auto_fallback_top", 0.0, 0.0
    axis /= norm

    low_width = _axis_end_width(points, axis, high_end=False)
    high_width = _axis_end_width(points, axis, high_end=True)
    tolerance = 0.03 * max(low_width, high_width, 1.0)
    if low_width > high_width + tolerance:
        axis *= -1.0
        resolved = "auto_broader_low_end"
    elif high_width > low_width + tolerance:
        resolved = "auto_broader_high_end"
    else:
        # Eigenvectors have an arbitrary sign.  A stable top/left preference
        # makes tied cases deterministic without pretending to infer anatomy.
        if abs(float(axis[1])) >= abs(float(axis[0])):
            if float(axis[1]) > 0.0:
                axis *= -1.0
            resolved = "auto_tie_top"
        else:
            if float(axis[0]) > 0.0:
                axis *= -1.0
            resolved = "auto_tie_left"
    return axis, resolved, low_width, high_width


def _compact_holes(
    mask: np.ndarray,
    *,
    connectivity: int,
    max_area: int,
    max_span: int,
    min_circularity: float,
    circularity_exemption_area: int,
    forbidden_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fill = np.zeros_like(mask, dtype=bool)
    records: list[dict[str, Any]] = []
    if forbidden_mask is None:
        forbidden = np.zeros_like(mask, dtype=bool)
    else:
        forbidden = np.asarray(forbidden_mask, dtype=bool)
        if forbidden.shape != mask.shape:
            raise ValueError("forbidden_mask and mask must have identical shapes")
    if max_area <= 0 or not bool(mask.any()):
        return fill, records

    inverse = (~mask).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=int(connectivity),
    )
    height, width = mask.shape
    effective_span = int(max_span)
    if effective_span <= 0:
        effective_span = max(1, int(math.ceil(2.0 * math.sqrt(float(max_area)))))
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        compact = max(component_width, component_height) <= effective_span
        within_area = area <= int(max_area)
        component_pixels = labels == label
        intersects_forbidden = bool(forbidden[component_pixels].any())
        fillable_geometry = not touches_border and within_area and compact
        circularity: float | None = None
        if fillable_geometry:
            contours, _ = cv2.findContours(
                component_pixels.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            contour_area = sum(float(cv2.contourArea(contour)) for contour in contours)
            perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
            circularity = (
                float(4.0 * math.pi * contour_area / (perimeter * perimeter))
                if perimeter > 0.0
                else 0.0
            )
        circularity_exempt = area <= int(circularity_exemption_area)
        passes_circularity = (
            float(min_circularity) <= 0.0
            or circularity_exempt
            or (circularity is not None and circularity >= float(min_circularity))
        )
        accepted = fillable_geometry and passes_circularity and not intersects_forbidden
        if accepted:
            fill[component_pixels] = True
        if accepted:
            reason = "fill_tiny_compact_hole"
        elif fillable_geometry and intersects_forbidden:
            reason = "preserve_hole_intersects_removed_component"
        elif fillable_geometry and not passes_circularity:
            reason = "preserve_hole_below_min_circularity"
        else:
            reason = "preserve_background"
        records.append(
            {
                "hole_id": int(label),
                "area_pixels": area,
                "bbox_xywh": [x, y, component_width, component_height],
                "centroid_xy": [float(centroids[label, 0]), float(centroids[label, 1])],
                "touches_image_border": bool(touches_border),
                "within_area_limit": bool(within_area),
                "compact_within_span_limit": bool(compact),
                "circularity": circularity,
                "minimum_circularity": float(min_circularity),
                "small_hole_circularity_exempt": bool(circularity_exempt),
                "passes_circularity_guard": bool(passes_circularity),
                "intersects_removed_component": bool(intersects_forbidden),
                "filled": bool(accepted),
                "reason": reason,
            }
        )
    return fill, records


def cleanup_full_sam_mask(
    full_sam_mask: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    config: FullSamCleanupConfig | None = None,
) -> FullSamCleanupResult:
    """Remove implausible disconnected fragments from one Full SAM mask.

    The input arrays are never mutated.  When ``support_mask`` is supplied,
    foreground outside it is clipped before components are evaluated.  Except
    for explicitly enabled tiny-hole filling, output foreground is therefore
    a strict subset of ``full_sam_mask & support_mask``.
    """

    policy = config or FullSamCleanupConfig()
    original = _binary_mask(full_sam_mask, "full_sam_mask")
    original_snapshot = original.copy()
    if support_mask is None:
        support = np.ones_like(original, dtype=bool)
    else:
        support = _binary_mask(support_mask, "support_mask")
        if support.shape != original.shape:
            raise ValueError("support_mask and full_sam_mask must have identical shapes")

    clipped = original & support
    outside_support = original & ~support
    labels, stats, centroids = _component_table(clipped, policy.connectivity)
    component_count = max(int(stats.shape[0]) - 1, 0)
    total_area = int(clipped.sum())

    if component_count == 0:
        empty = np.zeros_like(original, dtype=bool)
        audit: dict[str, Any] = {
            "format": FULL_SAM_CLEANUP_FORMAT,
            "shape_hw": [int(original.shape[0]), int(original.shape[1])],
            "config": policy.to_dict(),
            "orientation": None,
            "components": [],
            "holes": [],
            "counts": {
                "input_foreground_pixels": int(original.sum()),
                "support_clipped_foreground_pixels": 0,
                "cleaned_foreground_pixels": 0,
                "removed_outside_support_pixels": int(outside_support.sum()),
                "removed_component_pixels": 0,
                "filled_hole_pixels": 0,
                "components_before": 0,
                "components_after": 0,
                "components_kept": 0,
                "components_removed": 0,
            },
            "mask_sha256": {
                "input_binary": _mask_sha256(original),
                "support_clipped_binary": _mask_sha256(empty),
                "cleaned_binary": _mask_sha256(empty),
            },
            "invariants": {
                "source_array_unchanged": bool(np.array_equal(original, original_snapshot)),
                "outside_support_pixels_after": 0,
                "no_morphological_closing_or_bridging": True,
                "only_possible_added_pixels_are_audited_tiny_holes": True,
            },
        }
        return FullSamCleanupResult(
            original_mask=original,
            support_clipped_mask=empty.copy(),
            cleaned_mask=empty.copy(),
            removed_component_mask=empty.copy(),
            removed_outside_support_mask=outside_support,
            filled_hole_mask=empty.copy(),
            audit=audit,
        )

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest_label = int(np.argmax(areas)) + 1
    anchor_threshold = max(
        1,
        int(math.ceil(float(policy.anchor_component_area_ratio) * float(total_area))),
    )
    anchor_labels = {
        label
        for label in range(1, component_count + 1)
        if label == largest_label
        or int(stats[label, cv2.CC_STAT_AREA]) >= anchor_threshold
    }
    anchor_mask = np.isin(labels, np.asarray(sorted(anchor_labels), dtype=labels.dtype))
    toe_axis, orientation_reason, low_width, high_width = _canonical_axis(
        anchor_mask,
        policy.toe_direction,
    )
    perpendicular_axis = np.asarray([-toe_axis[1], toe_axis[0]], dtype=np.float64)

    anchor_y, anchor_x = np.nonzero(anchor_mask)
    anchor_points = np.column_stack((anchor_x, anchor_y)).astype(np.float64)
    anchor_long = anchor_points @ toe_axis
    anchor_lateral = anchor_points @ perpendicular_axis
    long_min, long_max = np.quantile(anchor_long, [0.01, 0.99])
    lateral_min, lateral_max = np.quantile(anchor_lateral, [0.01, 0.99])
    long_span = max(float(long_max - long_min), 1.0)
    lateral_span = max(float(lateral_max - lateral_min), 1.0)
    distal_threshold = float(long_max) - float(policy.distal_zone_fraction) * long_span
    maximum_toe_projection = float(long_max) + float(policy.max_toe_gap_ratio) * long_span
    lateral_margin = float(policy.toe_lateral_margin_ratio) * lateral_span

    scale = max(original.shape) / float(policy.reference_image_size)
    scaled_min_toe_pixels = int(
        math.ceil(float(policy.min_toe_component_area_pixels) * scale * scale)
    )
    if support_mask is not None:
        toe_area_reference = int(support.sum())
        toe_area_reference_source = "provided_support_foreground"
    else:
        # Full SAM exports are commonly already support-clipped and therefore
        # arrive without a separate support file.  The anchors' convex hull is
        # a closer deterministic proxy than their rectangle (which can be much
        # larger than a narrow plantar support).
        hull_points = np.column_stack((anchor_x, anchor_y)).astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(hull_points)
        hull_area = int(math.ceil(float(cv2.contourArea(hull))))
        toe_area_reference = max(total_area, hull_area)
        toe_area_reference_source = "anchor_convex_hull_proxy"
    toe_threshold = max(
        1,
        scaled_min_toe_pixels,
        int(
            math.ceil(
                float(policy.toe_component_support_area_ratio)
                * float(toe_area_reference)
            )
        ),
    )

    kept_labels: set[int] = set()
    component_records: list[dict[str, Any]] = []
    for label in range(1, component_count + 1):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid = np.asarray(centroids[label], dtype=np.float64)
        longitudinal = float(centroid @ toe_axis)
        lateral = float(centroid @ perpendicular_axis)
        normalized_longitudinal = (longitudinal - float(long_min)) / long_span
        in_distal_zone = longitudinal >= distal_threshold
        within_toe_gap = longitudinal <= maximum_toe_projection
        within_lateral_envelope = (
            float(lateral_min) - lateral_margin
            <= lateral
            <= float(lateral_max) + lateral_margin
        )
        large_enough_for_toe = area >= toe_threshold

        if label == largest_label:
            keep = True
            reason = "keep_largest_anchor"
        elif label in anchor_labels:
            keep = True
            reason = "keep_large_anatomical_anchor"
        elif (
            in_distal_zone
            and within_toe_gap
            and within_lateral_envelope
            and large_enough_for_toe
        ):
            keep = True
            reason = "keep_credible_distal_toe"
        else:
            keep = False
            if not in_distal_zone:
                reason = "remove_non_distal_disconnected_component"
            elif not large_enough_for_toe:
                reason = "remove_distal_component_below_toe_area"
            elif not within_toe_gap:
                reason = "remove_distal_component_beyond_toe_gap"
            else:
                reason = "remove_distal_component_outside_lateral_envelope"
        if keep:
            kept_labels.add(label)

        component_records.append(
            {
                "component_id": int(label),
                "area_pixels": area,
                "area_ratio_of_support_clipped_foreground": float(area / total_area),
                "bbox_xywh": [x, y, width, height],
                "centroid_xy": [float(centroid[0]), float(centroid[1])],
                "normalized_longitudinal_centroid": float(normalized_longitudinal),
                "in_distal_zone": bool(in_distal_zone),
                "within_toe_gap": bool(within_toe_gap),
                "within_lateral_envelope": bool(within_lateral_envelope),
                "meets_anchor_area_threshold": bool(area >= anchor_threshold),
                "meets_toe_area_threshold": bool(large_enough_for_toe),
                "decision": "keep" if keep else "remove",
                "reason": reason,
            }
        )

    keep_mask = np.isin(labels, np.asarray(sorted(kept_labels), dtype=labels.dtype))
    removed_components = clipped & ~keep_mask

    # Evaluate holes on the retained anatomical components, not on the raw
    # proposal.  Otherwise a hole enclosed by a component that was just
    # rejected could be filled after its surrounding component disappeared,
    # manufacturing a new isolated foreground speck.
    filled_holes, hole_records = _compact_holes(
        keep_mask,
        connectivity=policy.connectivity,
        max_area=int(policy.max_hole_area_pixels),
        max_span=int(policy.max_hole_span_pixels),
        min_circularity=float(policy.min_hole_circularity),
        circularity_exemption_area=int(
            policy.small_hole_circularity_exemption_pixels
        ),
        forbidden_mask=removed_components,
    )
    # Only pre-existing compact holes are eligible, and support clipping is
    # re-applied after filling.  No dilation/closing operation is used.
    filled_holes &= support
    cleaned = (keep_mask | filled_holes) & support

    after_count, _, _, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8),
        connectivity=int(policy.connectivity),
    )
    added_not_holes = cleaned & ~clipped & ~filled_holes
    audit = {
        "format": FULL_SAM_CLEANUP_FORMAT,
        "shape_hw": [int(original.shape[0]), int(original.shape[1])],
        "config": policy.to_dict(),
        "orientation": {
            "requested_toe_direction": policy.toe_direction,
            "resolved_by": orientation_reason,
            "toe_axis_xy": [float(toe_axis[0]), float(toe_axis[1])],
            "auto_low_end_width": float(low_width),
            "auto_high_end_width": float(high_width),
            "anchor_longitudinal_range": [float(long_min), float(long_max)],
            "anchor_lateral_range": [float(lateral_min), float(lateral_max)],
            "distal_threshold": float(distal_threshold),
            "maximum_toe_projection": float(maximum_toe_projection),
        },
        "thresholds_pixels": {
            "anchor_component_area": int(anchor_threshold),
            "credible_toe_component_area": int(toe_threshold),
            "credible_toe_area_reference": int(toe_area_reference),
            "credible_toe_area_reference_source": toe_area_reference_source,
            "maximum_tiny_hole_area": int(policy.max_hole_area_pixels),
            "minimum_hole_circularity": float(policy.min_hole_circularity),
            "small_hole_circularity_exemption_area": int(
                policy.small_hole_circularity_exemption_pixels
            ),
        },
        "components": component_records,
        "holes": hole_records,
        "counts": {
            "input_foreground_pixels": int(original.sum()),
            "support_clipped_foreground_pixels": total_area,
            "cleaned_foreground_pixels": int(cleaned.sum()),
            "removed_outside_support_pixels": int(outside_support.sum()),
            "removed_component_pixels": int(removed_components.sum()),
            "filled_hole_pixels": int(filled_holes.sum()),
            "components_before": int(component_count),
            "components_after": int(after_count - 1),
            "components_kept": int(len(kept_labels)),
            "components_removed": int(component_count - len(kept_labels)),
        },
        "mask_sha256": {
            "input_binary": _mask_sha256(original),
            "support_clipped_binary": _mask_sha256(clipped),
            "cleaned_binary": _mask_sha256(cleaned),
        },
        "invariants": {
            "source_array_unchanged": bool(np.array_equal(original, original_snapshot)),
            "outside_support_pixels_after": int((cleaned & ~support).sum()),
            "removed_component_pixels_after": int((cleaned & removed_components).sum()),
            "unexpected_added_pixels": int(added_not_holes.sum()),
            "no_morphological_closing_or_bridging": True,
            "no_removed_component_pixels_restored": bool(
                not bool((cleaned & removed_components).any())
            ),
            "only_possible_added_pixels_are_audited_tiny_holes": bool(
                not bool(added_not_holes.any())
            ),
        },
    }
    return FullSamCleanupResult(
        original_mask=original,
        support_clipped_mask=clipped,
        cleaned_mask=cleaned,
        removed_component_mask=removed_components,
        removed_outside_support_mask=outside_support,
        filled_hole_mask=filled_holes,
        audit=audit,
    )


__all__ = [
    "FULL_SAM_CLEANUP_FORMAT",
    "FullSamCleanupConfig",
    "FullSamCleanupResult",
    "ToeDirection",
    "cleanup_full_sam_mask",
]
