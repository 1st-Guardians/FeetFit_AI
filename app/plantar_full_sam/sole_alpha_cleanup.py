"""Conservative local cleanup for a canonical YOLO plantar alpha mask.

The sole must already be vertical with toes at the top and heel at the bottom.
This module never invents foreground and never edits the upper 76 percent of
the sole.  It only suppresses lower-end pixels that are implausibly far from
the final local Full-SAM contact evidence.  A top-contour metric is reported
for fail-closed routing, but a plateau alarm does not itself cut toes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SoleAlphaCleanupConfig:
    """Scale-normalized thresholds for :func:`cleanup_sole_alpha`."""

    lower_zone_fraction: float = 0.24
    max_contact_distance_fraction: float = 0.10
    top_measurement_fraction: float = 0.20
    polygon_epsilon_fraction: float = 0.002
    top_plateau_alarm_threshold: float = 0.35
    max_removed_area_ratio: float = 0.18
    minimum_main_component_retention: float = 0.80

    def __post_init__(self) -> None:
        for name in (
            "lower_zone_fraction",
            "max_contact_distance_fraction",
            "top_measurement_fraction",
            "polygon_epsilon_fraction",
            "top_plateau_alarm_threshold",
            "max_removed_area_ratio",
            "minimum_main_component_retention",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")


@dataclass(frozen=True)
class SoleAlphaCleanupResult:
    mask: np.ndarray
    audit: dict[str, Any]


def _binary(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got {array.shape}")
    return np.ascontiguousarray(array > 0)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("sole_mask must not be empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _component_stats(mask: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0, 0, labels, stats
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = 1 + int(np.argmax(areas))
    return count - 1, main_label, labels, stats


def top_plateau_ratio(
    sole_mask: np.ndarray,
    *,
    top_fraction: float = 0.20,
    polygon_epsilon_fraction: float = 0.002,
) -> float:
    """Return the longest straight upper-contour run divided by sole width.

    A normal toe outline is a sequence of short curved lobes.  A foreign flap
    merged into YOLO often creates one long polygon segment.  The metric is a
    QC signal only; cleanup never removes upper-foot pixels from this value.
    """

    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if not 0.0 < float(polygon_epsilon_fraction) < 1.0:
        raise ValueError("polygon_epsilon_fraction must be in (0, 1)")
    binary = _binary(sole_mask, name="sole_mask")
    x1, y1, x2, y2 = _bbox(binary)
    tight = binary[y1:y2, x1:x2]
    contours, _ = cv2.findContours(
        tight.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return 0.0
    polygon = cv2.approxPolyDP(
        contour,
        float(polygon_epsilon_fraction) * perimeter,
        True,
    ).reshape(-1, 2)
    if len(polygon) < 2:
        return 0.0
    height, width = tight.shape
    limit = float(top_fraction) * height
    longest = 0.0
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if max(float(first[1]), float(second[1])) <= limit:
            longest = max(longest, float(np.linalg.norm(second - first)))
    return float(longest / max(1, width))


def cleanup_sole_alpha(
    sole_mask: np.ndarray,
    contact_mask: np.ndarray,
    config: SoleAlphaCleanupConfig | None = None,
) -> SoleAlphaCleanupResult:
    """Suppress an unsupported lower lobe and return a complete audit.

    The maximum permitted distance decreases continuously from infinity at the
    upper edge of the lower 24% zone to ``0.10 * tight_bbox_width`` at the
    bottom.  This avoids a horizontal cut.  The output is always a subset of
    YOLO, all contact pixels are retained, and only the largest component plus
    components containing contact are published.  Any unsafe area/connectivity
    change falls back atomically to the original mask.
    """

    cfg = config or SoleAlphaCleanupConfig()
    sole = _binary(sole_mask, name="sole_mask")
    contact = _binary(contact_mask, name="contact_mask")
    if sole.shape != contact.shape:
        raise ValueError(
            f"sole_mask and contact_mask shapes differ: {sole.shape} != {contact.shape}"
        )
    outside_contact = contact & ~sole
    if bool(outside_contact.any()):
        raise ValueError("contact_mask must be a subset of sole_mask")

    x1, y1, x2, y2 = _bbox(sole)
    width, height = x2 - x1, y2 - y1
    plateau = top_plateau_ratio(
        sole,
        top_fraction=cfg.top_measurement_fraction,
        polygon_epsilon_fraction=cfg.polygon_epsilon_fraction,
    )
    base_audit: dict[str, Any] = {
        "policy": "lower_contact_distance_v1",
        "config": asdict(cfg),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width": int(width),
        "bbox_height": int(height),
        "input_pixels": int(sole.sum()),
        "contact_pixels": int(contact.sum()),
        "top_plateau_ratio": float(plateau),
        "top_plateau_alarm": bool(plateau > cfg.top_plateau_alarm_threshold),
        "upper_pixels_modified": 0,
    }
    if not bool(contact.any()):
        return SoleAlphaCleanupResult(
            mask=sole.copy(),
            audit={
                **base_audit,
                "applied": False,
                "fallback": True,
                "reason": "empty_contact_mask",
                "output_pixels": int(sole.sum()),
                "removed_pixels": 0,
                "removed_area_ratio": 0.0,
            },
        )

    distance = cv2.distanceTransform(
        (~contact).astype(np.uint8), cv2.DIST_L2, 5
    )
    lower_start = y1 + (1.0 - cfg.lower_zone_fraction) * height
    rows = np.arange(sole.shape[0], dtype=np.float32)[:, None]
    progress = np.clip(
        (rows - lower_start) / max(float(cfg.lower_zone_fraction * height), 1.0),
        0.0,
        1.0,
    )
    maximum_bottom_distance = float(cfg.max_contact_distance_fraction * width)
    allowed_distance = maximum_bottom_distance / np.maximum(progress, 1e-6)
    candidate = sole & ((progress <= 0.0) | (distance <= allowed_distance))
    candidate |= contact

    # Drop only disconnected debris created by the lower suppression.  The
    # largest sole component and every contact-bearing component are retained.
    component_count, main_label, labels, stats = _component_stats(candidate)
    if component_count:
        keep_labels = {int(main_label)}
        for label in range(1, component_count + 1):
            if bool(np.any(contact & (labels == label))):
                keep_labels.add(label)
        candidate = np.isin(labels, list(keep_labels)) | contact

    original_count, original_main_label, original_labels, original_stats = _component_stats(sole)
    original_main_area = (
        int(original_stats[original_main_label, cv2.CC_STAT_AREA])
        if original_main_label
        else 0
    )
    retained_main = int(np.count_nonzero(candidate & (original_labels == original_main_label)))
    main_retention = float(retained_main / max(1, original_main_area))
    removed = sole & ~candidate
    removed_ratio = float(removed.sum() / max(1, sole.sum()))
    upper_limit = int(np.floor(lower_start))
    upper_modified = int(np.count_nonzero(removed[:upper_limit]))
    candidate_count, _, _, _ = _component_stats(candidate)

    fallback_reason: str | None = None
    if upper_modified:
        fallback_reason = "upper_zone_would_change"
    elif not bool(np.all(candidate[contact])):
        fallback_reason = "contact_would_be_lost"
    elif removed_ratio > cfg.max_removed_area_ratio:
        fallback_reason = "removed_area_ratio_above_limit"
    elif main_retention < cfg.minimum_main_component_retention:
        fallback_reason = "main_component_retention_below_limit"
    elif not bool(candidate.any()):
        fallback_reason = "candidate_empty"

    applied = fallback_reason is None and bool(removed.any())
    output = sole.copy() if fallback_reason is not None else candidate
    audit = {
        **base_audit,
        "applied": bool(applied),
        "fallback": bool(fallback_reason is not None),
        "reason": fallback_reason or ("lower_lobe_suppressed" if applied else "no_change"),
        "lower_zone_start_row": float(lower_start),
        "maximum_bottom_contact_distance_pixels": float(maximum_bottom_distance),
        "input_component_count": int(original_count),
        "candidate_component_count": int(candidate_count),
        "main_component_retention": float(main_retention),
        "candidate_removed_pixels": int(removed.sum()),
        "candidate_removed_area_ratio": float(removed_ratio),
        "upper_pixels_modified": int(upper_modified),
        "output_pixels": int(output.sum()),
        "removed_pixels": int(np.count_nonzero(sole & ~output)),
        "removed_area_ratio": float(np.count_nonzero(sole & ~output) / max(1, sole.sum())),
        "contact_pixels_preserved": int(np.count_nonzero(output & contact)),
    }
    return SoleAlphaCleanupResult(mask=np.ascontiguousarray(output), audit=audit)


__all__ = [
    "SoleAlphaCleanupConfig",
    "SoleAlphaCleanupResult",
    "cleanup_sole_alpha",
    "top_plateau_ratio",
]
