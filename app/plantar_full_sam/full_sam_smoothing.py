"""Conservative, topology-checked boundary smoothing for Full SAM masks.

This module is intended to run *after* disconnected-component cleanup and any
explicit compact-hole filling.  It applies a very small Gaussian soft-mask
filter to each original foreground component independently.  Existing hole
pixels are protected before each proposal is checked.  A component proposal
that would split, create a hole, or open a hole is replaced by that original
component while safe proposals remain usable.  The composed mask is published
only when component identity, foreground connectivity, enclosed holes, and
border-connected background topology all still correspond one to one with the
input.  A composed-mask violation fails closed to an exact input copy.

The implementation is deterministic and local.  It does not call a model or
an external API, and it never performs a morphological close or bridge.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


FULL_SAM_SMOOTHING_FORMAT = "full_sam_topology_preserving_smoothing_v1"


@dataclass(frozen=True)
class FullSamSmoothingConfig:
    """Scale-aware defaults for a slight Full SAM contour cleanup.

    At the common 700--1024 px export sizes, the default effective Gaussian
    sigma is 1.05--1.25 px.  This is enough to soften one-pixel stair steps and
    isolated contour teeth without intentionally changing anatomical extent.
    Components smaller than the scale-adjusted area floor are copied verbatim;
    tiny toe contacts therefore cannot disappear merely to make an edge look
    smoother.

    The two global ratios and the per-component changed-pixel ratio are safety
    budgets, not targets.  A per-component excess copies that component; a
    composed-mask excess returns the complete original mask.
    """

    reference_image_size: int = 1024
    sigma_pixels_at_reference: float = 1.25
    minimum_sigma_pixels: float = 1.05
    maximum_sigma_pixels: float = 1.35
    gaussian_truncate: float = 3.0
    soft_threshold: float = 0.5
    minimum_component_area_at_reference: int = 32
    max_changed_pixel_ratio: float = 0.08
    max_area_delta_ratio: float = 0.025
    max_component_changed_pixel_ratio: float = 0.25
    connectivity: int = 8

    def __post_init__(self) -> None:
        if int(self.reference_image_size) <= 0:
            raise ValueError("reference_image_size must be positive")
        if float(self.sigma_pixels_at_reference) <= 0.0:
            raise ValueError("sigma_pixels_at_reference must be positive")
        if float(self.minimum_sigma_pixels) <= 0.0:
            raise ValueError("minimum_sigma_pixels must be positive")
        if float(self.maximum_sigma_pixels) < float(self.minimum_sigma_pixels):
            raise ValueError(
                "maximum_sigma_pixels must be greater than or equal to "
                "minimum_sigma_pixels"
            )
        if float(self.gaussian_truncate) <= 0.0:
            raise ValueError("gaussian_truncate must be positive")
        if not 0.0 < float(self.soft_threshold) < 1.0:
            raise ValueError("soft_threshold must be in (0, 1)")
        if int(self.minimum_component_area_at_reference) < 0:
            raise ValueError(
                "minimum_component_area_at_reference must be non-negative"
            )
        for name in (
            "max_changed_pixel_ratio",
            "max_area_delta_ratio",
            "max_component_changed_pixel_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(self.connectivity) not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSamSmoothingResult:
    """Binary masks and JSON-safe audit for one smoothing attempt.

    ``candidate_mask`` is the union of accepted smoothed components and
    component-local fallbacks.  Consumers should always use ``smoothed_mask``;
    it is the globally checked candidate, or an exact input copy when
    ``audit[\"fail_closed\"]`` is true.  Rejected raw component proposals are
    described in the per-component audit rather than exposed as an output.
    """

    original_mask: np.ndarray
    candidate_mask: np.ndarray
    smoothed_mask: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray
    changed_mask: np.ndarray
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
    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(binary.shape, dtype=np.int64).tobytes())
    digest.update(binary.tobytes())
    return digest.hexdigest()


def _connected_components(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8),
        connectivity=int(connectivity),
    )
    return int(count), labels, stats, centroids


def _background_regions(
    mask: np.ndarray,
    foreground_connectivity: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], dict[int, int]]:
    """Return inverse labels, enclosed holes, border regions, and their areas.

    Foreground and background use dual digital connectivities (8/4 or 4/8),
    preventing a diagonal contact from simultaneously connecting both sets.
    """

    background_connectivity = 4 if int(foreground_connectivity) == 8 else 8
    count, labels, stats, _ = _connected_components(
        ~mask,
        background_connectivity,
    )
    if count <= 1:
        return labels, (), (), {}

    border_values = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    border_ids = tuple(
        int(value) for value in np.unique(border_values) if int(value) > 0
    )
    border_set = set(border_ids)
    hole_ids = tuple(label for label in range(1, count) if label not in border_set)
    areas = {
        int(label): int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
    }
    return labels, hole_ids, border_ids, areas


def _label_correspondence(
    before_labels: np.ndarray,
    before_ids: tuple[int, ...] | list[int],
    after_labels: np.ndarray,
    after_ids: tuple[int, ...] | list[int],
) -> bool:
    """Require a positive-overlap bijection between two labelled region sets."""

    before = tuple(int(value) for value in before_ids)
    after = tuple(int(value) for value in after_ids)
    if len(before) != len(after):
        return False
    if not before:
        return True

    after_set = set(after)
    forward: dict[int, int] = {}
    for before_id in before:
        overlaps = {
            int(value)
            for value in np.unique(after_labels[before_labels == before_id])
            if int(value) in after_set
        }
        if len(overlaps) != 1:
            return False
        forward[before_id] = next(iter(overlaps))
    return len(set(forward.values())) == len(after)


def _common_padded_rois(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    union = before | after
    yy, xx = np.nonzero(union)
    if yy.size == 0:
        empty = np.zeros((3, 3), dtype=bool)
        return empty, empty.copy()
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    before_roi = np.pad(before[y0:y1, x0:x1], 1, constant_values=False)
    after_roi = np.pad(after[y0:y1, x0:x1], 1, constant_values=False)
    return before_roi, after_roi


def _component_hole_audit(
    before: np.ndarray,
    after: np.ndarray,
    connectivity: int,
) -> dict[str, Any]:
    before_roi, after_roi = _common_padded_rois(before, after)
    before_labels, before_holes, _, before_areas = _background_regions(
        before_roi,
        connectivity,
    )
    after_labels, after_holes, _, after_areas = _background_regions(
        after_roi,
        connectivity,
    )
    correspondence = _label_correspondence(
        before_labels,
        before_holes,
        after_labels,
        after_holes,
    )
    return {
        "holes_before": int(len(before_holes)),
        "holes_candidate": int(len(after_holes)),
        "hole_areas_before": [int(before_areas[label]) for label in before_holes],
        "hole_areas_candidate": [int(after_areas[label]) for label in after_holes],
        "hole_correspondence_preserved": bool(correspondence),
    }


def _enclosed_hole_mask(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Return every input pixel belonging to an enclosed background region."""

    labels, hole_ids, _, _ = _background_regions(mask, connectivity)
    if not hole_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, np.asarray(hole_ids, dtype=labels.dtype))


def _foreground_correspondence(
    before: np.ndarray,
    after: np.ndarray,
    connectivity: int,
) -> bool:
    before_count, before_labels, _, _ = _connected_components(before, connectivity)
    after_count, after_labels, _, _ = _connected_components(after, connectivity)
    return _label_correspondence(
        before_labels,
        list(range(1, before_count)),
        after_labels,
        list(range(1, after_count)),
    )


def _topology_summary(mask: np.ndarray, connectivity: int) -> dict[str, Any]:
    foreground_count, _, foreground_stats, _ = _connected_components(
        mask,
        connectivity,
    )
    _, hole_ids, border_ids, background_areas = _background_regions(
        mask,
        connectivity,
    )
    foreground_components = max(foreground_count - 1, 0)
    return {
        "foreground_connectivity": int(connectivity),
        "background_connectivity": 4 if int(connectivity) == 8 else 8,
        "foreground_components": int(foreground_components),
        "foreground_component_areas": [
            int(foreground_stats[label, cv2.CC_STAT_AREA])
            for label in range(1, foreground_count)
        ],
        "enclosed_holes": int(len(hole_ids)),
        "enclosed_hole_areas": [int(background_areas[label]) for label in hole_ids],
        "border_background_components": int(len(border_ids)),
        "border_background_areas": [
            int(background_areas[label]) for label in border_ids
        ],
        "euler_characteristic": int(foreground_components - len(hole_ids)),
    }


def _global_topology_preserved(
    before: np.ndarray,
    after: np.ndarray,
    connectivity: int,
) -> tuple[bool, dict[str, bool]]:
    before_count, before_fg_labels, _, _ = _connected_components(before, connectivity)
    after_count, after_fg_labels, _, _ = _connected_components(after, connectivity)
    foreground_ok = _label_correspondence(
        before_fg_labels,
        list(range(1, before_count)),
        after_fg_labels,
        list(range(1, after_count)),
    )

    before_bg_labels, before_holes, before_border, _ = _background_regions(
        before,
        connectivity,
    )
    after_bg_labels, after_holes, after_border, _ = _background_regions(
        after,
        connectivity,
    )
    holes_ok = _label_correspondence(
        before_bg_labels,
        before_holes,
        after_bg_labels,
        after_holes,
    )
    border_ok = _label_correspondence(
        before_bg_labels,
        before_border,
        after_bg_labels,
        after_border,
    )
    checks = {
        "foreground_component_correspondence": bool(foreground_ok),
        "enclosed_hole_correspondence": bool(holes_ok),
        "border_background_correspondence": bool(border_ok),
    }
    return bool(all(checks.values())), checks


def _edit_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int | float]:
    added = after & ~before
    removed = before & ~after
    before_area = int(before.sum())
    after_area = int(after.sum())
    denominator = max(before_area, 1)
    changed = int(added.sum() + removed.sum())
    delta = int(after_area - before_area)
    return {
        "input_foreground_pixels": before_area,
        "output_foreground_pixels": after_area,
        "added_pixels": int(added.sum()),
        "removed_pixels": int(removed.sum()),
        "changed_pixels": changed,
        "changed_pixel_ratio_of_input_foreground": float(changed / denominator),
        "area_delta_pixels": delta,
        "area_delta_ratio_of_input_foreground": float(delta / denominator),
        "absolute_area_delta_ratio_of_input_foreground": float(
            abs(delta) / denominator
        ),
    }


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def smooth_full_sam_mask(
    full_sam_mask: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    config: FullSamSmoothingConfig | None = None,
) -> FullSamSmoothingResult:
    """Slightly smooth a binary Full SAM mask with topology fail-close.

    The input arrays are never mutated.  When ``support_mask`` is supplied it
    must already contain all input foreground; it is then a hard bound on any
    candidate expansion.  Run component cleanup/support clipping before this
    function rather than relying on smoothing to perform either operation.
    """

    policy = config or FullSamSmoothingConfig()
    source_array = np.asarray(full_sam_mask)
    source_snapshot = source_array.copy()
    original = _binary_mask(full_sam_mask, "full_sam_mask")
    if support_mask is None:
        support = np.ones_like(original, dtype=bool)
    else:
        support = _binary_mask(support_mask, "support_mask")
        if support.shape != original.shape:
            raise ValueError("support_mask and full_sam_mask must have identical shapes")
        if bool((original & ~support).any()):
            raise ValueError(
                "support_mask must contain all input foreground; clip the Full SAM "
                "mask before smoothing"
            )

    height, width = original.shape
    scale = max(height, width) / float(policy.reference_image_size)
    effective_sigma = float(
        np.clip(
            float(policy.sigma_pixels_at_reference) * scale,
            float(policy.minimum_sigma_pixels),
            float(policy.maximum_sigma_pixels),
        )
    )
    radius = max(1, int(math.ceil(float(policy.gaussian_truncate) * effective_sigma)))
    kernel_size = 2 * radius + 1
    minimum_component_area = int(
        math.ceil(
            float(policy.minimum_component_area_at_reference) * scale * scale
        )
    )

    component_count, labels, stats, _ = _connected_components(
        original,
        policy.connectivity,
    )
    candidate = np.zeros_like(original, dtype=bool)
    candidate_owner = np.zeros_like(labels, dtype=np.int32)
    component_records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width, x + component_width + radius)
        y1 = min(height, y + component_height + radius)

        protected_holes = _enclosed_hole_mask(component, policy.connectivity)
        if area < minimum_component_area:
            component_proposal = component.copy()
            restored_hole_pixels = 0
        else:
            soft = cv2.GaussianBlur(
                component[y0:y1, x0:x1].astype(np.float32),
                (kernel_size, kernel_size),
                sigmaX=effective_sigma,
                sigmaY=effective_sigma,
                borderType=cv2.BORDER_CONSTANT,
            )
            candidate_roi = soft >= float(policy.soft_threshold)
            candidate_roi &= support[y0:y1, x0:x1]
            component_proposal = np.zeros_like(original, dtype=bool)
            component_proposal[y0:y1, x0:x1] = candidate_roi
            # Hole filling belongs to the preceding cleanup stage.  Gaussian
            # smoothing may soften a hole boundary, but may not consume any
            # pixel that cleanup deliberately left as enclosed background.
            restored_hole_pixels = int((component_proposal & protected_holes).sum())
            component_proposal[protected_holes] = False

        local_foreground_count, _, _, _ = _connected_components(
            component_proposal,
            policy.connectivity,
        )
        local_foreground_ok = int(local_foreground_count - 1) == 1
        hole_audit = _component_hole_audit(
            component,
            component_proposal,
            policy.connectivity,
        )
        local_topology_ok = bool(
            local_foreground_ok and hole_audit["hole_correspondence_preserved"]
        )
        proposal_edits = _edit_counts(component, component_proposal)
        changed_ratio_ok = (
            float(proposal_edits["changed_pixel_ratio_of_input_foreground"])
            <= float(policy.max_component_changed_pixel_ratio)
        )
        component_fallback_reasons: list[str] = []
        if not local_foreground_ok:
            _append_reason(
                component_fallback_reasons,
                "foreground_connectivity_changed",
            )
        if not bool(hole_audit["hole_correspondence_preserved"]):
            _append_reason(
                component_fallback_reasons,
                "enclosed_hole_topology_changed",
            )
        if not changed_ratio_ok:
            _append_reason(
                component_fallback_reasons,
                "changed_pixel_ratio_exceeded",
            )

        if area < minimum_component_area:
            component_selected = component.copy()
            processing_status = "copied_below_scale_aware_area_floor"
        elif not local_topology_ok:
            component_selected = component.copy()
            processing_status = "copied_topology_fallback"
        elif not changed_ratio_ok:
            component_selected = component.copy()
            processing_status = "copied_edit_budget_fallback"
        else:
            component_selected = component_proposal
            processing_status = (
                "accepted_smoothed"
                if int(proposal_edits["changed_pixels"]) > 0
                else "accepted_no_change"
            )
        selected_edits = _edit_counts(component, component_selected)

        overlap = component_selected & (candidate_owner > 0)
        if bool(overlap.any()):
            _append_reason(failure_reasons, "candidate_component_pixel_overlap")
        candidate_owner[component_selected & (candidate_owner == 0)] = int(label)
        candidate |= component_selected

        component_records.append(
            {
                "component_id": int(label),
                "bbox_xywh_before": [x, y, component_width, component_height],
                "processing_status": processing_status,
                "area_floor_pixels": int(minimum_component_area),
                "accepted_smoothed": processing_status == "accepted_smoothed",
                "copied_topology_fallback": (
                    processing_status == "copied_topology_fallback"
                ),
                "smoothed_at_selected_scale": (
                    processing_status == "accepted_smoothed"
                ),
                "component_fallback_reasons": component_fallback_reasons,
                "protected_existing_hole_pixels": int(protected_holes.sum()),
                "restored_existing_hole_pixels": int(restored_hole_pixels),
                "foreground_components_before": 1,
                "foreground_components_candidate": int(local_foreground_count - 1),
                "foreground_connectivity_preserved": bool(local_foreground_ok),
                **hole_audit,
                "proposal_edit_counts": proposal_edits,
                **selected_edits,
                "changed_pixel_ratio_within_limit": bool(changed_ratio_ok),
                "proposal_topology_preserved": bool(local_topology_ok),
                "topology_preserved": True,
            }
        )

    candidate &= support
    before_topology = _topology_summary(original, policy.connectivity)
    candidate_topology = _topology_summary(candidate, policy.connectivity)
    global_topology_ok, topology_checks = _global_topology_preserved(
        original,
        candidate,
        policy.connectivity,
    )
    if not topology_checks["foreground_component_correspondence"]:
        _append_reason(
            failure_reasons,
            "global_foreground_component_merge_split_or_identity_change",
        )
    if not topology_checks["enclosed_hole_correspondence"]:
        _append_reason(failure_reasons, "global_enclosed_hole_topology_changed")
    if not topology_checks["border_background_correspondence"]:
        _append_reason(failure_reasons, "global_border_background_topology_changed")

    candidate_edits = _edit_counts(original, candidate)
    if (
        float(candidate_edits["changed_pixel_ratio_of_input_foreground"])
        > float(policy.max_changed_pixel_ratio)
    ):
        _append_reason(failure_reasons, "global_changed_pixel_ratio_exceeded")
    if (
        float(candidate_edits["absolute_area_delta_ratio_of_input_foreground"])
        > float(policy.max_area_delta_ratio)
    ):
        _append_reason(failure_reasons, "global_area_delta_ratio_exceeded")
    if bool((candidate & ~support).any()):
        _append_reason(failure_reasons, "candidate_outside_support")

    fail_closed = bool(failure_reasons)
    output = original.copy() if fail_closed else candidate.copy()
    added = output & ~original
    removed = original & ~output
    changed = added | removed
    output_topology = _topology_summary(output, policy.connectivity)
    output_topology_ok, output_topology_checks = _global_topology_preserved(
        original,
        output,
        policy.connectivity,
    )
    output_edits = _edit_counts(original, output)

    audit: dict[str, Any] = {
        "format": FULL_SAM_SMOOTHING_FORMAT,
        "shape_hw": [int(height), int(width)],
        "config": policy.to_dict(),
        "effective_parameters": {
            "image_scale_relative_to_reference": float(scale),
            "sigma_pixels": float(effective_sigma),
            "gaussian_radius_pixels": int(radius),
            "gaussian_kernel_size": int(kernel_size),
            "minimum_component_area_pixels": int(minimum_component_area),
            "soft_threshold": float(policy.soft_threshold),
        },
        "components": component_records,
        "component_decisions": {
            "accepted_smoothed": int(
                sum(row["processing_status"] == "accepted_smoothed" for row in component_records)
            ),
            "accepted_no_change": int(
                sum(row["processing_status"] == "accepted_no_change" for row in component_records)
            ),
            "copied_topology_fallback": int(
                sum(
                    row["processing_status"] == "copied_topology_fallback"
                    for row in component_records
                )
            ),
            "copied_edit_budget_fallback": int(
                sum(
                    row["processing_status"] == "copied_edit_budget_fallback"
                    for row in component_records
                )
            ),
            "copied_below_scale_aware_area_floor": int(
                sum(
                    row["processing_status"]
                    == "copied_below_scale_aware_area_floor"
                    for row in component_records
                )
            ),
        },
        "counts": {
            "candidate": candidate_edits,
            "output": output_edits,
        },
        "topology": {
            "before": before_topology,
            "candidate": candidate_topology,
            "output": output_topology,
            "candidate_checks": topology_checks,
            "candidate_preserved": bool(global_topology_ok),
            "output_checks": output_topology_checks,
            "output_preserved": bool(output_topology_ok),
        },
        "mask_sha256": {
            "input_binary": _mask_sha256(original),
            "candidate_binary": _mask_sha256(candidate),
            "output_binary": _mask_sha256(output),
        },
        "smoothing_applied": bool(changed.any()),
        "fail_closed": bool(fail_closed),
        "failure_reasons": failure_reasons,
        "api_used": False,
        "invariants": {
            "source_array_unchanged": bool(
                np.array_equal(source_array, source_snapshot, equal_nan=True)
            ),
            "components_processed_independently": True,
            "existing_enclosed_hole_pixels_are_protected": True,
            "unsafe_component_proposals_are_copied_from_input": True,
            "no_morphological_closing_or_bridging": True,
            "output_outside_support_pixels": int((output & ~support).sum()),
            "output_topology_matches_input": bool(output_topology_ok),
            "fail_close_returns_exact_input": bool(
                not fail_closed or np.array_equal(output, original)
            ),
        },
    }
    return FullSamSmoothingResult(
        original_mask=original,
        candidate_mask=candidate,
        smoothed_mask=output,
        added_mask=added,
        removed_mask=removed,
        changed_mask=changed,
        audit=audit,
    )


__all__ = [
    "FULL_SAM_SMOOTHING_FORMAT",
    "FullSamSmoothingConfig",
    "FullSamSmoothingResult",
    "smooth_full_sam_mask",
]
