"""Local SAM refinement for a transparent complete-sole alpha matte.

The YOLO sole is treated as a hard maximum support.  SAM can remove an
attached background/limb lobe but can never add pixels beyond that support.
Pressure contact is used only for deterministic foreground prompts and for a
fail-safe retention check; it is not published as the alpha matte.

No network service, GPT, or OpenAI client is imported by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Sequence

import cv2
import numpy as np

from .sam_refiner import _candidate_to_predictor_mask_input


@dataclass(frozen=True)
class SoleAlphaRefinementConfig:
    """Fixed, auditable thresholds for complete-sole refinement."""

    top_probe_fraction: float = 0.015
    top_plateau_threshold: float = 0.28
    bottom_band_fraction: float = 0.24
    bottom_contact_distance_ratio: float = 0.10
    minimum_contact_retention: float = 1.0
    minimum_area_ratio: float = 0.80
    minimum_sam_predicted_iou: float = 0.75
    dense_prompt_logit_scale: float = 6.0
    minimum_bottom_outlier_ratio: float = 0.001
    minimum_component_ratio: float = 0.001

    def __post_init__(self) -> None:
        unit_interval = {
            "top_probe_fraction": self.top_probe_fraction,
            "top_plateau_threshold": self.top_plateau_threshold,
            "bottom_band_fraction": self.bottom_band_fraction,
            "bottom_contact_distance_ratio": self.bottom_contact_distance_ratio,
            "minimum_contact_retention": self.minimum_contact_retention,
            "minimum_area_ratio": self.minimum_area_ratio,
            "minimum_sam_predicted_iou": self.minimum_sam_predicted_iou,
            "minimum_bottom_outlier_ratio": self.minimum_bottom_outlier_ratio,
            "minimum_component_ratio": self.minimum_component_ratio,
        }
        for name, value in unit_interval.items():
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not np.isfinite(self.dense_prompt_logit_scale) or self.dense_prompt_logit_scale <= 0:
            raise ValueError("dense_prompt_logit_scale must be positive and finite")
        if float(self.minimum_contact_retention) != 1.0:
            raise ValueError(
                "minimum_contact_retention is a complete-sole invariant and must equal 1.0"
            )


@dataclass(frozen=True)
class SoleAlphaRefinementResult:
    mask: np.ndarray
    selected_variant: str
    fallback_used: bool
    audit: dict[str, Any]


def _binary(values: np.ndarray, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.shape}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array > 0)


def _rgb_uint8(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"image_rgb must be a non-empty HxWx3 array, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        if maximum <= 1.0:
            image = image * 255.0
    if not bool(np.isfinite(image).all()):
        raise ValueError("image_rgb must contain finite values")
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("yolo_sole_mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _preserve_complete_components(
    mask: np.ndarray,
    contact: np.ndarray,
    *,
    minimum_component_pixels: int,
) -> np.ndarray:
    """Drop SAM speckles without discarding a detached real toe/contact island."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = max(
        range(1, count),
        key=lambda item: (int(stats[item, cv2.CC_STAT_AREA]), -int(item)),
    )
    output = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        component = labels == label
        keep = bool(
            label == largest_label
            or int(stats[label, cv2.CC_STAT_AREA]) >= int(minimum_component_pixels)
            or np.any(component & contact)
        )
        if keep:
            output |= component
    # Exact invariant: contact never disappears because of SAM or component QC.
    output |= contact
    return np.ascontiguousarray(output)


def _deepest_point(mask: np.ndarray) -> tuple[float, float] | None:
    if not bool(mask.any()):
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return float(x), float(y)


def _contact_positive_points(sole: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """One stable point in each longitudinal anatomy band."""

    x1, y1, x2, y2 = _bbox(sole)
    span = max(1, y2 - y1)
    radius = max(1, int(round(min(x2 - x1, y2 - y1) * 0.025)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    core = cv2.erode(sole.astype(np.uint8), kernel) > 0
    points: list[tuple[float, float]] = []
    for low, high in ((0.02, 0.20), (0.18, 0.42), (0.42, 0.68), (0.68, 0.98)):
        begin = min(y2 - 1, y1 + int(round(span * low)))
        end = max(begin + 1, min(y2, y1 + int(round(span * high))))
        for source in (contact, core, sole):
            band = np.zeros_like(sole, dtype=bool)
            band[begin:end] = source[begin:end]
            point = _deepest_point(band)
            if point is not None:
                points.append(point)
                break
    if not points:
        raise ValueError("could not derive a foreground prompt inside the sole")
    return np.ascontiguousarray(points, dtype=np.float32)


def compute_top_plateau_ratio(
    sole_mask: np.ndarray,
    *,
    probe_fraction: float = 0.015,
) -> float:
    """Width of the flat distal tip divided by maximum longitudinal width.

    The median span over the first max(3, 1.5%) occupied rows avoids a
    one-pixel aliasing spike.  A normal toe end starts with narrow rounded
    tips; a rectangular attached flap starts with a broad plateau.
    """

    sole = _binary(sole_mask, "sole_mask")
    _, y1, _, y2 = _bbox(sole)
    widths: list[int] = []
    for row in sole[y1:y2]:
        columns = np.flatnonzero(row)
        widths.append(int(columns[-1] - columns[0] + 1) if len(columns) else 0)
    maximum = max(widths, default=0)
    if maximum <= 0:
        return 0.0
    count = max(3, int(round(len(widths) * float(probe_fraction))))
    return float(np.median(np.asarray(widths[:count], dtype=np.float32)) / maximum)


def bottom_contact_outlier_mask(
    sole_mask: np.ndarray,
    contact_mask: np.ndarray,
    *,
    bottom_fraction: float = 0.24,
    horizontal_distance_ratio: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return lateral lower-end pixels far from the nearest contact row.

    Distance is horizontal, not Euclidean.  This is important: a genuine heel
    tip may be vertically far below the last pressure pixel, while a reflected
    leg/fabric lobe is laterally outside the heel's contact corridor.

    The returned mask is an integration hook and SAM negative-prompt source;
    it is never directly subtracted from the alpha matte.
    """

    sole = _binary(sole_mask, "sole_mask")
    contact = _binary(contact_mask, "contact_mask", tuple(sole.shape)) & sole
    x1, y1, x2, y2 = _bbox(sole)
    foot_width = max(1, x2 - x1)
    band_start = y1 + int(round((y2 - y1) * (1.0 - float(bottom_fraction))))
    maximum_distance = max(1, int(round(foot_width * float(horizontal_distance_ratio))))
    contact_rows: list[tuple[int, int, int]] = []
    for y in range(y1, y2):
        columns = np.flatnonzero(contact[y])
        if len(columns):
            contact_rows.append((y, int(columns[0]), int(columns[-1])))
    outlier = np.zeros_like(sole, dtype=bool)
    if not contact_rows:
        return outlier, {
            "enabled": False,
            "reason": "contact_mask_empty",
            "band_start_y": int(band_start),
            "maximum_horizontal_distance_pixels": int(maximum_distance),
            "outlier_pixels": 0,
        }
    for y in range(max(y1, band_start), y2):
        _, left, right = min(contact_rows, key=lambda row: (abs(row[0] - y), row[0]))
        row_columns = np.flatnonzero(sole[y])
        if not len(row_columns):
            continue
        invalid = row_columns[
            (row_columns < left - maximum_distance)
            | (row_columns > right + maximum_distance)
        ]
        outlier[y, invalid] = True
    return np.ascontiguousarray(outlier), {
        "enabled": True,
        "reason": None,
        "band_start_y": int(band_start),
        "maximum_horizontal_distance_pixels": int(maximum_distance),
        "outlier_pixels": int(outlier.sum()),
        "outlier_ratio": float(outlier.sum() / max(int(sole.sum()), 1)),
    }


def _negative_points_from_outlier(
    outlier: np.ndarray,
    sole_pixels: int,
    minimum_ratio: float,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        outlier.astype(np.uint8), connectivity=8
    )
    minimum_area = max(8, int(round(sole_pixels * float(minimum_ratio))))
    points: list[tuple[float, float]] = []
    for label in range(1, count):
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area:
            continue
        ys = np.flatnonzero(component.any(axis=1))
        if not len(ys):
            continue
        split = np.linspace(int(ys.min()), int(ys.max()) + 1, num=4, dtype=int)
        for begin, end in zip(split[:-1], split[1:]):
            band = np.zeros_like(component)
            band[begin:max(begin + 1, end)] = component[begin:max(begin + 1, end)]
            point = _deepest_point(band)
            if point is not None:
                points.append(point)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _prompt_measurements(
    mask: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
) -> tuple[float, float]:
    height, width = mask.shape
    px = np.clip(np.rint(positives[:, 0]).astype(np.int64), 0, width - 1)
    py = np.clip(np.rint(positives[:, 1]).astype(np.int64), 0, height - 1)
    positive_recall = float(mask[py, px].mean()) if len(px) else 1.0
    if len(negatives):
        nx = np.clip(np.rint(negatives[:, 0]).astype(np.int64), 0, width - 1)
        ny = np.clip(np.rint(negatives[:, 1]).astype(np.int64), 0, height - 1)
        negative_rejection = float((~mask[ny, nx]).mean())
    else:
        negative_rejection = 1.0
    return positive_recall, negative_rejection


def _variant_prediction(
    predictor: Any,
    sole: np.ndarray,
    contact: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    variant: str,
    dense_logit_scale: float,
    config: SoleAlphaRefinementConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    point_coords = np.concatenate((positives, negatives), axis=0).astype(np.float32)
    point_labels = np.concatenate(
        (np.ones(len(positives), dtype=np.int32), np.zeros(len(negatives), dtype=np.int32))
    )
    x1, y1, x2, y2 = _bbox(sole)
    box = np.asarray([x1, y1, x2 - 1, y2 - 1], dtype=np.float32)
    dense_input = None
    if variant == "yolo_mask_contact":
        dense_input = _candidate_to_predictor_mask_input(
            sole,
            predictor,
            (256, 256),
            logit_scale=float(dense_logit_scale),
        )
    prediction = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=dense_input,
        multimask_output=True,
        return_logits=False,
    )
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise ValueError("SAM predictor must return masks and predicted IoUs")
    masks = np.asarray(prediction[0])
    scores = np.asarray(prediction[1], dtype=np.float32).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(sole.shape):
        raise ValueError(f"SAM returned invalid mask shape: {masks.shape}")
    if len(scores) != len(masks) or not bool(np.isfinite(scores).all()):
        raise ValueError("SAM returned invalid predicted IoUs")

    rows: list[dict[str, Any]] = []
    candidates: list[np.ndarray] = []
    sole_pixels = max(int(sole.sum()), 1)
    contact_pixels = max(int(contact.sum()), 1)
    for index, raw in enumerate(masks):
        clipped = (np.asarray(raw) > 0) & sole
        raw_clipped_pixels = int(clipped.sum())
        clipped |= contact
        candidate = _preserve_complete_components(
            clipped,
            contact,
            minimum_component_pixels=max(
                8, int(round(sole_pixels * config.minimum_component_ratio))
            ),
        )
        area_ratio = float(candidate.sum() / sole_pixels)
        contact_retention = float((candidate & contact).sum() / contact_pixels)
        positive_recall, negative_rejection = _prompt_measurements(
            candidate, positives, negatives
        )
        predicted_iou = float(np.clip(scores[index], 0.0, 1.0))
        rejection_reasons: list[str] = []
        if raw_clipped_pixels == 0:
            rejection_reasons.append("sam_output_empty_before_contact_preservation")
        if predicted_iou < config.minimum_sam_predicted_iou:
            rejection_reasons.append("sam_predicted_iou_below_minimum")
        if area_ratio < config.minimum_area_ratio:
            rejection_reasons.append("area_ratio_below_minimum")
        if contact_retention < config.minimum_contact_retention:
            rejection_reasons.append("contact_retention_below_minimum")
        if positive_recall < 1.0:
            rejection_reasons.append("positive_prompt_missed")
        if negatives.size and negative_rejection < 1.0:
            rejection_reasons.append("negative_prompt_not_rejected")
        rank = (
            0.45 * predicted_iou
            + 0.25 * contact_retention
            + 0.15 * positive_recall
            + 0.10 * negative_rejection
            + 0.05 * min(area_ratio / 0.85, 1.0)
        )
        rows.append(
            {
                "index": int(index),
                "sam_predicted_iou": predicted_iou,
                "raw_clipped_pixels": raw_clipped_pixels,
                "area_pixels": int(candidate.sum()),
                "area_ratio_to_yolo": area_ratio,
                "contact_retention": contact_retention,
                "positive_prompt_recall": positive_recall,
                "negative_prompt_rejection": negative_rejection,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                "rank": float(rank),
            }
        )
        candidates.append(candidate)
    eligible = [index for index, row in enumerate(rows) if row["eligible"]]
    if not eligible:
        return None, {
            "variant": variant,
            "eligible": False,
            "selected_output_index": None,
            "outputs": rows,
        }
    selected = max(eligible, key=lambda index: (rows[index]["rank"], -index))
    return np.ascontiguousarray(candidates[selected]), {
        "variant": variant,
        "eligible": True,
        "selected_output_index": int(selected),
        "selected": rows[selected],
        "outputs": rows,
    }


def refine_sole_alpha_with_sam(
    predictor: Any,
    image_rgb: np.ndarray,
    yolo_sole_mask: np.ndarray,
    contact_mask: np.ndarray,
    *,
    config: SoleAlphaRefinementConfig | None = None,
    set_image: bool = True,
) -> SoleAlphaRefinementResult:
    """Refine a canonical toes-up YOLO sole with two local SAM proposals.

    Selection is intentionally not a learned confidence blend:

    * a broad distal plateau (>=0.28) selects ``box_contact`` because the YOLO
      dense prompt would preserve the attached top flap;
    * otherwise ``yolo_mask_contact`` preserves all real toe lobes and receives
      bottom-contact outlier points as negative prompts;
    * any QC failure falls back to the other eligible SAM proposal and finally
      to the original YOLO support, never to an empty or truncated foot.
    """

    options = config or SoleAlphaRefinementConfig()
    image = _rgb_uint8(image_rgb)
    sole = _binary(yolo_sole_mask, "yolo_sole_mask", tuple(image.shape[:2]))
    if not bool(sole.any()):
        raise ValueError("yolo_sole_mask is empty")
    contact = _binary(contact_mask, "contact_mask", tuple(sole.shape)) & sole
    if not bool(contact.any()):
        raise ValueError("contact_mask is empty inside yolo_sole_mask")
    if set_image:
        if not hasattr(predictor, "set_image"):
            raise TypeError("predictor must provide set_image when set_image=True")
        predictor.set_image(image)
    if not hasattr(predictor, "predict"):
        raise TypeError("predictor must provide predict")

    positives = _contact_positive_points(sole, contact)
    outlier, outlier_audit = bottom_contact_outlier_mask(
        sole,
        contact,
        bottom_fraction=options.bottom_band_fraction,
        horizontal_distance_ratio=options.bottom_contact_distance_ratio,
    )
    negatives = _negative_points_from_outlier(
        outlier,
        int(sole.sum()),
        options.minimum_bottom_outlier_ratio,
    )
    top_plateau_ratio = compute_top_plateau_ratio(
        sole, probe_fraction=options.top_probe_fraction
    )

    variant_audits: dict[str, Any] = {}
    variants: dict[str, np.ndarray | None] = {}
    errors: dict[str, str] = {}
    for variant, variant_negatives in (
        ("box_contact", np.empty((0, 2), dtype=np.float32)),
        ("yolo_mask_contact", negatives),
    ):
        try:
            variants[variant], variant_audits[variant] = _variant_prediction(
                predictor,
                sole,
                contact,
                positives,
                variant_negatives,
                variant=variant,
                dense_logit_scale=options.dense_prompt_logit_scale,
                config=options,
            )
        except Exception as exc:  # fail-safe is part of the public contract
            variants[variant] = None
            errors[variant] = f"{type(exc).__name__}: {exc}"
            variant_audits[variant] = {
                "variant": variant,
                "eligible": False,
                "error": errors[variant],
            }

    preferred = (
        "box_contact"
        if top_plateau_ratio >= options.top_plateau_threshold
        else "yolo_mask_contact"
    )
    alternate = "yolo_mask_contact" if preferred == "box_contact" else "box_contact"
    if variants[preferred] is not None:
        selected_variant = preferred
        selected = variants[preferred]
        fallback_used = False
        selection_reason = (
            "top_plateau_at_or_above_threshold"
            if preferred == "box_contact"
            else "top_plateau_below_threshold_conservative_dense_prompt"
        )
    elif variants[alternate] is not None:
        selected_variant = alternate
        selected = variants[alternate]
        fallback_used = True
        selection_reason = "preferred_sam_variant_failed_qc"
    else:
        selected_variant = "yolo_fail_safe"
        selected = sole.copy()
        fallback_used = True
        selection_reason = "all_sam_variants_failed_qc"

    selected = np.ascontiguousarray((selected | contact) & sole)
    final_contact_retention = float(
        (selected & contact).sum() / max(int(contact.sum()), 1)
    )
    if final_contact_retention != 1.0:
        selected_variant = "yolo_fail_safe"
        selected = sole.copy()
        fallback_used = True
        selection_reason = "postselection_contact_retention_fail_safe"
        final_contact_retention = 1.0

    audit: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "local_sam_complete_sole_alpha_v1",
        "provenance": {
            "local_only": True,
            "gpt_used": False,
            "openai_used": False,
            "network_used": False,
        },
        "config": asdict(options),
        "input": {
            "shape": [int(value) for value in sole.shape],
            "yolo_pixels": int(sole.sum()),
            "contact_pixels": int(contact.sum()),
        },
        "prompts": {
            "positive_points_xy": positives.tolist(),
            "bottom_negative_points_xy": negatives.tolist(),
        },
        "top_plateau_ratio": float(top_plateau_ratio),
        "top_plateau_triggered": bool(
            top_plateau_ratio >= options.top_plateau_threshold
        ),
        "bottom_contact_outlier": outlier_audit,
        "bottom_contact_outlier_used_as_sam_negative_only": True,
        "bottom_contact_outlier_directly_subtracted": False,
        "variants": variant_audits,
        "errors": errors,
        "selection": {
            "preferred_variant": preferred,
            "selected_variant": selected_variant,
            "reason": selection_reason,
            "fallback_used": bool(fallback_used),
        },
        "final": {
            "pixels": int(selected.sum()),
            "area_ratio_to_yolo": float(selected.sum() / max(int(sole.sum()), 1)),
            "contact_retention": final_contact_retention,
            "contained_in_yolo": bool(not np.any(selected & ~sole)),
            "mask_sha256": sha256(
                np.ascontiguousarray(selected.astype(np.uint8)).tobytes()
            ).hexdigest(),
        },
    }
    return SoleAlphaRefinementResult(
        mask=selected,
        selected_variant=selected_variant,
        fallback_used=fallback_used,
        audit=audit,
    )


__all__ = [
    "SoleAlphaRefinementConfig",
    "SoleAlphaRefinementResult",
    "bottom_contact_outlier_mask",
    "compute_top_plateau_ratio",
    "refine_sole_alpha_with_sam",
]
