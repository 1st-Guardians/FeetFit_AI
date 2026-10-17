"""YOLO11-seg foot extraction for qualitative/external inference only.

This module adapts the foot-boundary pipeline from ``the original foot-analysis project``.  It is
deliberately not used by the quantitative train/valid/test datasets: callers
opt in only for ``image/test`` monitoring or standalone inference.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Ultralytics appends its own "Ultralytics" directory. Keeping that settings
# file under the project avoids permission failures in locked Windows profiles.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


_ROTATION_SELECTION_POLICY = "quarter_turn_consensus_vertical_v1"
_QUARTER_TURN_CONSENSUS_IOU = 0.75
_MIN_IMAGE_AREA_RATIO = 0.02
_MAX_IMAGE_AREA_RATIO = 0.60
_MIN_BBOX_FILL_RATIO = 0.42
_MAX_BBOX_FILL_RATIO = 0.86
_MIN_PCA_ELONGATION = 1.80
_MAX_PCA_ELONGATION = 4.50
_EDGE_BAND_RATIO = 0.003
_MIN_EDGE_BAND_PIXELS = 2
_MIN_CONSENSUS_AREA_RATIO = 0.75
_MAX_CONSENSUS_AREA_RATIO = 1.12
_VERTICAL_AXIS_MAX_DEGREES = 30.0
_VERTICAL_FALLBACK_TOLERANCE_DEGREES = 10.0
_SINGLE_VIEW_MIN_CONFIDENCE = 0.50

# Degrees are clockwise in image coordinates.  The inverse code always maps
# a YOLO mask back to the untouched source image before any candidates are
# compared or returned.
_QUARTER_TURNS: tuple[tuple[str, int, int | None, int | None], ...] = (
    ("original", 0, None, None),
    (
        "rotate_90_clockwise",
        90,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ),
    ("rotate_180", 180, cv2.ROTATE_180, cv2.ROTATE_180),
    (
        "rotate_270_clockwise",
        270,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_90_CLOCKWISE,
    ),
)


@dataclass(frozen=True)
class FootRoiConfig:
    confidence: float = 0.25
    image_size: int = 960
    mask_threshold: float = 0.5
    min_component_area: int = 2000
    duplicate_iou_threshold: float = 0.6
    crop_padding: int = 28
    crop_padding_ratio: float = 0.0
    device: str = "cpu"
    retina_masks: bool = True
    cutout_background: bool = True
    select_primary_only: bool = True
    # A sole-segmentation model is orientation-sensitive.  When enabled, always
    # compare the 0/90/180/270-degree detector views numerically.  Reflections
    # are intentionally excluded so left/right anatomy is never mirrored.
    transform_tta: bool = True
    # Reproduce the fine-tuned SegFormer evaluation pipeline: accept the
    # primary original-view mask and try H/V/180 TTA only when it is absent.
    # The stricter quarter-turn consensus remains the default for legacy users.
    target_compatible_mode: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= float(self.mask_threshold) <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if not 0.0 <= float(self.duplicate_iou_threshold) <= 1.0:
            raise ValueError("duplicate_iou_threshold must be in [0, 1]")
        if min(self.image_size, self.min_component_area) <= 0:
            raise ValueError("image_size and min_component_area must be positive")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if not 0.0 <= float(self.crop_padding_ratio) <= 0.5:
            raise ValueError("crop_padding_ratio must be in [0, 0.5]")


@dataclass
class FootRoiResult:
    """Detected original-space foot mask plus the exact SAM/Swin model crop."""

    found: bool
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    crop_image: np.ndarray
    crop_mask: np.ndarray
    confidence: float
    instance_count: int
    reason: str = ""
    detection_pass: str = "original"
    tta_attempted_transforms: tuple[str, ...] = ()
    selection_audit: dict[str, Any] = field(default_factory=dict)
    # Compatibility fields above remain in untouched source coordinates.
    # These fields expose the exact selected cardinal YOLO view so pressure
    # segmentation can use the same geometry that produced the reliable mask.
    cardinal_rotation_degrees_clockwise: int = 0
    cardinal_view_image: np.ndarray | None = None
    cardinal_view_mask: np.ndarray | None = None
    cardinal_bbox: tuple[int, int, int, int] | None = None
    cardinal_crop_image: np.ndarray | None = None
    cardinal_crop_mask: np.ndarray | None = None

    @property
    def original_shape(self) -> tuple[int, int]:
        return int(self.mask.shape[0]), int(self.mask.shape[1])

    @property
    def crop_shape(self) -> tuple[int, int]:
        return int(self.crop_image.shape[0]), int(self.crop_image.shape[1])

    @property
    def status(self) -> str:
        """Machine-readable detector status without discarding TTA provenance."""

        if self.found:
            return (
                "detected"
                if self.detection_pass == "original"
                else f"detected_{self.detection_pass}_tta"
            )
        return self.reason or "yolo_foot_not_detected"


def _as_rgb_uint8(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA image, got {array.shape}")
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first > 0, second > 0).sum()
    union = np.logical_or(first > 0, second > 0).sum()
    return 0.0 if union == 0 else float(intersection / union)


def _components(binary_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    output: list[np.ndarray] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_area):
            continue
        output.append((labels == label).astype(np.uint8))
    return output


def _deduplicate(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    # Prefer a confident, large mask. Expected inputs contain one photographed
    # sole; selecting the primary instance prevents a second leg/object mask
    # from being unioned into the valid pressure region.
    ordered = sorted(
        items,
        key=lambda item: (int(item["area"]), float(item["score"])),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for item in ordered:
        if not any(_mask_iou(item["mask"], other["mask"]) >= threshold for other in kept):
            kept.append(item)
    return kept


def bbox_from_mask(
    mask: np.ndarray,
    padding: int = 28,
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    height, width = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0, 0, width, height
    foot_width = int(xs.max()) - int(xs.min()) + 1
    foot_height = int(ys.max()) - int(ys.min()) + 1
    relative_padding = int(round(max(foot_width, foot_height) * float(padding_ratio)))
    effective_padding = max(int(padding), relative_padding)
    return (
        max(0, int(xs.min()) - effective_padding),
        max(0, int(ys.min()) - effective_padding),
        min(width, int(xs.max()) + 1 + effective_padding),
        min(height, int(ys.max()) + 1 + effective_padding),
    )


def _rotate_quarter_turn(values: np.ndarray, code: int | None) -> np.ndarray:
    if code is None:
        return np.ascontiguousarray(values)
    return np.ascontiguousarray(cv2.rotate(values, code))


def _candidate_shape_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Return rotation-view geometry used to reject non-sole components.

    ``bbox_fill_ratio`` is foreground area divided by the inclusive tight-bbox
    area. ``pca_elongation`` is ``sqrt(lambda_major / lambda_minor)``.  The
    unsigned PCA angle is measured from the detector image's vertical axis, so
    zero means the model saw an upright long axis and 90 means horizontal.
    """

    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not bool(binary.any()):
        raise ValueError("candidate mask must be a non-empty 2D array")
    height, width = binary.shape
    ys, xs = np.where(binary)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    area = int(binary.sum())
    bbox_area = max(1, bbox_width * bbox_height)

    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_value = max(float(eigenvalues[order[0]]), 1e-6)
    minor_value = max(float(eigenvalues[order[1]]), 1e-6)
    major_vector = eigenvectors[:, order[0]]
    vertical_cosine = float(np.clip(abs(float(major_vector[1])), 0.0, 1.0))
    angle_from_vertical = float(math.degrees(math.acos(vertical_cosine)))

    edge_band = max(
        _MIN_EDGE_BAND_PIXELS,
        int(round(min(height, width) * _EDGE_BAND_RATIO)),
    )
    edge_zone = np.zeros_like(binary, dtype=bool)
    edge_zone[:edge_band, :] = True
    edge_zone[-edge_band:, :] = True
    edge_zone[:, :edge_band] = True
    edge_zone[:, -edge_band:] = True
    edge_pixels = int(np.count_nonzero(binary & edge_zone))

    return {
        "area_pixels": area,
        "image_area_ratio": float(area / max(1, height * width)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "bbox_height_width_ratio": float(bbox_height / max(1, bbox_width)),
        "bbox_fill_ratio": float(area / bbox_area),
        "pca_elongation": float(math.sqrt(major_value / minor_value)),
        "pca_axis_from_vertical_degrees": angle_from_vertical,
        "edge_band_pixels": int(edge_band),
        "edge_foreground_pixels": edge_pixels,
        "edge_foreground_ratio": float(edge_pixels / max(1, area)),
        "edge_clipped": bool(edge_pixels > 0),
    }


def _shape_rejection_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    area_ratio = float(metrics["image_area_ratio"])
    if not _MIN_IMAGE_AREA_RATIO <= area_ratio <= _MAX_IMAGE_AREA_RATIO:
        reasons.append("image_area_ratio")
    bbox_fill = float(metrics["bbox_fill_ratio"])
    if not _MIN_BBOX_FILL_RATIO <= bbox_fill <= _MAX_BBOX_FILL_RATIO:
        reasons.append("bbox_fill_ratio")
    elongation = float(metrics["pca_elongation"])
    if not _MIN_PCA_ELONGATION <= elongation <= _MAX_PCA_ELONGATION:
        reasons.append("pca_elongation")
    if bool(metrics["edge_clipped"]):
        reasons.append("edge_clipped")
    return reasons


def _mean_pairwise_iou(indices: tuple[int, ...], matrix: np.ndarray) -> float:
    values = [
        float(matrix[first, second])
        for offset, first in enumerate(indices)
        for second in indices[offset + 1 :]
    ]
    return float(np.mean(values)) if values else 0.0


def _maximum_consensus_clique(
    records: list[dict[str, Any]],
    matrix: np.ndarray,
) -> tuple[int, ...]:
    """Find the largest all-pairs-consistent subset of at most four views."""

    best: tuple[int, ...] = ()
    best_key: tuple[Any, ...] | None = None
    count = len(records)
    for bitmask in range(1, 1 << count):
        indices = tuple(index for index in range(count) if bitmask & (1 << index))
        if any(
            float(matrix[first, second]) < _QUARTER_TURN_CONSENSUS_IOU
            for offset, first in enumerate(indices)
            for second in indices[offset + 1 :]
        ):
            continue
        angles = [
            float(records[index]["metrics"]["pca_axis_from_vertical_degrees"])
            for index in indices
        ]
        # In an equal-size split consensus, prefer the cluster containing the
        # detector's vertical views before considering confidence.  The final
        # tuple makes lower clockwise degrees win an exact tie.
        key = (
            len(indices),
            sum(angle <= _VERTICAL_AXIS_MAX_DEGREES for angle in angles),
            _mean_pairwise_iou(indices, matrix),
            -float(np.mean(angles)),
            float(sum(float(records[index]["confidence"]) for index in indices)),
            tuple(-int(records[index]["rotation_index"]) for index in indices),
        )
        if best_key is None or key > best_key:
            best = indices
            best_key = key
    return best


def _select_quarter_turn_record(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select one inverse-mapped YOLO sole without using image semantics."""

    if not records:
        return None, {
            "failure_reason": "no_shape_valid_rotation_candidate",
            "consensus_rotations": [],
            "eligible_rotations": [],
        }

    count = len(records)
    matrix = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = _mask_iou(
                records[first]["source_mask"], records[second]["source_mask"]
            )
            matrix[first, second] = value
            matrix[second, first] = value

    clique_indices = _maximum_consensus_clique(records, matrix)
    if len(records) > 1 and len(clique_indices) < 2:
        return None, {
            "failure_reason": "quarter_turn_masks_have_no_iou_consensus",
            "consensus_rotations": [],
            "eligible_rotations": [],
            "pairwise_iou": matrix.tolist(),
        }

    clique = [records[index] for index in clique_indices]
    if len(clique) == 1:
        only = clique[0]
        if (
            float(only["confidence"]) < _SINGLE_VIEW_MIN_CONFIDENCE
            or float(only["metrics"]["pca_axis_from_vertical_degrees"])
            > _VERTICAL_AXIS_MAX_DEGREES
        ):
            return None, {
                "failure_reason": "single_view_candidate_is_not_reliable",
                "consensus_rotations": [only["name"]],
                "eligible_rotations": [],
                "pairwise_iou": matrix.tolist(),
            }

    median_area = float(
        np.median([float(record["metrics"]["area_pixels"]) for record in clique])
    )
    area_stable: list[dict[str, Any]] = []
    for record in clique:
        ratio = float(record["metrics"]["area_pixels"]) / max(median_area, 1.0)
        record["consensus_area_ratio"] = ratio
        if _MIN_CONSENSUS_AREA_RATIO <= ratio <= _MAX_CONSENSUS_AREA_RATIO:
            area_stable.append(record)
    if not area_stable:
        return None, {
            "failure_reason": "all_consensus_candidates_failed_area_stability",
            "consensus_rotations": [record["name"] for record in clique],
            "eligible_rotations": [],
            "consensus_median_area_pixels": median_area,
            "pairwise_iou": matrix.tolist(),
        }

    strict_vertical = [
        record
        for record in area_stable
        if float(record["metrics"]["pca_axis_from_vertical_degrees"])
        <= _VERTICAL_AXIS_MAX_DEGREES
    ]
    if strict_vertical:
        eligible = strict_vertical
        vertical_mode = "strict_30_degrees"
    else:
        best_angle = min(
            float(record["metrics"]["pca_axis_from_vertical_degrees"])
            for record in area_stable
        )
        eligible = [
            record
            for record in area_stable
            if float(record["metrics"]["pca_axis_from_vertical_degrees"])
            <= best_angle + _VERTICAL_FALLBACK_TOLERANCE_DEGREES
        ]
        vertical_mode = "nearest_axis_fallback"

    for record in eligible:
        source_index = int(record["record_index"])
        peer_indices = [index for index in clique_indices if index != source_index]
        consensus_score = (
            float(np.mean([matrix[source_index, index] for index in peer_indices]))
            if peer_indices
            else 0.0
        )
        area_stability = float(
            math.exp(-abs(math.log(max(float(record["consensus_area_ratio"]), 1e-9))))
        )
        selection_score = (
            0.60 * float(record["confidence"])
            + 0.25 * consensus_score
            + 0.15 * area_stability
        )
        record["mean_consensus_iou"] = consensus_score
        record["area_stability_score"] = area_stability
        record["selection_score"] = float(selection_score)

    selected = max(
        eligible,
        key=lambda record: (
            float(record["selection_score"]),
            -int(record["rotation_index"]),
        ),
    )
    return selected, {
        "failure_reason": None,
        "consensus_rotations": [record["name"] for record in clique],
        "eligible_rotations": [record["name"] for record in eligible],
        "consensus_median_area_pixels": median_area,
        "vertical_selection_mode": vertical_mode,
        "pairwise_iou": matrix.tolist(),
        "selected_rotation": selected["name"],
        "selected_score": float(selected["selection_score"]),
    }


class YoloFootRoiDetector:
    """Load one local YOLO11n-seg model and extract the primary sole mask."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: FootRoiConfig | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"YOLO foot checkpoint not found: {checkpoint_path}")
        self.checkpoint = checkpoint_path.resolve()
        self.config = config or FootRoiConfig()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - actionable environment error
            raise ImportError(
                "Foot ROI extraction requires ultralytics. Install requirements.txt."
            ) from exc
        self.model = YOLO(str(self.checkpoint), task="segment")
        if getattr(self.model, "task", None) != "segment":
            raise ValueError(f"Expected a YOLO segmentation checkpoint: {checkpoint_path}")
        names = getattr(self.model, "names", None)
        class_zero = names.get(0) if isinstance(names, dict) else (
            names[0] if isinstance(names, (list, tuple)) and names else None
        )
        if class_zero is not None and str(class_zero).strip().lower() != "foot":
            raise ValueError(
                "Expected YOLO class 0 to be 'foot', "
                f"got {class_zero!r}: {checkpoint_path}"
            )

    def _predict_candidates(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return component candidates in the coordinate system of ``rgb``."""

        height, width = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            source=bgr,
            conf=float(self.config.confidence),
            imgsz=int(self.config.image_size),
            retina_masks=bool(self.config.retina_masks),
            device=self.config.device,
            save=False,
            verbose=False,
        )
        candidates: list[dict[str, Any]] = []
        if results:
            result = results[0]
            masks = getattr(result, "masks", None)
            mask_data = None if masks is None else getattr(masks, "data", None)
            if mask_data is not None and len(mask_data):
                raw_masks = mask_data.detach().float().cpu().numpy()
                boxes = getattr(result, "boxes", None)
                scores = (
                    boxes.conf.detach().float().cpu().numpy()
                    if boxes is not None and boxes.conf is not None
                    else np.ones(len(raw_masks), dtype=np.float32)
                )
                classes = (
                    boxes.cls.detach().long().cpu().numpy()
                    if boxes is not None and boxes.cls is not None
                    else np.zeros(len(raw_masks), dtype=np.int64)
                )
                for index, raw_mask in enumerate(raw_masks):
                    if index < len(classes) and int(classes[index]) != 0:
                        continue
                    resized = (
                        raw_mask
                        if raw_mask.shape == (height, width)
                        else cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    )
                    binary = (resized > float(self.config.mask_threshold)).astype(np.uint8)
                    score = float(scores[index]) if index < len(scores) else 1.0
                    for component in _components(binary, self.config.min_component_area):
                        candidates.append(
                            {
                                "mask": component,
                                "score": score,
                                "area": int(component.sum()),
                            }
                        )

        return _deduplicate(candidates, self.config.duplicate_iou_threshold)

    def _target_compatible_extract(
        self,
        rgb: np.ndarray,
        original_candidates: list[dict[str, Any]],
        selection_audit: dict[str, Any],
    ) -> FootRoiResult | None:
        """Reproduce the validated SegFormer script's permissive YOLO policy.

        It first accepts the primary original-view mask; when none exists, it
        tries the same horizontal/vertical/180-degree self-inverse transforms
        used by the standalone evaluation script.
        """

        height, width = rgb.shape[:2]
        candidates = list(original_candidates)
        detection_pass = "original"
        attempted_transforms: list[str] = []
        if not candidates and self.config.transform_tta:
            transforms = (
                (
                    "horizontal_flip",
                    lambda values: np.ascontiguousarray(values[:, ::-1, ...]),
                ),
                (
                    "vertical_flip",
                    lambda values: np.ascontiguousarray(values[::-1, :, ...]),
                ),
                (
                    "rotate_180",
                    lambda values: np.ascontiguousarray(values[::-1, ::-1, ...]),
                ),
            )
            valid_passes: dict[str, list[dict[str, Any]]] = {}
            for transform_name, transform in transforms:
                attempted_transforms.append(transform_name)
                transformed_candidates = self._predict_candidates(transform(rgb))
                restored = [
                    {**item, "mask": transform(item["mask"])}
                    for item in transformed_candidates
                ]
                if restored:
                    valid_passes[transform_name] = restored
            if valid_passes:
                detection_pass = max(
                    valid_passes,
                    key=lambda name: max(
                        float(item["score"]) for item in valid_passes[name]
                    ),
                )
                candidates = sorted(
                    valid_passes[detection_pass],
                    key=lambda item: (float(item["score"]), int(item["area"])),
                    reverse=True,
                )

        if not candidates:
            selection_audit["target_compatible_selection"] = {
                "executed": True,
                "succeeded": False,
                "policy": "primary_mask_then_self_inverse_transform_tta",
                "attempted_transforms": attempted_transforms,
            }
            return None

        selected = candidates[:1] if self.config.select_primary_only else candidates
        combined = np.zeros((height, width), dtype=np.uint8)
        for item in selected:
            combined[np.asarray(item["mask"]) > 0] = 1
        bbox = bbox_from_mask(
            combined,
            self.config.crop_padding,
            self.config.crop_padding_ratio,
        )
        x1, y1, x2, y2 = bbox
        crop_mask = np.ascontiguousarray(combined[y1:y2, x1:x2])
        crop = np.ascontiguousarray(rgb[y1:y2, x1:x2].copy())
        if self.config.cutout_background:
            crop[crop_mask == 0] = 0
        confidence = float(max(float(item["score"]) for item in selected))
        selection_audit["target_compatible_selection"] = {
            "executed": True,
            "succeeded": True,
            "policy": "primary_mask_then_self_inverse_transform_tta",
            "detection_pass": detection_pass,
            "candidate_count": len(candidates),
            "selected_confidence": confidence,
            "selected_area_pixels": int(np.count_nonzero(combined)),
            "attempted_transforms": attempted_transforms,
        }
        return FootRoiResult(
            found=True,
            mask=np.ascontiguousarray(combined),
            bbox=bbox,
            crop_image=crop,
            crop_mask=crop_mask,
            confidence=confidence,
            instance_count=len(candidates),
            reason="",
            detection_pass=detection_pass,
            tta_attempted_transforms=tuple(attempted_transforms),
            selection_audit=selection_audit,
            cardinal_rotation_degrees_clockwise=0,
            cardinal_view_image=np.ascontiguousarray(rgb),
            cardinal_view_mask=np.ascontiguousarray(combined),
            cardinal_bbox=bbox,
            cardinal_crop_image=np.ascontiguousarray(crop),
            cardinal_crop_mask=np.ascontiguousarray(crop_mask),
        )

    def extract(self, image: np.ndarray | Image.Image) -> FootRoiResult:
        rgb = _as_rgb_uint8(image)
        height, width = rgb.shape[:2]
        if self.config.target_compatible_mode:
            selection_audit: dict[str, Any] = {
                "policy": "segformer_evaluation_primary_mask_v1",
                "strict_quarter_turn_consensus_bypassed": True,
                "reflections_used_for_detection_only": True,
            }
            result = self._target_compatible_extract(
                rgb,
                self._predict_candidates(rgb),
                selection_audit,
            )
            if result is not None:
                return result
            empty = np.zeros((height, width), dtype=np.uint8)
            return FootRoiResult(
                found=False,
                mask=empty,
                bbox=(0, 0, width, height),
                crop_image=np.zeros_like(rgb),
                crop_mask=empty.copy(),
                confidence=0.0,
                instance_count=0,
                reason=(
                    "YOLO foot not detected after original and transform TTA; "
                    "pressure output forced to zero"
                ),
                detection_pass="none",
                tta_attempted_transforms=tuple(
                    selection_audit["target_compatible_selection"][
                        "attempted_transforms"
                    ]
                ),
                selection_audit=selection_audit,
            )

        turns = _QUARTER_TURNS if self.config.transform_tta else _QUARTER_TURNS[:1]
        attempted_transforms = [name for name, _, _, _ in turns if name != "original"]
        records: list[dict[str, Any]] = []
        rotation_rows: list[dict[str, Any]] = []

        for rotation_index, (name, degrees, forward_code, inverse_code) in enumerate(turns):
            view_image = _rotate_quarter_turn(rgb, forward_code)
            raw_candidates = self._predict_candidates(view_image)
            candidate_rows: list[dict[str, Any]] = []
            valid_items: list[dict[str, Any]] = []
            for candidate_index, item in enumerate(raw_candidates):
                metrics = _candidate_shape_metrics(item["mask"])
                rejection_reasons = _shape_rejection_reasons(metrics)
                candidate_rows.append(
                    {
                        "candidate_index": int(candidate_index),
                        "confidence": float(item["score"]),
                        **metrics,
                        "shape_qc_pass": not rejection_reasons,
                        "rejection_reasons": rejection_reasons,
                    }
                )
                if not rejection_reasons:
                    valid_items.append(
                        {
                            "candidate_index": int(candidate_index),
                            "item": item,
                            "metrics": metrics,
                        }
                    )

            # Area-first selection prevents a tiny high-confidence apparatus
            # fragment from replacing the complete plantar silhouette.
            valid_items.sort(
                key=lambda payload: (
                    int(payload["metrics"]["area_pixels"]),
                    float(payload["item"]["score"]),
                    -int(payload["candidate_index"]),
                ),
                reverse=True,
            )
            primary = valid_items[0] if valid_items else None
            row: dict[str, Any] = {
                "name": name,
                "degrees_clockwise": int(degrees),
                "raw_candidate_count": len(raw_candidates),
                "shape_valid_candidate_count": len(valid_items),
                "candidates": candidate_rows,
                "primary_candidate_index": (
                    None if primary is None else int(primary["candidate_index"])
                ),
            }
            rotation_rows.append(row)
            if primary is None:
                continue

            view_mask = np.ascontiguousarray(primary["item"]["mask"] > 0)
            source_mask = _rotate_quarter_turn(view_mask.astype(np.uint8), inverse_code) > 0
            record = {
                "record_index": len(records),
                "rotation_index": int(rotation_index),
                "name": name,
                "degrees_clockwise": int(degrees),
                "confidence": float(primary["item"]["score"]),
                "metrics": primary["metrics"],
                "source_mask": np.ascontiguousarray(source_mask),
                "view_mask": view_mask,
                "view_image": view_image,
                "valid_items": valid_items,
                "raw_candidate_count": len(raw_candidates),
            }
            records.append(record)

        selected_record, selector_audit = _select_quarter_turn_record(records)
        audit_records: list[dict[str, Any]] = []
        for record in records:
            record_index = int(record["record_index"])
            pairwise = selector_audit.get("pairwise_iou")
            audit_records.append(
                {
                    "name": record["name"],
                    "degrees_clockwise": int(record["degrees_clockwise"]),
                    "confidence": float(record["confidence"]),
                    **record["metrics"],
                    "consensus_area_ratio": record.get("consensus_area_ratio"),
                    "mean_consensus_iou": record.get("mean_consensus_iou"),
                    "area_stability_score": record.get("area_stability_score"),
                    "selection_score": record.get("selection_score"),
                    "pairwise_iou": (
                        None
                        if pairwise is None
                        else {
                            records[other]["name"]: float(pairwise[record_index][other])
                            for other in range(len(records))
                            if other != record_index
                        }
                    ),
                }
            )
        selection_audit: dict[str, Any] = {
            "policy": _ROTATION_SELECTION_POLICY,
            "rotation_convention": (
                "degrees_clockwise_in_detector_input; selected mask is also "
                "inverse-rotated to untouched source coordinates"
            ),
            "reflections_used": False,
            "thresholds": {
                "consensus_iou": _QUARTER_TURN_CONSENSUS_IOU,
                "image_area_ratio": [_MIN_IMAGE_AREA_RATIO, _MAX_IMAGE_AREA_RATIO],
                "bbox_fill_ratio": [_MIN_BBOX_FILL_RATIO, _MAX_BBOX_FILL_RATIO],
                "pca_elongation": [_MIN_PCA_ELONGATION, _MAX_PCA_ELONGATION],
                "edge_band_ratio": _EDGE_BAND_RATIO,
                "minimum_edge_band_pixels": _MIN_EDGE_BAND_PIXELS,
                "edge_clipped_definition": "any foreground pixel in the edge band",
                "consensus_area_ratio": [
                    _MIN_CONSENSUS_AREA_RATIO,
                    _MAX_CONSENSUS_AREA_RATIO,
                ],
                "vertical_axis_max_degrees": _VERTICAL_AXIS_MAX_DEGREES,
                "vertical_fallback_tolerance_degrees": (
                    _VERTICAL_FALLBACK_TOLERANCE_DEGREES
                ),
                "single_view_min_confidence": _SINGLE_VIEW_MIN_CONFIDENCE,
                "selection_score": (
                    "0.60*confidence + 0.25*mean_consensus_iou + "
                    "0.15*exp(-abs(log(area/median_area)))"
                ),
            },
            "attempted_rotations": [name for name, _, _, _ in turns],
            "rotation_rows": rotation_rows,
            "shape_valid_rotations": audit_records,
            **selector_audit,
        }

        if selected_record is None:
            # Conservative fail-closed behavior. Callers should report the
            # sample as invalid and skip SAM/Swin; the black crop also prevents
            # an accidental fallback from treating the complete background as
            # a valid foot image.
            empty = np.zeros((height, width), dtype=np.uint8)
            return FootRoiResult(
                found=False,
                mask=empty,
                bbox=(0, 0, width, height),
                crop_image=np.zeros_like(rgb),
                crop_mask=empty.copy(),
                confidence=0.0,
                instance_count=0,
                reason=(
                    "YOLO quarter-turn bank found no reliable consensus sole; "
                    "pressure output forced to zero"
                    if attempted_transforms
                    else "YOLO foot not detected; pressure output forced to zero"
                ),
                detection_pass="none",
                tta_attempted_transforms=tuple(attempted_transforms),
                selection_audit=selection_audit,
            )

        detection_pass = str(selected_record["name"])
        view_image = np.ascontiguousarray(selected_record["view_image"])
        view_mask = np.zeros(view_image.shape[:2], dtype=np.uint8)
        selected_items = (
            selected_record["valid_items"][:1]
            if self.config.select_primary_only
            else selected_record["valid_items"]
        )
        for payload in selected_items:
            view_mask[np.asarray(payload["item"]["mask"]) > 0] = 1
        _, _, _, inverse_code = _QUARTER_TURNS[int(selected_record["rotation_index"])]
        combined = _rotate_quarter_turn(view_mask, inverse_code).astype(np.uint8)

        bbox = bbox_from_mask(
            combined,
            self.config.crop_padding,
            self.config.crop_padding_ratio,
        )
        x1, y1, x2, y2 = bbox
        crop_mask = combined[y1:y2, x1:x2].copy()
        crop = rgb[y1:y2, x1:x2].copy()
        if self.config.cutout_background:
            crop[crop_mask == 0] = 0

        cardinal_bbox = bbox_from_mask(
            view_mask,
            self.config.crop_padding,
            self.config.crop_padding_ratio,
        )
        cx1, cy1, cx2, cy2 = cardinal_bbox
        cardinal_crop_mask = view_mask[cy1:cy2, cx1:cx2].copy()
        cardinal_crop = view_image[cy1:cy2, cx1:cx2].copy()
        if self.config.cutout_background:
            cardinal_crop[cardinal_crop_mask == 0] = 0

        selection_audit["source_bbox_xyxy"] = list(bbox)
        selection_audit["cardinal_bbox_xyxy"] = list(cardinal_bbox)
        selection_audit["cardinal_rotation_degrees_clockwise"] = int(
            selected_record["degrees_clockwise"]
        )
        selection_audit["output_instance_count"] = len(selected_items)
        return FootRoiResult(
            found=True,
            mask=combined,
            bbox=bbox,
            crop_image=np.ascontiguousarray(crop),
            crop_mask=np.ascontiguousarray(crop_mask),
            confidence=float(selected_record["confidence"]),
            instance_count=int(selected_record["raw_candidate_count"]),
            reason="",
            detection_pass=detection_pass,
            tta_attempted_transforms=tuple(attempted_transforms),
            selection_audit=selection_audit,
            cardinal_rotation_degrees_clockwise=int(
                selected_record["degrees_clockwise"]
            ),
            cardinal_view_image=view_image,
            cardinal_view_mask=np.ascontiguousarray(view_mask),
            cardinal_bbox=cardinal_bbox,
            cardinal_crop_image=np.ascontiguousarray(cardinal_crop),
            cardinal_crop_mask=np.ascontiguousarray(cardinal_crop_mask),
        )


def support_mask_to_low_res(
    support_mask: np.ndarray,
    *,
    input_size: tuple[int, int],
    sam_image_size: int,
    output_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map an original crop mask into SAM's top-left low-resolution canvas."""

    mask = np.asarray(support_mask)
    if mask.ndim != 2 or min(mask.shape) <= 0:
        raise ValueError(f"support_mask must be a non-empty 2D array, got {mask.shape}")
    input_height, input_width = int(input_size[0]), int(input_size[1])
    if min(input_height, input_width, sam_image_size, output_size) <= 0:
        raise ValueError("SAM sizes must be positive")
    tensor = torch.from_numpy((mask > 0).astype(np.float32)).view(1, 1, *mask.shape)
    tensor = F.interpolate(tensor, size=(input_height, input_width), mode="nearest")
    canvas = torch.zeros((1, 1, sam_image_size, sam_image_size), dtype=torch.float32)
    canvas[:, :, :input_height, :input_width] = tensor
    low = F.interpolate(canvas, size=(output_size, output_size), mode="nearest")
    return low.to(device=device, dtype=dtype)


def constrain_pressure_logits(
    pressure_logits: torch.Tensor,
    support_mask: torch.Tensor,
    outside_logit: float = -20.0,
) -> torch.Tensor:
    """Force pressure probability to approximately zero outside the YOLO foot."""

    if support_mask.shape[-2:] != pressure_logits.shape[-2:]:
        support_mask = F.interpolate(
            support_mask.to(dtype=torch.float32),
            size=pressure_logits.shape[-2:],
            mode="nearest",
        )
    support_mask = support_mask.to(device=pressure_logits.device)
    if support_mask.shape[0] != pressure_logits.shape[0]:
        raise ValueError("support mask and pressure logits batch sizes differ")
    return torch.where(
        support_mask > 0.5,
        pressure_logits,
        torch.as_tensor(outside_logit, device=pressure_logits.device, dtype=pressure_logits.dtype),
    )


def restore_roi_map(
    crop_map: np.ndarray,
    roi: FootRoiResult,
    *,
    restrict_to_foot: bool,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Paste a crop-space 2D map into the detector's original image space."""

    values = np.asarray(crop_map, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"crop_map must resolve to 2D, got {values.shape}")
    x1, y1, x2, y2 = roi.bbox
    target_height, target_width = y2 - y1, x2 - x1
    if values.shape != (target_height, target_width):
        values = cv2.resize(values, (target_width, target_height), interpolation=interpolation)
    canvas = np.zeros(roi.original_shape, dtype=np.float32)
    canvas[y1:y2, x1:x2] = values
    if restrict_to_foot:
        canvas[roi.mask == 0] = 0.0
    return canvas


def foot_roi_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB audit overlay: green foot support and yellow boundary."""

    rgb = _as_rgb_uint8(image)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    overlay = rgb.astype(np.float32)
    green = np.zeros_like(overlay)
    green[..., 1] = 255.0
    selected = binary > 0
    overlay[selected] = 0.65 * overlay[selected] + 0.35 * green[selected]
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 220, 0), thickness=max(1, round(max(rgb.shape[:2]) / 400)))
    return np.clip(overlay, 0, 255).astype(np.uint8)


__all__ = [
    "FootRoiConfig",
    "FootRoiResult",
    "YoloFootRoiDetector",
    "bbox_from_mask",
    "constrain_pressure_logits",
    "foot_roi_overlay",
    "restore_roi_map",
    "support_mask_to_low_res",
]
