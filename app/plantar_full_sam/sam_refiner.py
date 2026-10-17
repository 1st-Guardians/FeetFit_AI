"""Deterministic local helpers for candidate-guided SAM mask refinement.

The functions in this module deliberately do not call OpenAI (or any other
network service). They prepare conservative local mask alternatives and run
an injected SAM predictor with deterministic prompts. Keeping this layer
local makes every pixel operation reproducible.

The expected mask convention is a two-dimensional array.  Floating point
masks in ``[0, 1]`` are thresholded at ``0.5``; conventional uint8 masks are
thresholded at half their range.  Every candidate and every selected SAM mask
is hard-clipped to the supplied whole-foot support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CANDIDATE_STRENGTHS = (0.35, 0.50, 0.65)
DEFAULT_CLASS_NAMES = ("cavus", "normal", "flat")


@dataclass(frozen=True)
class MaskCandidate:
    """One numbered, auditable mask offered to the external reviewer."""

    number: int
    name: str
    mask: np.ndarray
    method: str
    strength: float | None
    area_pixels: int
    area_ratio: float
    base_retention: float

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("mask")
        return payload


# A descriptive alias is convenient at integration call sites.
CandidateMask = MaskCandidate


@dataclass(frozen=True)
class SamOutputScore:
    """Deterministic measurements for one SAM output mask."""

    index: int
    sam_predicted_iou: float
    containment: float
    positive_prompt_recall: float
    negative_prompt_rejection: float
    prompt_consistency: float
    candidate_iou: float
    area_similarity: float
    raw_area_pixels: int
    enforced_area_pixels: int
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamMaskSelection:
    """Selected support-clipped SAM mask and the complete score audit."""

    mask: np.ndarray
    selected_index: int
    selected_score: float
    scores: tuple[SamOutputScore, ...]


@dataclass(frozen=True)
class MaskQC:
    """Quality-control measurements for a refined contact mask."""

    area_pixels: int
    support_pixels: int
    area_ratio: float
    base_area_ratio: float | None
    refined_to_base_area_ratio: float | None
    component_count: int
    largest_component_ratio: float
    hole_count: int
    hole_area_ratio: float
    outside_support_ratio: float
    positive_prompt_recall: float
    negative_prompt_rejection: float
    geometry_features: Mapping[str, float]
    geometry_probabilities: tuple[float, ...] | None
    classifier_probabilities: tuple[float, ...] | None
    geometry_label: str | None
    classifier_label: str | None
    probability_disagreement: float | None
    geometry_contradiction: bool
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry_features"] = dict(self.geometry_features)
        payload["reasons"] = list(self.reasons)
        if self.geometry_probabilities is not None:
            payload["geometry_probabilities"] = list(self.geometry_probabilities)
        if self.classifier_probabilities is not None:
            payload["classifier_probabilities"] = list(self.classifier_probabilities)
        return payload


MaskQualityControl = MaskQC


@dataclass(frozen=True)
class SamRefinementResult:
    """Result of one injected SAM predictor invocation."""

    mask: np.ndarray
    selected_index: int
    selected_score: float
    scores: tuple[SamOutputScore, ...]
    mask_input: np.ndarray
    point_coords: np.ndarray
    point_labels: np.ndarray
    qc: MaskQC


@dataclass(frozen=True)
class SamConsensusMember:
    """Audit row for one deterministic prompt-jitter SAM invocation."""

    run_index: int
    jitter_xy: tuple[float, float]
    selected_index: int
    selected_score: float
    sam_predicted_iou: float
    candidate_iou: float
    usable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamConfidenceConsensus:
    """Soft, support-clipped consensus from repeated SAM prompt runs.

    ``confident_negative`` is intentionally separate from
    ``confident_positive``.  Callers can suppress only stable negative pixels
    while preserving the previous mask in ``uncertain`` regions.  This avoids
    the destructive pattern of replacing a useful base mask with only the
    tiny set of pixels on which every SAM run is strongly positive.
    """

    mean_probability: np.ndarray
    std_probability: np.ndarray
    positive_consensus: np.ndarray
    negative_consensus: np.ndarray
    soft_confidence: np.ndarray
    confident_positive: np.ndarray
    confident_negative: np.ndarray
    uncertain: np.ndarray
    gated_mask: np.ndarray
    reference_mask: np.ndarray
    preserve_mask: np.ndarray
    foot_support: np.ndarray
    members: tuple[SamConsensusMember, ...]
    valid_member_count: int
    reliable: bool

    def audit_dict(self) -> dict[str, Any]:
        """Return compact JSON-safe diagnostics without serialising images."""

        support_pixels = max(int(self.foot_support.sum()), 1)
        return {
            "member_count": len(self.members),
            "valid_member_count": int(self.valid_member_count),
            "reliable": bool(self.reliable),
            "confident_positive_ratio": float(
                self.confident_positive.sum() / support_pixels
            ),
            "confident_negative_ratio": float(
                self.confident_negative.sum() / support_pixels
            ),
            "uncertain_ratio": float(self.uncertain.sum() / support_pixels),
            "preserve_area_ratio": float(self.preserve_mask.sum() / support_pixels),
            "gated_area_ratio": float(self.gated_mask.sum() / support_pixels),
            "members": [member.to_dict() for member in self.members],
        }


def _as_rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H, W, 3], got {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image_rgb must be numeric")
    if not bool(np.isfinite(image).all()):
        raise ValueError("image_rgb must be finite")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image.copy())
    values = image.astype(np.float32, copy=False)
    if values.size and float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        values = values * 255.0
    return np.ascontiguousarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))


def _as_2d_values(mask: np.ndarray, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    values = np.asarray(mask)
    while values.ndim > 2 and 1 in values.shape:
        axis = next(index for index, size in enumerate(values.shape) if size == 1)
        values = np.squeeze(values, axis=axis)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-D mask, got {values.shape}")
    if shape is not None and tuple(values.shape) != tuple(shape):
        raise ValueError(f"{name} shape {values.shape} does not match {shape}")
    if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
        raise ValueError(f"{name} must be numeric or boolean")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")
    return values


def _as_binary(
    mask: np.ndarray,
    name: str,
    shape: tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    values = _as_2d_values(mask, name, shape)
    if values.dtype == np.bool_:
        binary = values.copy()
    elif values.size and float(values.max()) > 1.0:
        binary = values.astype(np.float32) >= 127.5
    else:
        binary = values.astype(np.float32) >= float(threshold)
    return np.ascontiguousarray(binary, dtype=np.bool_)


def _normalised_probability(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = _as_2d_values(mask, "base_mask", shape).astype(np.float32)
    if values.size and float(values.max()) > 1.0:
        values = values / 255.0
    return np.ascontiguousarray(np.clip(values, 0.0, 1.0))


def _odd_kernel_size(shape: tuple[int, int], fraction: float) -> int:
    if fraction <= 0:
        raise ValueError("opening_kernel_fraction must be positive")
    size = max(3, int(round(min(shape) * float(fraction))))
    if size % 2 == 0:
        size += 1
    # Avoid a nonsensical kernel on tiny crops.
    maximum = max(1, min(shape))
    if maximum % 2 == 0:
        maximum -= 1
    return max(1, min(size, maximum))


def _robust_unit(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[region]
    if not selected.size:
        return np.zeros_like(values, dtype=np.float32)
    lower, upper = np.percentile(selected, (5.0, 95.0))
    if float(upper - lower) <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values.astype(np.float32) - float(lower)) / float(upper - lower), 0.0, 1.0)


def _colour_features(image: np.ndarray, colour_space: str) -> tuple[np.ndarray, np.ndarray]:
    """Return colour feature vectors and a generic contact-evidence fallback."""

    if colour_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        features = converted / 255.0
        chroma = np.sqrt(
            (converted[..., 1] - 128.0) ** 2 + (converted[..., 2] - 128.0) ** 2
        )
        fallback = chroma
    elif colour_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        angle = converted[..., 0] * (2.0 * np.pi / 180.0)
        saturation = converted[..., 1] / 255.0
        value = converted[..., 2] / 255.0
        features = np.stack(
            (
                np.cos(angle) * saturation,
                np.sin(angle) * saturation,
                saturation,
                value,
            ),
            axis=-1,
        ).astype(np.float32)
        fallback = saturation
    else:
        raise ValueError(f"Unsupported colour space: {colour_space}")
    return features, fallback.astype(np.float32)


def _visual_contact_score(
    image: np.ndarray,
    base: np.ndarray,
    support: np.ndarray,
    colour_space: str,
    kernel: np.ndarray,
) -> np.ndarray:
    features, generic_fallback = _colour_features(image, colour_space)
    base_u8 = base.astype(np.uint8)
    core = cv2.erode(base_u8, kernel, iterations=1) > 0
    if int(core.sum()) < max(4, int(base.sum() * 0.02)):
        core = base
    expanded = cv2.dilate(base_u8, kernel, iterations=1) > 0
    reference = support & ~expanded
    if int(reference.sum()) < max(8, int(support.sum() * 0.01)):
        reference = support & ~base

    contact_values = features[core]
    if contact_values.size:
        contact_centre = np.median(contact_values, axis=0)
        scale_values = features[support]
        feature_scale = np.percentile(scale_values, 75.0, axis=0) - np.percentile(
            scale_values, 25.0, axis=0
        )
        feature_scale = np.maximum(feature_scale, 0.04)
        distance_contact = np.sqrt(
            np.sum(((features - contact_centre) / feature_scale) ** 2, axis=-1)
        )
    else:
        distance_contact = np.ones(support.shape, dtype=np.float32)

    if int(reference.sum()) >= 4:
        reference_centre = np.median(features[reference], axis=0)
        distance_reference = np.sqrt(
            np.sum(((features - reference_centre) / feature_scale) ** 2, axis=-1)
        )
        prototype_score = distance_reference / (
            distance_contact + distance_reference + 1e-6
        )
    else:
        prototype_score = _robust_unit(generic_fallback, support)

    # A small gradient contribution preserves visually supported boundaries;
    # the colour prototype remains dominant so thin glass highlights cannot
    # become a complete contact region by themselves.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    edge_score = _robust_unit(gradient, support)
    score = 0.88 * prototype_score.astype(np.float32) + 0.12 * edge_score
    score[~support] = 0.0
    return np.ascontiguousarray(np.clip(score, 0.0, 1.0), dtype=np.float32)


def _seeded_visual_intersection(
    score: np.ndarray,
    base: np.ndarray,
    support: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Grow colour-supported components from conservative base-mask seeds."""

    allowed = support & (score >= float(strength))
    seed_threshold = min(0.98, float(strength) + 0.15)
    seeds = base & (score >= seed_threshold)
    if not np.any(seeds) and np.any(base):
        # Always retain a real base pixel as a seed, even in an uninformative
        # monochrome crop.  This avoids manufacturing evidence outside S0.
        base_scores = score[base]
        cutoff = float(np.percentile(base_scores, 90.0))
        seeds = base & (score >= cutoff)
    allowed |= seeds
    count, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(base)
    if count > 1:
        seed_labels = np.unique(labels[seeds])
        seed_labels = seed_labels[seed_labels != 0]
        if len(seed_labels):
            kept = np.isin(labels, seed_labels)
    # Intersecting last is the conservative guarantee: candidates can remove
    # questionable S0 pixels but can never add a new contact pixel.
    return np.ascontiguousarray(kept & base & support)


def _make_candidate(
    number: int,
    name: str,
    mask: np.ndarray,
    method: str,
    strength: float | None,
    support_pixels: int,
    base_pixels: int,
) -> MaskCandidate:
    binary = np.ascontiguousarray(mask, dtype=np.bool_)
    pixels = int(binary.sum())
    return MaskCandidate(
        number=int(number),
        name=str(name),
        mask=binary,
        method=str(method),
        strength=None if strength is None else float(strength),
        area_pixels=pixels,
        area_ratio=float(pixels / max(support_pixels, 1)),
        base_retention=float(pixels / max(base_pixels, 1)),
    )


def generate_candidate_masks(
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    foot_support: np.ndarray,
    *,
    strengths: Sequence[float] = DEFAULT_CANDIDATE_STRENGTHS,
    mask_threshold: float = 0.5,
    opening_kernel_fraction: float = 0.025,
) -> list[MaskCandidate]:
    """Generate conservative base, opened, Lab, HSV, and joint candidates.

    For every requested strength three evidence candidates are appended in a
    stable order: Lab-seeded intersection, HSV-seeded intersection, and their
    intersection.  Thus candidate numbering is reproducible and suitable for
    a structured external-review response.
    """

    image = _as_rgb_uint8(image_rgb)
    shape = tuple(image.shape[:2])
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("mask_threshold must be in [0, 1]")
    support = _as_binary(foot_support, "foot_support", shape, threshold=mask_threshold)
    if not np.any(support):
        raise ValueError("foot_support is empty")
    probability = _normalised_probability(base_mask, shape)
    base = (probability >= float(mask_threshold)) & support
    if not np.any(base):
        raise ValueError("base_mask is empty inside foot_support")

    requested_strengths = tuple(float(value) for value in strengths)
    if not requested_strengths:
        raise ValueError("strengths must contain at least one value")
    if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in requested_strengths):
        raise ValueError("candidate strengths must be finite values in (0, 1)")
    if any(
        requested_strengths[index] >= requested_strengths[index + 1]
        for index in range(len(requested_strengths) - 1)
    ):
        raise ValueError("candidate strengths must be strictly increasing")

    kernel_size = _odd_kernel_size(shape, float(opening_kernel_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    opened &= support
    # A tiny crop or thin true contact can be completely erased by opening.
    # In that exceptional case base is the safer candidate.
    if not np.any(opened):
        opened = base.copy()

    support_pixels = int(support.sum())
    base_pixels = int(base.sum())
    candidates: list[MaskCandidate] = [
        _make_candidate(1, "base", base, "base", None, support_pixels, base_pixels),
        _make_candidate(
            2,
            "morphology_open",
            opened,
            "morphology_open",
            None,
            support_pixels,
            base_pixels,
        ),
    ]
    lab_score = _visual_contact_score(image, base, support, "lab", kernel)
    hsv_score = _visual_contact_score(image, base, support, "hsv", kernel)
    number = 3
    for strength in requested_strengths:
        suffix = f"{int(round(100 * strength)):02d}"
        lab = _seeded_visual_intersection(lab_score, base, support, strength)
        hsv = _seeded_visual_intersection(hsv_score, base, support, strength)
        joint = lab & hsv
        candidates.extend(
            (
                _make_candidate(
                    number,
                    f"lab_seeded_intersection_s{suffix}",
                    lab,
                    "lab_seeded_intersection",
                    strength,
                    support_pixels,
                    base_pixels,
                ),
                _make_candidate(
                    number + 1,
                    f"hsv_seeded_intersection_s{suffix}",
                    hsv,
                    "hsv_seeded_intersection",
                    strength,
                    support_pixels,
                    base_pixels,
                ),
                _make_candidate(
                    number + 2,
                    f"lab_hsv_intersection_s{suffix}",
                    joint,
                    "lab_hsv_intersection",
                    strength,
                    support_pixels,
                    base_pixels,
                ),
            )
        )
        number += 3
    return candidates


# Alternate verb form retained for simple integration code.
generate_conservative_candidate_masks = generate_candidate_masks


def _candidate_parts(candidate: MaskCandidate | np.ndarray, index: int) -> tuple[int, str, np.ndarray]:
    if isinstance(candidate, MaskCandidate):
        return candidate.number, candidate.name, candidate.mask
    return index + 1, f"candidate_{index + 1}", np.asarray(candidate)


def _candidate_review_label(
    candidate: MaskCandidate | np.ndarray,
    number: int,
    name: str,
    mask: np.ndarray,
) -> str:
    """Return an unambiguous two-line area label for the review board.

    ``MaskCandidate.area_ratio`` is measured against the detected whole-foot
    support, which is the clinically useful denominator for a contact mask.
    A bare ndarray has no support metadata, so only that fallback reports its
    fraction of the complete image crop.
    """

    if isinstance(candidate, MaskCandidate):
        return (
            f"#{number}  {name}\n"
            f"foot={100.0 * candidate.area_ratio:.1f}%, "
            f"retain={100.0 * candidate.base_retention:.1f}%"
        )
    image_percent = 100.0 * float(mask.sum()) / max(int(mask.size), 1)
    return f"#{number}  {name}\nimage={image_percent:.1f}%"


def render_candidate_review_board(
    image_rgb: np.ndarray,
    candidates: Sequence[MaskCandidate | np.ndarray],
    *,
    columns: int = 3,
    tile_size: tuple[int, int] = (320, 320),
    overlay_alpha: float = 0.48,
    overlay_colour: tuple[int, int, int] = (255, 45, 35),
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Render RGB overlays with unambiguous candidate numbers.

    The returned board is an RGB uint8 array.  ``output_path`` is optional and
    exists only as a convenience for the caller; no path is touched by
    default.
    """

    image = _as_rgb_uint8(image_rgb)
    if not candidates:
        raise ValueError("candidates must not be empty")
    if int(columns) < 1:
        raise ValueError("columns must be positive")
    tile_width, tile_height = int(tile_size[0]), int(tile_size[1])
    if tile_width < 80 or tile_height < 80:
        raise ValueError("tile_size values must be at least 80 pixels")
    if not 0.0 <= float(overlay_alpha) <= 1.0:
        raise ValueError("overlay_alpha must be in [0, 1]")
    if len(overlay_colour) != 3 or any(not 0 <= int(value) <= 255 for value in overlay_colour):
        raise ValueError("overlay_colour must be an RGB triplet")

    header_height = max(32, int(round(tile_height * 0.12)))
    content_height = tile_height - header_height
    rows = (len(candidates) + int(columns) - 1) // int(columns)
    board = Image.new(
        "RGB",
        (int(columns) * tile_width, rows * tile_height),
        color=(24, 27, 31),
    )
    try:
        font = ImageFont.load_default(size=max(12, header_height // 3))
    except TypeError:  # Pillow < 10 compatibility.
        font = ImageFont.load_default()

    for index, candidate in enumerate(candidates):
        number, name, raw_mask = _candidate_parts(candidate, index)
        mask = _as_binary(raw_mask, f"candidate {number}", tuple(image.shape[:2]))
        base_image = Image.fromarray(image, mode="RGB")
        scale = min(tile_width / image.shape[1], content_height / image.shape[0])
        resized_width = max(1, int(round(image.shape[1] * scale)))
        resized_height = max(1, int(round(image.shape[0] * scale)))
        resized_image = base_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (resized_width, resized_height), Image.Resampling.NEAREST
        )
        colour_layer = Image.new("RGB", resized_image.size, tuple(map(int, overlay_colour)))
        alpha_mask = resized_mask.point(lambda value: int(round(value * float(overlay_alpha))))
        overlay = Image.composite(colour_layer, resized_image, alpha_mask)

        tile = Image.new("RGB", (tile_width, tile_height), color=(10, 12, 14))
        offset_x = (tile_width - resized_width) // 2
        offset_y = header_height + (content_height - resized_height) // 2
        tile.paste(overlay, (offset_x, offset_y))
        draw = ImageDraw.Draw(tile)
        label = _candidate_review_label(candidate, number, name, mask)
        draw.rectangle((0, 0, tile_width, header_height - 1), fill=(35, 39, 45))
        draw.multiline_text(
            (8, 2),
            label,
            fill=(255, 255, 255),
            font=font,
            spacing=1,
        )
        column = index % int(columns)
        row = index // int(columns)
        board.paste(tile, (column * tile_width, row * tile_height))

    output = np.ascontiguousarray(np.asarray(board), dtype=np.uint8)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        board.save(destination)
    return output


render_review_board = render_candidate_review_board


def _point_pairs(points: Sequence[Any] | np.ndarray | None, name: str) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    if isinstance(points, np.ndarray) and points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    parsed: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            if "x" not in point or "y" not in point:
                raise ValueError(f"Each {name} mapping must contain x and y")
            x, y = point["x"], point["y"]
        else:
            if len(point) != 2:
                raise ValueError(f"Each {name} point must contain exactly x and y")
            x, y = point[0], point[1]
        parsed.append((float(x), float(y)))
    output = np.asarray(parsed, dtype=np.float32).reshape(-1, 2)
    if not bool(np.isfinite(output).all()):
        raise ValueError(f"{name} points must be finite")
    return output


def normalized_points_to_crop(
    points: Sequence[Any] | np.ndarray | None,
    crop_shape: tuple[int, int] | Sequence[int] | np.ndarray,
    *,
    clip: bool = True,
    normalized_scale: float = 1.0,
) -> np.ndarray:
    """Map normalised reviewer coordinates to crop-local XY pixels.

    The structured review schema uses ``[0, 1]`` coordinates, hence the
    default ``normalized_scale=1.0``.  A legacy 0..1000 response can be mapped
    explicitly with ``normalized_scale=1000.0``.
    """

    normalised = _point_pairs(points, "normalised")
    if isinstance(crop_shape, np.ndarray):
        height, width = crop_shape.shape[:2]
    else:
        dimensions = tuple(int(value) for value in crop_shape)
        if len(dimensions) < 2:
            raise ValueError("crop_shape must contain height and width")
        height, width = dimensions[0], dimensions[1]
    if height <= 0 or width <= 0:
        raise ValueError("crop dimensions must be positive")
    if not np.isfinite(normalized_scale) or float(normalized_scale) <= 0.0:
        raise ValueError("normalized_scale must be positive and finite")
    scale = float(normalized_scale)
    if not clip and bool(((normalised < 0.0) | (normalised > scale)).any()):
        raise ValueError(f"normalised points must be in [0, {scale:g}]")
    normalised = np.clip(normalised, 0.0, scale)
    output = normalised.copy()
    output[:, 0] *= float(max(width - 1, 0)) / scale
    output[:, 1] *= float(max(height - 1, 0)) / scale
    return np.ascontiguousarray(output, dtype=np.float32)


map_normalized_points_to_crop = normalized_points_to_crop


def enforce_foot_support(mask: np.ndarray, foot_support: np.ndarray) -> np.ndarray:
    """Hard-clip a binary mask to its anatomically valid support."""

    binary = _as_binary(mask, "mask")
    support = _as_binary(foot_support, "foot_support", tuple(binary.shape))
    return np.ascontiguousarray(binary & support)


def candidate_to_sam_mask_input(
    candidate_mask: np.ndarray,
    *,
    mask_input_size: tuple[int, int] = (256, 256),
    logit_scale: float = 8.0,
) -> np.ndarray:
    """Convert a full-resolution candidate to SAM's low-resolution logits."""

    candidate = _as_binary(candidate_mask, "candidate_mask")
    height, width = int(mask_input_size[0]), int(mask_input_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("mask_input_size dimensions must be positive")
    if not np.isfinite(logit_scale) or float(logit_scale) <= 0:
        raise ValueError("logit_scale must be positive and finite")
    resized = cv2.resize(
        candidate.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    logits = (2.0 * resized - 1.0) * float(logit_scale)
    return np.ascontiguousarray(logits[None, ...], dtype=np.float32)


def _candidate_to_predictor_mask_input(
    candidate_mask: np.ndarray,
    predictor: Any,
    mask_input_size: tuple[int, int],
    *,
    logit_scale: float = 8.0,
) -> np.ndarray:
    """Map a candidate through SAM's resize-and-pad coordinate system.

    A standard :class:`segment_anything.SamPredictor` resizes an image so its
    long edge equals ``image_encoder.img_size`` and pads the bottom/right to a
    square. Dense mask prompts live in that padded coordinate system. Directly
    stretching a non-square crop to 256x256 moves its prompt relative to the
    image embedding and can make every predicted mask miss the foot support.

    Lightweight injected test predictors do not necessarily expose the SAM
    geometry attributes; those retain the public helper's legacy direct-resize
    behaviour.
    """

    candidate = _as_binary(candidate_mask, "candidate_mask")
    try:
        original_size = tuple(int(value) for value in predictor.original_size)
        input_size = tuple(int(value) for value in predictor.input_size)
        encoder_size = int(predictor.model.image_encoder.img_size)
        # ``transform`` is part of the public SamPredictor implementation and
        # distinguishes it from small duck-typed predictors used in tests.
        predictor.transform
    except (AttributeError, TypeError, ValueError):
        return candidate_to_sam_mask_input(
            candidate,
            mask_input_size=mask_input_size,
            logit_scale=logit_scale,
        )

    if original_size != tuple(candidate.shape):
        raise ValueError(
            "predictor original_size does not match the candidate mask shape"
        )
    if (
        len(input_size) != 2
        or min(input_size) <= 0
        or max(input_size) > encoder_size
        or encoder_size <= 0
    ):
        raise ValueError("predictor exposes invalid SAM resize geometry")

    resized = cv2.resize(
        candidate.astype(np.float32),
        (input_size[1], input_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.zeros((encoder_size, encoder_size), dtype=np.float32)
    padded[: input_size[0], : input_size[1]] = resized
    low_height, low_width = int(mask_input_size[0]), int(mask_input_size[1])
    if low_height <= 0 or low_width <= 0:
        raise ValueError("mask_input_size dimensions must be positive")
    low_resolution = cv2.resize(
        padded,
        (low_width, low_height),
        interpolation=cv2.INTER_LINEAR,
    )
    logits = (2.0 * low_resolution - 1.0) * float(logit_scale)
    return np.ascontiguousarray(logits[None, ...], dtype=np.float32)


def _deterministic_interior_positive_points(
    candidate_mask: np.ndarray,
    *,
    maximum_points: int = 3,
    minimum_component_ratio: float = 0.01,
) -> np.ndarray:
    """Choose stable foreground prompts strictly inside a candidate mask.

    The deepest point of each sufficiently large connected component is used,
    largest component first. This is deterministic, cannot introduce evidence
    outside the GPT-selected candidate, and prevents mask-only SAM prompting
    from treating the empty mask as an equally consistent answer.
    """

    candidate = _as_binary(candidate_mask, "candidate_mask")
    if not np.any(candidate):
        raise ValueError("candidate_mask is empty")
    if int(maximum_points) < 1:
        raise ValueError("maximum_points must be positive")
    if not 0.0 <= float(minimum_component_ratio) <= 1.0:
        raise ValueError("minimum_component_ratio must be in [0, 1]")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    total = int(candidate.sum())
    components = sorted(
        (
            (int(stats[label, cv2.CC_STAT_AREA]), int(label))
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA])
            >= max(1, int(np.ceil(total * float(minimum_component_ratio))))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not components:
        components = [
            max(
                (
                    (int(stats[label, cv2.CC_STAT_AREA]), int(label))
                    for label in range(1, count)
                ),
                key=lambda item: (item[0], -item[1]),
            )
        ]

    points: list[tuple[float, float]] = []
    for _, label in components[: int(maximum_points)]:
        component = labels == label
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        flat_index = int(np.argmax(distance))
        y, x = np.unravel_index(flat_index, component.shape)
        # ``argmax`` is row-major and therefore resolves plateaus identically
        # across runs. The distance maximum is guaranteed to be foreground.
        points.append((float(x), float(y)))
    return np.ascontiguousarray(points, dtype=np.float32)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def _area_similarity(first_area: int, second_area: int) -> float:
    if first_area == 0 and second_area == 0:
        return 1.0
    if first_area <= 0 or second_area <= 0:
        return 0.0
    return float(min(first_area, second_area) / max(first_area, second_area))


def _prompt_measurements(
    mask: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
) -> tuple[float, float, float]:
    height, width = mask.shape

    def values_at(points: np.ndarray) -> np.ndarray:
        if not len(points):
            return np.empty((0,), dtype=np.bool_)
        xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
        return mask[ys, xs]

    positive_values = values_at(positive_points)
    negative_values = values_at(negative_points)
    positive_recall = float(positive_values.mean()) if len(positive_values) else 1.0
    negative_rejection = float((~negative_values).mean()) if len(negative_values) else 1.0
    prompt_count = len(positive_values) + len(negative_values)
    if prompt_count:
        consistent = int(positive_values.sum()) + int((~negative_values).sum())
        prompt_consistency = float(consistent / prompt_count)
    else:
        prompt_consistency = 1.0
    return positive_recall, negative_rejection, prompt_consistency


def _sam_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = _as_2d_values(mask, "SAM output", shape)
    if values.dtype == np.bool_:
        return np.ascontiguousarray(values.copy())
    numeric = values.astype(np.float32)
    threshold = 0.0 if (float(numeric.min()) < 0.0 or float(numeric.max()) > 1.0) else 0.5
    return np.ascontiguousarray(numeric > threshold)


def select_best_sam_mask(
    masks: np.ndarray | Sequence[np.ndarray],
    predicted_ious: Sequence[float] | np.ndarray,
    candidate_mask: np.ndarray,
    foot_support: np.ndarray,
    *,
    positive_points: Sequence[Any] | np.ndarray | None = None,
    negative_points: Sequence[Any] | np.ndarray | None = None,
) -> SamMaskSelection:
    """Rank SAM outputs with containment, prompts, IoU, and area stability.

    Score weights are intentionally fixed.  This makes selection auditable and
    ensures a high SAM self-score cannot compensate for spilling beyond the
    foot support or violating reviewer prompts.
    """

    reference = _as_binary(candidate_mask, "candidate_mask")
    support = _as_binary(foot_support, "foot_support", tuple(reference.shape))
    reference = reference & support
    if not np.any(reference):
        raise ValueError("candidate_mask is empty inside foot_support")
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3 or tuple(mask_array.shape[1:]) != tuple(reference.shape):
        raise ValueError(
            f"masks must have shape [K, {reference.shape[0]}, {reference.shape[1]}]"
        )
    if mask_array.shape[0] == 0:
        raise ValueError("SAM returned no masks")
    sam_ious = np.asarray(predicted_ious, dtype=np.float32).reshape(-1)
    if len(sam_ious) != mask_array.shape[0]:
        raise ValueError("predicted_ious length must equal the number of masks")
    if not bool(np.isfinite(sam_ious).all()):
        raise ValueError("predicted_ious must be finite")
    sam_ious = np.clip(sam_ious, 0.0, 1.0)
    positive = _point_pairs(positive_points, "positive")
    negative = _point_pairs(negative_points, "negative")
    reference_area = int(reference.sum())

    score_rows: list[SamOutputScore] = []
    enforced_masks: list[np.ndarray] = []
    for index in range(mask_array.shape[0]):
        raw = _sam_binary(mask_array[index], tuple(reference.shape))
        raw_area = int(raw.sum())
        contained_pixels = int((raw & support).sum())
        containment = float(contained_pixels / max(raw_area, 1)) if raw_area else 0.0
        enforced = np.ascontiguousarray(raw & support)
        positive_recall, negative_rejection, prompt_consistency = _prompt_measurements(
            enforced, positive, negative
        )
        candidate_iou = _mask_iou(enforced, reference)
        area_similarity = _area_similarity(int(enforced.sum()), reference_area)
        total = (
            0.25 * containment
            + 0.25 * prompt_consistency
            + 0.20 * candidate_iou
            + 0.20 * float(sam_ious[index])
            + 0.10 * area_similarity
        )
        score_rows.append(
            SamOutputScore(
                index=index,
                sam_predicted_iou=float(sam_ious[index]),
                containment=containment,
                positive_prompt_recall=positive_recall,
                negative_prompt_rejection=negative_rejection,
                prompt_consistency=prompt_consistency,
                candidate_iou=candidate_iou,
                area_similarity=area_similarity,
                raw_area_pixels=raw_area,
                enforced_area_pixels=int(enforced.sum()),
                total_score=float(total),
            )
        )
        enforced_masks.append(enforced)

    # The negative index is the final key, making exact ties choose the first
    # SAM output across platforms and NumPy versions.
    best = max(
        range(len(score_rows)),
        key=lambda index: (
            score_rows[index].total_score,
            score_rows[index].prompt_consistency,
            score_rows[index].containment,
            score_rows[index].candidate_iou,
            -index,
        ),
    )
    return SamMaskSelection(
        mask=enforced_masks[best],
        selected_index=best,
        selected_score=score_rows[best].total_score,
        scores=tuple(score_rows),
    )


select_sam_output = select_best_sam_mask


def _component_statistics(mask: np.ndarray) -> tuple[int, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty((0,), dtype=np.int32)
    total = int(mask.sum())
    largest = int(areas.max()) if len(areas) else 0
    return int(count - 1), float(largest / max(total, 1))


def _hole_statistics(mask: np.ndarray) -> tuple[int, float]:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8)
    exterior = int(labels[0, 0])
    hole_areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if label != exterior
    ]
    return len(hole_areas), float(sum(hole_areas) / max(int(mask.sum()), 1))


def _region_contact_ratio(mask: np.ndarray, region: np.ndarray) -> float:
    pixels = int(region.sum())
    return float((mask & region).sum() / pixels) if pixels else 0.0


def _geometry_features(mask: np.ndarray, support: np.ndarray) -> dict[str, float]:
    """Rotation/mirror-tolerant inputs for downstream geometry checks."""

    ys, xs = np.nonzero(support)
    if len(xs) < 3:
        return {
            "forefoot_contact_ratio": 0.0,
            "midfoot_contact_ratio": 0.0,
            "heel_contact_ratio": 0.0,
            "midfoot_min_side_contact_ratio": 0.0,
            "midfoot_max_side_contact_ratio": 0.0,
            "arch_side_contrast": 0.0,
            "long_axis_anisotropy": 0.0,
        }
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    centre = coordinates.mean(axis=0)
    centred = coordinates - centre
    covariance = centred.T @ centred / max(len(coordinates), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_vector = eigenvectors[:, order[0]]
    minor_vector = eigenvectors[:, order[1]]
    major = centred @ major_vector
    minor = centred @ minor_vector
    low, high = np.percentile(major, (3.0, 97.0))
    span = max(float(high - low), 1e-6)

    # Orient the major axis so the wider end is treated as forefoot.
    end_fraction = 0.18
    low_end = major <= low + end_fraction * span
    high_end = major >= high - end_fraction * span

    def end_width(selector: np.ndarray) -> float:
        values = minor[selector]
        return float(np.percentile(values, 90.0) - np.percentile(values, 10.0)) if len(values) else 0.0

    if end_width(high_end) >= end_width(low_end):
        distal_values = (major - low) / span
    else:
        distal_values = (high - major) / span
    distal_values = np.clip(distal_values, 0.0, 1.0)
    distal_map = np.zeros(support.shape, dtype=np.float32)
    minor_map = np.zeros(support.shape, dtype=np.float32)
    distal_map[ys, xs] = distal_values.astype(np.float32)
    minor_map[ys, xs] = minor.astype(np.float32)
    forefoot = support & (distal_map >= 0.68)
    midfoot = support & (distal_map >= 0.30) & (distal_map < 0.68)
    heel = support & (distal_map < 0.30)
    mid_negative = midfoot & (minor_map < 0.0)
    mid_positive = midfoot & (minor_map >= 0.0)
    negative_ratio = _region_contact_ratio(mask, mid_negative)
    positive_ratio = _region_contact_ratio(mask, mid_positive)
    eigen_major = max(float(eigenvalues[order[0]]), 1e-9)
    eigen_minor = max(float(eigenvalues[order[1]]), 0.0)
    anisotropy = (eigen_major - eigen_minor) / eigen_major
    return {
        "forefoot_contact_ratio": _region_contact_ratio(mask, forefoot),
        "midfoot_contact_ratio": _region_contact_ratio(mask, midfoot),
        "heel_contact_ratio": _region_contact_ratio(mask, heel),
        "midfoot_min_side_contact_ratio": min(negative_ratio, positive_ratio),
        "midfoot_max_side_contact_ratio": max(negative_ratio, positive_ratio),
        "arch_side_contrast": abs(negative_ratio - positive_ratio),
        "long_axis_anisotropy": float(np.clip(anisotropy, 0.0, 1.0)),
    }


def _normalise_probabilities(
    probabilities: Sequence[float] | np.ndarray | None,
    name: str,
    expected_length: int,
) -> tuple[float, ...] | None:
    if probabilities is None:
        return None
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} class values")
    if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError(f"{name} must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return tuple(float(value) for value in values / total)


def compute_mask_qc(
    mask: np.ndarray,
    foot_support: np.ndarray,
    *,
    base_mask: np.ndarray | None = None,
    positive_points: Sequence[Any] | np.ndarray | None = None,
    negative_points: Sequence[Any] | np.ndarray | None = None,
    geometry_probabilities: Sequence[float] | np.ndarray | None = None,
    classifier_probabilities: Sequence[float] | np.ndarray | None = None,
    geometry_label: str | None = None,
    classifier_label: str | None = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    contradiction_confidence: float = 0.70,
    minimum_area_ratio: float = 0.02,
    maximum_area_ratio: float = 0.75,
    maximum_components: int = 12,
    maximum_hole_area_ratio: float = 0.12,
) -> MaskQC:
    """Compute shape QC and explicit geometry/classifier contradiction data."""

    raw = _as_binary(mask, "mask")
    support = _as_binary(foot_support, "foot_support", tuple(raw.shape))
    if not np.any(support):
        raise ValueError("foot_support is empty")
    if not 0.0 <= minimum_area_ratio <= maximum_area_ratio <= 1.0:
        raise ValueError("area ratio bounds must satisfy 0 <= minimum <= maximum <= 1")
    if int(maximum_components) < 1:
        raise ValueError("maximum_components must be positive")
    if not 0.0 <= maximum_hole_area_ratio <= 1.0:
        raise ValueError("maximum_hole_area_ratio must be in [0, 1]")
    if not 0.0 <= contradiction_confidence <= 1.0:
        raise ValueError("contradiction_confidence must be in [0, 1]")
    names = tuple(str(value) for value in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names")

    raw_pixels = int(raw.sum())
    outside_pixels = int((raw & ~support).sum())
    outside_ratio = float(outside_pixels / max(raw_pixels, 1))
    effective = raw & support
    area_pixels = int(effective.sum())
    support_pixels = int(support.sum())
    area_ratio = float(area_pixels / support_pixels)
    component_count, largest_component_ratio = _component_statistics(effective)
    hole_count, hole_area_ratio = _hole_statistics(effective)
    positive = _point_pairs(positive_points, "positive")
    negative = _point_pairs(negative_points, "negative")
    positive_recall, negative_rejection, _ = _prompt_measurements(effective, positive, negative)

    base_area_ratio: float | None = None
    relative_area: float | None = None
    if base_mask is not None:
        base = _as_binary(base_mask, "base_mask", tuple(raw.shape)) & support
        base_pixels = int(base.sum())
        base_area_ratio = float(base_pixels / support_pixels)
        relative_area = float(area_pixels / max(base_pixels, 1))

    geometry_values = _normalise_probabilities(
        geometry_probabilities, "geometry_probabilities", len(names)
    )
    classifier_values = _normalise_probabilities(
        classifier_probabilities, "classifier_probabilities", len(names)
    )
    inferred_geometry_label = geometry_label
    inferred_classifier_label = classifier_label
    disagreement: float | None = None
    contradiction = False
    if geometry_values is not None:
        inferred_geometry_label = names[int(np.argmax(geometry_values))]
    if classifier_values is not None:
        inferred_classifier_label = names[int(np.argmax(classifier_values))]
    if geometry_values is not None and classifier_values is not None:
        disagreement = float(
            0.5
            * np.abs(np.asarray(geometry_values) - np.asarray(classifier_values)).sum()
        )
        contradiction = (
            inferred_geometry_label != inferred_classifier_label
            and max(geometry_values) >= float(contradiction_confidence)
            and max(classifier_values) >= float(contradiction_confidence)
        )
    elif inferred_geometry_label is not None and inferred_classifier_label is not None:
        contradiction = inferred_geometry_label != inferred_classifier_label

    reasons: list[str] = []
    if area_pixels == 0:
        reasons.append("empty_mask")
    if area_ratio < float(minimum_area_ratio):
        reasons.append("area_ratio_below_minimum")
    if area_ratio > float(maximum_area_ratio):
        reasons.append("area_ratio_above_maximum")
    if component_count > int(maximum_components):
        reasons.append("component_count_above_maximum")
    if hole_area_ratio > float(maximum_hole_area_ratio):
        reasons.append("hole_area_ratio_above_maximum")
    if outside_pixels:
        reasons.append("pixels_outside_foot_support")
    if positive_recall < 1.0:
        reasons.append("positive_prompt_missed")
    if negative_rejection < 1.0:
        reasons.append("negative_prompt_violated")
    if contradiction:
        reasons.append("geometry_classifier_contradiction")

    return MaskQC(
        area_pixels=area_pixels,
        support_pixels=support_pixels,
        area_ratio=area_ratio,
        base_area_ratio=base_area_ratio,
        refined_to_base_area_ratio=relative_area,
        component_count=component_count,
        largest_component_ratio=largest_component_ratio,
        hole_count=hole_count,
        hole_area_ratio=hole_area_ratio,
        outside_support_ratio=outside_ratio,
        positive_prompt_recall=positive_recall,
        negative_prompt_rejection=negative_rejection,
        geometry_features=_geometry_features(effective, support),
        geometry_probabilities=geometry_values,
        classifier_probabilities=classifier_values,
        geometry_label=inferred_geometry_label,
        classifier_label=inferred_classifier_label,
        probability_disagreement=disagreement,
        geometry_contradiction=bool(contradiction),
        passed=not reasons,
        reasons=tuple(reasons),
    )


compute_refinement_qc = compute_mask_qc


def _predictor_mask_input_size(predictor: Any) -> tuple[int, int]:
    direct = getattr(predictor, "mask_input_size", None)
    if direct is not None:
        values = tuple(int(value) for value in direct)
        if len(values) == 2:
            return values
    try:
        values = tuple(int(value) for value in predictor.model.prompt_encoder.mask_input_size)
        if len(values) == 2:
            return values
    except (AttributeError, TypeError):
        pass
    return 256, 256


DEFAULT_CONSENSUS_JITTER_OFFSETS = (
    (0.0, 0.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
)


def _sam_logits_to_probability(logits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert one high-resolution SAM output to a stable probability map."""

    values = _as_2d_values(logits, "SAM high-resolution logits", shape).astype(
        np.float32
    )
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    if minimum >= 0.0 and maximum <= 1.0:
        probability = values
    else:
        # Clipping prevents overflow without changing meaningful SAM logits.
        bounded = np.clip(values, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-bounded))
    return np.ascontiguousarray(np.clip(probability, 0.0, 1.0), dtype=np.float32)


def _jitter_prompt_points(
    points: np.ndarray,
    offset_xy: tuple[float, float],
    shape: tuple[int, int],
    *,
    positive_support: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a deterministic crop-pixel offset and keep prompts valid."""

    original = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(original):
        return np.empty((0, 2), dtype=np.float32)
    shifted = original.copy()
    shifted[:, 0] += float(offset_xy[0])
    shifted[:, 1] += float(offset_xy[1])
    shifted[:, 0] = np.clip(shifted[:, 0], 0.0, float(shape[1] - 1))
    shifted[:, 1] = np.clip(shifted[:, 1], 0.0, float(shape[0] - 1))
    if positive_support is not None:
        support = _as_binary(positive_support, "positive_support", shape)
        xs = np.rint(shifted[:, 0]).astype(np.int64)
        ys = np.rint(shifted[:, 1]).astype(np.int64)
        invalid = ~support[ys, xs]
        # A foreground jitter must never turn into a background prompt.  The
        # unjittered reviewed point is the deterministic safe fallback.
        shifted[invalid] = original[invalid]
    return np.ascontiguousarray(shifted, dtype=np.float32)


def apply_sam_consensus_gate(
    preserve_mask: np.ndarray,
    foot_support: np.ndarray,
    confident_positive: np.ndarray,
    confident_negative: np.ndarray,
    *,
    reliable: bool = True,
    allow_positive_expansion: bool = False,
) -> np.ndarray:
    """Apply only high-confidence consensus edits and preserve uncertainty.

    Stable negative evidence removes pixels from ``preserve_mask``.  Uncertain
    pixels retain their previous state.  Stable positive evidence can recover
    pixels already present in the preserve mask; expansion into new pixels is
    opt-in because it is a materially less conservative operation.
    """

    preserve = _as_binary(preserve_mask, "preserve_mask")
    shape = tuple(preserve.shape)
    support = _as_binary(foot_support, "foot_support", shape)
    positive = _as_binary(confident_positive, "confident_positive", shape) & support
    negative = _as_binary(confident_negative, "confident_negative", shape) & support
    preserve &= support
    if not reliable:
        return np.ascontiguousarray(preserve)
    gated = preserve & ~negative
    if allow_positive_expansion:
        gated |= positive
    else:
        gated |= positive & preserve
    return np.ascontiguousarray(gated & support)


def refine_with_sam_consensus(
    predictor: Any,
    image_rgb: np.ndarray,
    chosen_candidate: MaskCandidate | np.ndarray,
    foot_support: np.ndarray,
    *,
    preserve_mask: np.ndarray | None = None,
    positive_points: Sequence[Any] | np.ndarray | None = None,
    negative_points: Sequence[Any] | np.ndarray | None = None,
    points_are_normalized: bool = True,
    normalized_scale: float = 1.0,
    set_image: bool = True,
    mask_input_size: tuple[int, int] | None = None,
    jitter_offsets: Sequence[tuple[float, float]] = DEFAULT_CONSENSUS_JITTER_OFFSETS,
    jitter_pixels: float = 2.0,
    positive_probability_threshold: float = 0.70,
    negative_probability_threshold: float = 0.30,
    consensus_fraction: float = 0.80,
    maximum_probability_std: float = 0.16,
    minimum_selected_score: float = 0.45,
    minimum_candidate_iou: float = 0.05,
    minimum_valid_members: int = 3,
    allow_positive_expansion: bool = False,
) -> SamConfidenceConsensus:
    """Estimate high-resolution SAM confidence with a prompt-jitter ensemble.

    Every invocation requests high-resolution logits with
    ``return_logits=True`` and all multimask alternatives.  The best output of
    each deterministic jitter is selected using the existing audited scoring
    function.  The returned mean/std/vote maps let integration code apply
    confident-negative suppression while preserving uncertain base pixels.
    No network or external API is used here.
    """

    image = _as_rgb_uint8(image_rgb)
    shape = tuple(image.shape[:2])
    raw_candidate = (
        chosen_candidate.mask
        if isinstance(chosen_candidate, MaskCandidate)
        else chosen_candidate
    )
    support = _as_binary(foot_support, "foot_support", shape)
    if not np.any(support):
        raise ValueError("foot_support is empty")
    candidate = _as_binary(raw_candidate, "chosen_candidate", shape) & support
    if not np.any(candidate):
        raise ValueError("chosen_candidate is empty inside foot_support")
    preserve = (
        candidate.copy()
        if preserve_mask is None
        else _as_binary(preserve_mask, "preserve_mask", shape) & support
    )

    probability_parameters = {
        "positive_probability_threshold": positive_probability_threshold,
        "negative_probability_threshold": negative_probability_threshold,
        "consensus_fraction": consensus_fraction,
        "maximum_probability_std": maximum_probability_std,
        "minimum_selected_score": minimum_selected_score,
        "minimum_candidate_iou": minimum_candidate_iou,
    }
    for name, value in probability_parameters.items():
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if float(negative_probability_threshold) >= float(positive_probability_threshold):
        raise ValueError(
            "negative_probability_threshold must be below positive_probability_threshold"
        )
    if not np.isfinite(jitter_pixels) or float(jitter_pixels) < 0.0:
        raise ValueError("jitter_pixels must be finite and non-negative")
    offsets = tuple((float(x), float(y)) for x, y in jitter_offsets)
    if not offsets:
        raise ValueError("jitter_offsets must not be empty")
    if not all(np.isfinite(value) for pair in offsets for value in pair):
        raise ValueError("jitter_offsets must be finite")
    if int(minimum_valid_members) < 1:
        raise ValueError("minimum_valid_members must be positive")

    if points_are_normalized:
        positive = normalized_points_to_crop(
            positive_points, shape, normalized_scale=normalized_scale
        )
        negative = normalized_points_to_crop(
            negative_points, shape, normalized_scale=normalized_scale
        )
    else:
        positive = _point_pairs(positive_points, "positive")
        negative = _point_pairs(negative_points, "negative")
        for name, points in (("positive", positive), ("negative", negative)):
            if bool(
                (
                    (points[:, 0] < 0)
                    | (points[:, 0] > shape[1] - 1)
                    | (points[:, 1] < 0)
                    | (points[:, 1] > shape[0] - 1)
                ).any()
            ):
                raise ValueError(f"{name} crop-pixel points fall outside the image")
    if not len(positive):
        positive = _deterministic_interior_positive_points(candidate)

    if set_image:
        if not hasattr(predictor, "set_image"):
            raise TypeError("predictor must provide set_image when set_image=True")
        predictor.set_image(image)
    if not hasattr(predictor, "predict"):
        raise TypeError("predictor must provide predict")
    low_resolution_size = mask_input_size or _predictor_mask_input_size(predictor)
    mask_input = _candidate_to_predictor_mask_input(
        candidate, predictor, low_resolution_size
    )

    probability_maps: list[np.ndarray] = []
    member_rows: list[SamConsensusMember] = []
    for run_index, unit_offset in enumerate(offsets):
        offset = (
            float(unit_offset[0]) * float(jitter_pixels),
            float(unit_offset[1]) * float(jitter_pixels),
        )
        run_positive = _jitter_prompt_points(
            positive, offset, shape, positive_support=candidate
        )
        run_negative = _jitter_prompt_points(negative, offset, shape)
        point_coords = np.ascontiguousarray(
            np.concatenate((run_positive, run_negative), axis=0), dtype=np.float32
        )
        point_labels = np.ascontiguousarray(
            np.concatenate(
                (
                    np.ones(len(run_positive), dtype=np.int32),
                    np.zeros(len(run_negative), dtype=np.int32),
                )
            )
        )
        prediction = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=True,
        )
        if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
            raise ValueError("predictor.predict must return masks and predicted IoUs")
        high_resolution_logits, predicted_ious = prediction[0], prediction[1]
        selection = select_best_sam_mask(
            high_resolution_logits,
            predicted_ious,
            candidate,
            support,
            positive_points=run_positive,
            negative_points=run_negative,
        )
        row = selection.scores[selection.selected_index]
        probability = _sam_logits_to_probability(
            np.asarray(high_resolution_logits)[selection.selected_index], shape
        )
        probability[~support] = 0.0
        usable = bool(
            row.enforced_area_pixels > 0
            and row.total_score >= float(minimum_selected_score)
            and row.candidate_iou >= float(minimum_candidate_iou)
        )
        member_rows.append(
            SamConsensusMember(
                run_index=int(run_index),
                jitter_xy=offset,
                selected_index=int(selection.selected_index),
                selected_score=float(row.total_score),
                sam_predicted_iou=float(row.sam_predicted_iou),
                candidate_iou=float(row.candidate_iou),
                usable=usable,
            )
        )
        if usable:
            probability_maps.append(probability)

    valid_count = len(probability_maps)
    reliable = bool(valid_count >= int(minimum_valid_members))
    if valid_count:
        stack = np.stack(probability_maps, axis=0).astype(np.float32)
        mean_probability = np.mean(stack, axis=0, dtype=np.float32)
        std_probability = np.std(stack, axis=0, dtype=np.float32)
        positive_consensus = np.mean(
            stack >= float(positive_probability_threshold), axis=0, dtype=np.float32
        )
        negative_consensus = np.mean(
            stack <= float(negative_probability_threshold), axis=0, dtype=np.float32
        )
    else:
        # Expose a neutral soft map and fail closed.  An all-empty/invalid SAM
        # ensemble must never become confident-negative evidence that erases
        # the existing contact mask.
        mean_probability = np.full(shape, 0.5, dtype=np.float32)
        std_probability = np.full(shape, 0.5, dtype=np.float32)
        positive_consensus = np.zeros(shape, dtype=np.float32)
        negative_consensus = np.zeros(shape, dtype=np.float32)

    stable = std_probability <= float(maximum_probability_std)
    confident_positive = (
        reliable
        & support
        & stable
        & (mean_probability >= float(positive_probability_threshold))
        & (positive_consensus >= float(consensus_fraction))
    )
    confident_negative = (
        reliable
        & support
        & stable
        & (mean_probability <= float(negative_probability_threshold))
        & (negative_consensus >= float(consensus_fraction))
    )
    uncertain = support & ~confident_positive & ~confident_negative
    stability = np.clip(
        1.0 - std_probability / max(float(maximum_probability_std), 1e-6),
        0.0,
        1.0,
    )
    polarity = np.maximum(positive_consensus, negative_consensus)
    soft_confidence = np.ascontiguousarray(stability * polarity, dtype=np.float32)
    soft_confidence[~support] = 0.0
    gated = apply_sam_consensus_gate(
        preserve,
        support,
        confident_positive,
        confident_negative,
        reliable=reliable,
        allow_positive_expansion=bool(allow_positive_expansion),
    )
    for array in (
        mean_probability,
        std_probability,
        positive_consensus,
        negative_consensus,
    ):
        array[~support] = 0.0
    return SamConfidenceConsensus(
        mean_probability=np.ascontiguousarray(mean_probability, dtype=np.float32),
        std_probability=np.ascontiguousarray(std_probability, dtype=np.float32),
        positive_consensus=np.ascontiguousarray(positive_consensus, dtype=np.float32),
        negative_consensus=np.ascontiguousarray(negative_consensus, dtype=np.float32),
        soft_confidence=soft_confidence,
        confident_positive=np.ascontiguousarray(confident_positive, dtype=np.bool_),
        confident_negative=np.ascontiguousarray(confident_negative, dtype=np.bool_),
        uncertain=np.ascontiguousarray(uncertain, dtype=np.bool_),
        gated_mask=gated,
        reference_mask=np.ascontiguousarray(candidate, dtype=np.bool_),
        preserve_mask=np.ascontiguousarray(preserve, dtype=np.bool_),
        foot_support=np.ascontiguousarray(support, dtype=np.bool_),
        members=tuple(member_rows),
        valid_member_count=int(valid_count),
        reliable=reliable,
    )


def refine_with_sam(
    predictor: Any,
    image_rgb: np.ndarray,
    chosen_candidate: MaskCandidate | np.ndarray,
    foot_support: np.ndarray,
    *,
    positive_points: Sequence[Any] | np.ndarray | None = None,
    negative_points: Sequence[Any] | np.ndarray | None = None,
    points_are_normalized: bool = True,
    normalized_scale: float = 1.0,
    set_image: bool = True,
    multimask_output: bool = True,
    mask_input_size: tuple[int, int] | None = None,
    geometry_probabilities: Sequence[float] | np.ndarray | None = None,
    classifier_probabilities: Sequence[float] | np.ndarray | None = None,
    qc_kwargs: Mapping[str, Any] | None = None,
) -> SamRefinementResult:
    """Refine one reviewed candidate with an injected ``SamPredictor``.

    ``positive_points`` and ``negative_points`` are interpreted in the GPT
    reviewer's 0..1 coordinate system by default.  Set
    ``points_are_normalized=False`` when supplying crop-pixel XY coordinates.
    """

    image = _as_rgb_uint8(image_rgb)
    shape = tuple(image.shape[:2])
    raw_candidate = chosen_candidate.mask if isinstance(chosen_candidate, MaskCandidate) else chosen_candidate
    candidate = _as_binary(raw_candidate, "chosen_candidate", shape)
    support = _as_binary(foot_support, "foot_support", shape)
    candidate &= support
    if not np.any(candidate):
        raise ValueError("chosen_candidate is empty inside foot_support")

    if points_are_normalized:
        positive = normalized_points_to_crop(
            positive_points, shape, normalized_scale=normalized_scale
        )
        negative = normalized_points_to_crop(
            negative_points, shape, normalized_scale=normalized_scale
        )
    else:
        positive = _point_pairs(positive_points, "positive")
        negative = _point_pairs(negative_points, "negative")
        for name, points in (("positive", positive), ("negative", negative)):
            if bool(
                (
                    (points[:, 0] < 0)
                    | (points[:, 0] > shape[1] - 1)
                    | (points[:, 1] < 0)
                    | (points[:, 1] > shape[0] - 1)
                ).any()
            ):
                raise ValueError(f"{name} crop-pixel points fall outside the image")

    if not len(positive):
        # SAM's dense mask prompt is an iterative-refinement hint, not a
        # complete standalone segmentation prompt. A deterministic point in
        # the reviewed candidate supplies the missing foreground semantics.
        # It is never placed outside that candidate and therefore cannot add
        # external visual evidence that GPT did not select.
        positive = _deterministic_interior_positive_points(candidate)

    if len(positive) or len(negative):
        point_coords = np.ascontiguousarray(np.concatenate((positive, negative), axis=0))
        point_labels = np.ascontiguousarray(
            np.concatenate(
                (
                    np.ones(len(positive), dtype=np.int32),
                    np.zeros(len(negative), dtype=np.int32),
                )
            )
        )
        predictor_coords: np.ndarray | None = point_coords
        predictor_labels: np.ndarray | None = point_labels
    else:
        point_coords = np.empty((0, 2), dtype=np.float32)
        point_labels = np.empty((0,), dtype=np.int32)
        predictor_coords = None
        predictor_labels = None

    if set_image:
        if not hasattr(predictor, "set_image"):
            raise TypeError("predictor must provide set_image when set_image=True")
        predictor.set_image(image)
    low_resolution_size = mask_input_size or _predictor_mask_input_size(predictor)
    mask_input = _candidate_to_predictor_mask_input(
        candidate,
        predictor,
        low_resolution_size,
    )
    if not hasattr(predictor, "predict"):
        raise TypeError("predictor must provide predict")
    prediction = predictor.predict(
        point_coords=predictor_coords,
        point_labels=predictor_labels,
        box=None,
        mask_input=mask_input,
        multimask_output=bool(multimask_output),
        return_logits=False,
    )
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise ValueError("predictor.predict must return masks and predicted IoUs")
    masks, predicted_ious = prediction[0], prediction[1]
    selection = select_best_sam_mask(
        masks,
        predicted_ious,
        candidate,
        support,
        positive_points=positive,
        negative_points=negative,
    )
    selected_row = selection.scores[selection.selected_index]
    if selected_row.enforced_area_pixels == 0 or selected_row.candidate_iou <= 0.0:
        # Preserve a useful visual artifact instead of emitting an empty PNG,
        # but make it impossible for the caller's normal SAM score threshold
        # to auto-accept this unrefined fallback. ``selected_index=-1`` is also
        # an explicit audit marker; the original SAM score rows are retained.
        selection = SamMaskSelection(
            mask=np.ascontiguousarray(candidate.copy()),
            selected_index=-1,
            selected_score=0.0,
            scores=selection.scores,
        )
    qc_options = dict(qc_kwargs or {})
    qc = compute_mask_qc(
        selection.mask,
        support,
        base_mask=candidate,
        positive_points=positive,
        negative_points=negative,
        geometry_probabilities=geometry_probabilities,
        classifier_probabilities=classifier_probabilities,
        **qc_options,
    )
    return SamRefinementResult(
        mask=selection.mask,
        selected_index=selection.selected_index,
        selected_score=selection.selected_score,
        scores=selection.scores,
        mask_input=mask_input,
        point_coords=point_coords,
        point_labels=point_labels,
        qc=qc,
    )


run_sam_refinement = refine_with_sam
run_sam_confidence_consensus = refine_with_sam_consensus


__all__ = [
    "CandidateMask",
    "DEFAULT_CANDIDATE_STRENGTHS",
    "MaskCandidate",
    "MaskQC",
    "MaskQualityControl",
    "SamMaskSelection",
    "SamConfidenceConsensus",
    "SamConsensusMember",
    "SamOutputScore",
    "SamRefinementResult",
    "DEFAULT_CONSENSUS_JITTER_OFFSETS",
    "apply_sam_consensus_gate",
    "candidate_to_sam_mask_input",
    "compute_mask_qc",
    "compute_refinement_qc",
    "enforce_foot_support",
    "generate_candidate_masks",
    "generate_conservative_candidate_masks",
    "map_normalized_points_to_crop",
    "normalized_points_to_crop",
    "refine_with_sam",
    "refine_with_sam_consensus",
    "render_candidate_review_board",
    "render_review_board",
    "run_sam_refinement",
    "run_sam_confidence_consensus",
    "select_best_sam_mask",
    "select_sam_output",
]
