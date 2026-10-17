from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

import cv2
import httpx
import numpy as np

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights
from app.services.pressed_region_references import build_reference_montage

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from ultralytics import YOLO  # noqa: E402
from app.services.tinea_analysis import predict_foot_mask


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
_model: YOLO | None = None
_model_lock = threading.Lock()
logger = logging.getLogger(__name__)
PROMPT_CACHE_KEY = "feetfit-plantar-guidance-v8-independent-refinement-v1"


class PressedRegionAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedFoot:
    image_bgr: np.ndarray
    foot_mask: np.ndarray
    gpt_image_png: bytes
    analysis_bbox: tuple[int, int, int, int]
    # Optional for compatibility with older local callers that construct this class.
    # prepare_foot always supplies an aligned, unmasked visual-evidence canvas.
    context_image_png: bytes | None = None


@dataclass(frozen=True)
class ContactRegionGuidance:
    roi: tuple[int, int, int, int]
    coarse_polygon: list[list[int]]
    positive_points: list[list[int]]
    negative_points: list[list[int]]
    confidence: float


@dataclass(frozen=True)
class FootContactGuidance:
    toe_direction: str
    orientation_uncertain: bool
    orientation_confidence: float
    global_confidence: float
    regions: list[ContactRegionGuidance]


@dataclass(frozen=True)
class RegionRefinementMetrics:
    confidence: float
    area_ratio: float
    support_fill_ratio: float
    coarse_retention_ratio: float
    stability_iou: float
    boundary_support: float
    feature_separation: float
    positive_coverage: float
    negative_leakage: float
    glare_ratio: float
    quality_score: float
    accepted: bool
    reason: str | None


@dataclass(frozen=True)
class RefinementResult:
    mask: np.ndarray
    score_map: np.ndarray
    metrics: list[RegionRefinementMetrics]


@dataclass(frozen=True)
class SavedPressedRegion:
    side: str
    visualization_path: Path
    mask_path: Path
    score_path: Path
    guidance_path: Path
    analysis_input_path: Path
    foot_mask_path: Path
    pressed_area_ratio: float
    region_count: int


def build_contact_guidance_prompt() -> str:
    return """
Analyze the patient's LEFT and RIGHT plantar feet independently. Each anatomical side is
supplied as an ordered pair of aligned images on the exact same fixed portrait canvas:
(1) COORDINATE CUTOUT, followed by (2) CONTEXT EVIDENCE. The coordinate cutout has already
been clipped by the existing YOLO sole mask, so pixels outside the sole are pure white.
The context evidence preserves original plate, reflection, and surrounding pixels, but it
has exactly the same rotation, scale, offset, and canvas dimensions as its cutout.
OpenCV has rotated the long foot axis vertically and has attempted to orient the toe end at the top.
All four inputs use the exact same portrait canvas size. Your output is coarse
guidance for a deterministic OpenCV refinement stage, not a final segmentation mask.

All output coordinates normalized to 0..1000 refer ONLY to the coordinate cutout canvas.
The aligned context image is visual evidence only: never use it as a second coordinate
system, never transform or offset coordinates because of it, and never let its apparent
frame change the cutout geometry. Analyze candidate pixels only inside the visible YOLO
sole silhouette on the coordinate cutout.

A separately supplied image may be labeled `REFERENCE_EXAMPLES`. It is sent before the
patient images and is a montage of paired source plantar images and their magenta
pressed-region masks. Study it first to learn the requested contact topology, island
separation, and mask style. It is not a patient image or coordinate canvas: never copy
its shapes, pixels, or coordinates into LEFT or RIGHT output. Magenta is an annotation
color, never evidence that the patient's similarly colored pixels are pressed. Ignore
the montage completely when it is not supplied, and output guidance only for the patient
LEFT and RIGHT images.

The upload fields already provide the anatomical side. The first ordered pair is the
patient's anatomical LEFT plantar foot (`leftFootImage`), and the second ordered pair is
the patient's anatomical RIGHT plantar foot (`rightFootImage`). Preserve these labels
exactly. Never infer, mirror, swap, or relabel the side from visual appearance or camera
orientation.

First verify that each cutout actually shows the plantar surface from a frontal view.
A dorsal/top-of-foot view, lateral side view, hand, truncated limb, or apparatus false
positive is not a plantar sole. For such an input, set `orientation_uncertain: true`,
set `global_confidence` near zero, and return `regions: []`; never reinterpret it as a
plantar footprint.

For every valid, clear, complete plantar sole, favor candidate recall over candidate
precision: return every region that could plausibly be pressed against the transparent
plate. Cross-check the cutout against its aligned context for plate-contact, reflection,
and compression cues: a locally flattened surface, continuous compression texture, a
distinct image-supported transition, contact-boundary optical change, a locally widened
load-bearing surface, or a coherent color/brightness change in a plausible load-bearing
location. Color or brightness may generate a coarse hypothesis, but cannot determine
the final boundary by itself. Plantar anatomy may also generate a coarse hypothesis when
the optical transition is subtle. Do not suppress a plausible candidate merely because
its evidence or your confidence is weak: report it with a lower `confidence` value.
OpenCV, not GPT confidence, is the final pixel-level accept/reject and boundary-refinement
stage. Your job is to provide broad, well-seeded candidate hypotheses, not final masks.

Use normal plantar-load topology only to challenge false positives: the heel pad,
metatarsal-head/forefoot band, and individual toe pads often form separate islands;
the medial longitudinal arch commonly leaves a gap; and the lateral midfoot can form
a narrower connection. These are possibilities, not mandatory regions. Preserve a
visible arch gap and toe gaps instead of drawing an anatomically imagined solid sole.

Before producing regions, inspect these zones one by one: each individual toe pad,
each disconnected metatarsal-head/forefoot island, lateral midfoot, medial arch, and
heel. Return all plausible candidates found in that exhaustive zone scan. Never merge
toe pads, forefoot, midfoot, and heel into one polygon merely because their skin color is
similar. Return separate small regions whenever the image shows a gap or an uncertain
bridge; disconnected toe, forefoot, midfoot, and heel islands must be separate regions.

Strict false-positive controls:
- Color alone is not contact evidence for a final boundary. Color, brightness, and
  saturation can still be hypothesis cues because OpenCV will test them against local
  image structure; they may also come from skin tone, exposure, or illumination.
- Anatomy is a valid coarse hypothesis cue for a clear complete sole with subtle signal,
  but is not a final boundary. It may support small heel, forefoot, toe-pad, arch, or
  lateral-load candidates. Preserve expected gaps and let OpenCV accept, refine, or
  reject every candidate from patient-image evidence.
- Exclude glare, specular highlights, residual reflections, droplets, bubbles,
  scratches, dirt, and mask-edge artifacts. Pure-white background is never contact.
- Do not bridge disconnected heel, midfoot, forefoot, or toe-pad contact islands.
- For a valid clear complete plantar sole, `regions: []` is forbidden. Even when all
  transitions are subtle, return the plausible, anatomically coherent candidates with
  lower confidence and usable positive/negative seeds so OpenCV can decide.

For every disconnected candidate return one region:
- `roi` is a loose bounding box around the candidate and its seeds.
- `coarse_polygon` is only a smooth coarse enclosure. Do not trace tiny pixel details
  and do not add long spikes or self-intersections.
- `positive_points` are clearly inside visible contact, away from its boundary.
- `negative_points` are clearly non-contact pixels just outside the candidate boundary,
  inside the anatomical sole and the same ROI, and not inside another contact island.
  Put them on multiple sides of the candidate. Never place a negative point on the
  pure-white background or any pixel outside the YOLO sole silhouette.
- `confidence` expresses visual evidence strength for diagnostics only. Low confidence
  must not cause you to omit an otherwise plausible candidate.

Coordinates cover the entire fixed-size COORDINATE CUTOUT canvas and are integer values
normalized to 0..1000. The expected orientation is long axis vertical with toes at the
top; still report the actually observed toe direction and mark orientation uncertainty
if this is not clear. The LEFT pair maps only to `left`; the RIGHT pair maps only to
`right`. Never use either context image to swap, mirror, or relabel the sides.
Use `regions: []` only for an invalid or non-plantar input: for example a dorsal/top view,
lateral view, hand, apparatus false positive, severely truncated foot, or an image with
no usable plantar sole. Never use an empty list for a valid clear complete plantar sole.
""".strip()


def build_pressed_polygon_prompt() -> str:
    """Backward-compatible name used by older local callers and tests."""
    return build_contact_guidance_prompt()


def _foot_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not weights.foot_segmentation.exists():
                    raise PressedRegionAnalysisError(
                        f"Foot segmentation weights not found: {weights.foot_segmentation}"
                    )
                _model = YOLO(str(weights.foot_segmentation))
    return _model


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise PressedRegionAnalysisError("Failed to encode an analysis image.")
    return encoded.tobytes()


def _mask_bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise PressedRegionAnalysisError("The sole mask is empty.")
    pad = max(0, int(padding))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )


def _select_primary_foot_cluster(mask: np.ndarray) -> np.ndarray:
    """Drop distant detections while retaining nearby disconnected toe components."""
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    if not len(xs):
        return binary.astype(bool)
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    radius = max(2, min(48, round(span * 0.015)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    linked = cv2.dilate(binary, kernel, iterations=1)
    count, labels = cv2.connectedComponents(linked)
    if count <= 2:
        return binary.astype(bool)
    scores = [int(binary[labels == idx].sum()) for idx in range(1, count)]
    best_label = int(np.argmax(scores)) + 1
    return ((labels == best_label) & (binary > 0))


def _remove_rectangular_top_protrusions(mask: np.ndarray) -> np.ndarray:
    """Remove a guarded, far-lateral rectangular branch from a plantar mask.

    YOLO occasionally unions a scan-apparatus rectangle with the toe end of an
    otherwise usable sole. The branch is still a separate component in shallow
    top slices even though it merges into the forefoot lower down. Only that
    shallow component is removed; broad bounding-box deletion would cut real toes.
    """
    source = np.asarray(mask)
    squeezed = np.squeeze(source)
    if squeezed.ndim != 2:
        return source.copy()
    binary = (squeezed > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    total_area = int(binary.sum())
    if total_area < 64 or not len(xs):
        return source.copy()

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width < 16 or bbox_height < 32:
        return source.copy()

    body_y1 = y_min + round(bbox_height * 0.35)
    body_y2 = y_min + round(bbox_height * 0.85)
    _body_rows, body_columns = np.where(binary[body_y1:body_y2] > 0)
    if not len(body_columns):
        return source.copy()
    body_center_x = float(np.median(body_columns))

    best_pixels: tuple[np.ndarray, np.ndarray] | None = None
    best_area = 0
    for fraction in np.arange(0.08, 0.2201, 0.005):
        cut_y = min(y_max, y_min + max(2, round(bbox_height * float(fraction))))
        top_slice = binary[y_min:cut_y, x_min:x_max]
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            top_slice,
            connectivity=8,
        )
        if count <= 2:
            continue
        eligible = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= total_area * 0.005
        ]
        if len(eligible) < 2:
            continue
        main_label = min(
            eligible,
            key=lambda label: abs(
                (x_min + float(centroids[label, 0])) - body_center_x
            ),
        )
        for label in eligible:
            if label == main_label:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            rectangularity = area / max(1, width * height)
            center_x = x_min + float(centroids[label, 0])
            lateral_distance = abs(center_x - body_center_x) / max(1, bbox_width)
            if (
                area >= total_area * 0.015
                and rectangularity >= 0.86
                and lateral_distance >= 0.22
                and area > best_area
            ):
                local_y, local_x = np.where(labels == label)
                best_pixels = (local_y + y_min, local_x + x_min)
                best_area = area

    if best_pixels is None or best_area > total_area * 0.08:
        return source.copy()

    cleaned = squeezed.copy()
    cleaned[best_pixels] = 0
    # Roll back if the conservative branch removal destroys the basic plantar
    # silhouette. A separately evaluated heel-taper gate may still reject a
    # truncated candidate and make the detector choose another YOLO proposal.
    if not _plantar_mask_shape_quality(cleaned > 0)[0]:
        return source.copy()
    return cleaned.astype(source.dtype, copy=False)


def _rotate_bound(
    image: np.ndarray,
    mask: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    rotated_width = max(1, int(np.ceil(height * sine + width * cosine)))
    rotated_height = max(1, int(np.ceil(height * cosine + width * sine)))
    matrix[0, 2] += rotated_width / 2.0 - center[0]
    matrix[1, 2] += rotated_height / 2.0 - center[1]
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return rotated_image, rotated_mask


def _canonicalize_sole(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _canonicalize_sole_impl(image, mask)


def _canonicalize_sole_impl(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize one image layer using geometry derived only from ``mask``."""
    mask = _select_primary_foot_cluster(mask)
    ys, xs = np.where(mask)
    if len(xs) < 3:
        raise PressedRegionAnalysisError("The sole mask is too small to orient.")
    coordinates = np.column_stack((xs, ys)).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(coordinates, mean=None, maxComponents=2)
    axis_x, axis_y = eigenvectors[0]
    long_axis_angle = float(np.degrees(np.arctan2(axis_y, axis_x)))
    # PCA axes have no direction: choose the equivalent rotation closest to the
    # source orientation instead of introducing an arbitrary 180-degree flip.
    rotation = (long_axis_angle % 180.0) - 90.0
    rotated_image, rotated_mask = _rotate_bound(
        image,
        mask,
        rotation,
    )

    # PCA makes the long axis vertical but cannot distinguish toe from heel because
    # an eigenvector has no direction. Resolve that 180-degree ambiguity from the
    # sole silhouette: the toe/forefoot end is normally wider and contains more mask
    # pixels than the heel end. No horizontal mirroring is performed, so anatomical
    # left/right identity remains unchanged.
    x1, y1, x2, y2 = _mask_bbox(rotated_mask, 0)
    tight = rotated_mask[y1:y2, x1:x2]
    tight_height, tight_width = tight.shape
    if tight_width >= tight_height:
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        rotated_mask = cv2.rotate(
            rotated_mask.astype(np.uint8),
            cv2.ROTATE_90_CLOCKWISE,
        ) > 0
        x1, y1, x2, y2 = _mask_bbox(rotated_mask, 0)
        tight = rotated_mask[y1:y2, x1:x2]

    if _toe_end_margin(tight) < 0.0:
        # Exact 180-degree rotation avoids another interpolation pass.
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
        rotated_mask = cv2.rotate(
            rotated_mask.astype(np.uint8),
            cv2.ROTATE_180,
        ) > 0
    return rotated_image, rotated_mask


def _toe_end_margin(tight_mask: np.ndarray) -> float:
    """Positive means the wider/lobed toe-forefoot end is at the top."""
    binary = np.asarray(tight_mask > 0, dtype=np.uint8)
    height, width = binary.shape
    if height < 8 or width < 3:
        return 0.0
    spans = np.zeros(height, np.float32)
    mass = binary.sum(axis=1).astype(np.float32)
    runs = np.zeros(height, np.float32)
    for y, row in enumerate(binary):
        columns = np.flatnonzero(row)
        if len(columns):
            spans[y] = float(columns[-1] - columns[0] + 1)
        padded = np.pad(row, (1, 0), constant_values=0)
        runs[y] = float(np.count_nonzero(np.diff(padded.astype(np.int8)) == 1))
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    diameter = 2.0 * distance.max(axis=1)
    sigma = max(1.0, height * 0.015)
    kernel = max(3, round(sigma * 6.0) | 1)

    def normalized(profile: np.ndarray) -> np.ndarray:
        smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1),
            (1, kernel),
            sigmaX=0.0,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        maximum = float(smooth.max())
        return smooth / maximum if maximum > 0.0 else smooth

    width_profile = (
        0.45 * normalized(spans)
        + 0.25 * normalized(mass)
        + 0.30 * normalized(diameter)
    )

    def quantile(start: float, end: float, profile: np.ndarray) -> float:
        begin = int(np.clip(round(height * start), 0, height - 1))
        finish = int(np.clip(round(height * end), begin + 1, height))
        return float(np.percentile(profile[begin:finish], 90))

    top_forefoot = quantile(0.08, 0.38, width_profile)
    bottom_forefoot = quantile(0.62, 0.92, width_profile)
    peak_threshold = float(np.percentile(width_profile, 90))
    peak_rows = np.flatnonzero(width_profile >= peak_threshold)
    if len(peak_rows):
        peak_weights = width_profile[peak_rows]
        peak_position = float(
            np.average(peak_rows, weights=peak_weights) / max(1, height - 1)
        )
    else:
        peak_position = 0.5
    top_runs = quantile(0.00, 0.22, runs)
    bottom_runs = quantile(0.78, 1.00, runs)
    return float(
        (top_forefoot - bottom_forefoot)
        + 0.35 * (0.5 - peak_position) * 2.0
        + 0.15 * np.clip((top_runs - bottom_runs) / 3.0, -1.0, 1.0)
    )


def _fit_clipped_sole_canvas(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """YOLO-clip one upright sole and letterbox it onto the shared GPT canvas."""
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise PressedRegionAnalysisError("Image and sole mask shapes are incompatible.")
    canvas_width = int(settings.pressed_region_canvas_width)
    canvas_height = int(settings.pressed_region_canvas_height)
    if canvas_width < 1 or canvas_height <= canvas_width:
        raise PressedRegionAnalysisError(
            "Pressed-region canvas must be portrait (height greater than width)."
        )
    margin_ratio = float(np.clip(settings.pressed_region_canvas_margin_ratio, 0.0, 0.25))
    x1, y1, x2, y2 = _mask_bbox(mask, 0)
    crop_mask = np.asarray(mask[y1:y2, x1:x2] > 0, dtype=bool)
    crop = image[y1:y2, x1:x2].copy()
    crop[~crop_mask] = 255

    usable_width = max(1, round(canvas_width * (1.0 - 2.0 * margin_ratio)))
    usable_height = max(1, round(canvas_height * (1.0 - 2.0 * margin_ratio)))
    scale = min(usable_width / crop.shape[1], usable_height / crop.shape[0])
    target_width = max(1, min(usable_width, round(crop.shape[1] * scale)))
    target_height = max(1, min(usable_height, round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (target_width, target_height), interpolation=interpolation)
    resized_mask = cv2.resize(
        crop_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    resized[~resized_mask] = 255

    canvas = np.full((canvas_height, canvas_width, 3), 255, np.uint8)
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    offset_x = (canvas_width - target_width) // 2
    offset_y = (canvas_height - target_height) // 2
    canvas[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = resized
    canvas_mask[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = resized_mask
    canvas[~canvas_mask] = 255
    return canvas, canvas_mask


def _fit_context_sole_canvas(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Map original visual context into the clipped sole's fixed coordinate canvas.

    The tight sole bounding box receives the exact same resize and placement as
    :func:`_fit_clipped_sole_canvas`. Source pixels just outside that box are warped into
    the otherwise-white canvas margins so plate/reflection cues remain available without
    introducing a second geometry.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise PressedRegionAnalysisError("Image and sole mask shapes are incompatible.")
    canvas_width = int(settings.pressed_region_canvas_width)
    canvas_height = int(settings.pressed_region_canvas_height)
    if canvas_width < 1 or canvas_height <= canvas_width:
        raise PressedRegionAnalysisError(
            "Pressed-region canvas must be portrait (height greater than width)."
        )
    margin_ratio = float(np.clip(settings.pressed_region_canvas_margin_ratio, 0.0, 0.25))
    x1, y1, x2, y2 = _mask_bbox(mask, 0)
    crop = image[y1:y2, x1:x2].copy()

    usable_width = max(1, round(canvas_width * (1.0 - 2.0 * margin_ratio)))
    usable_height = max(1, round(canvas_height * (1.0 - 2.0 * margin_ratio)))
    scale = min(usable_width / crop.shape[1], usable_height / crop.shape[0])
    target_width = max(1, min(usable_width, round(crop.shape[1] * scale)))
    target_height = max(1, min(usable_height, round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    offset_x = (canvas_width - target_width) // 2
    offset_y = (canvas_height - target_height) // 2

    # OpenCV resize uses pixel-center coordinates. This affine transform extends the
    # same source-to-canvas mapping beyond the sole box into the available margins.
    scale_x = target_width / crop.shape[1]
    scale_y = target_height / crop.shape[0]
    matrix = np.array(
        [
            [scale_x, 0.0, offset_x - scale_x * x1 + (scale_x - 1.0) * 0.5],
            [0.0, scale_y, offset_y - scale_y * y1 + (scale_y - 1.0) * 0.5],
        ],
        dtype=np.float32,
    )
    canvas = cv2.warpAffine(
        image,
        matrix,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # Override the coordinate-bearing rectangle with cv2.resize itself so pixels inside
    # and around the sole have precisely the same sampling grid as the clipped canvas.
    resized = cv2.resize(crop, (target_width, target_height), interpolation=interpolation)
    canvas[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = resized
    return canvas


def _plantar_mask_shape_quality(mask: np.ndarray) -> tuple[bool, float]:
    """Check that a YOLO candidate has a forefoot, midfoot, and non-truncated heel."""
    binary = _select_primary_foot_cluster(mask)
    ys, xs = np.where(binary)
    if len(xs) < 64:
        return False, 0.0
    coordinates = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        coordinates,
        mean=None,
        maxComponents=2,
    )
    long_axis = eigenvectors[0]
    short_axis = eigenvectors[1]
    centered = coordinates - mean.reshape(1, 2)
    longitudinal = centered @ long_axis
    lateral = centered @ short_axis
    long_range = float(np.ptp(longitudinal))
    short_range = float(np.ptp(lateral))
    if long_range < 8.0 or short_range < 4.0:
        return False, 0.0
    elongation = math.sqrt(
        float(eigenvalues[0, 0]) / max(1e-6, float(eigenvalues[1, 0]))
    )
    short_long_ratio = short_range / long_range

    bin_count = 100
    bins = np.clip(
        ((longitudinal - longitudinal.min()) * bin_count / long_range).astype(np.int32),
        0,
        bin_count - 1,
    )
    widths = np.zeros(bin_count, np.float32)
    for index in range(bin_count):
        values = lateral[bins == index]
        if len(values):
            widths[index] = float(np.ptp(values))
    widths = cv2.GaussianBlur(
        widths.reshape(-1, 1),
        (1, 7),
        sigmaX=0.0,
        sigmaY=1.2,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    top_forefoot = float(np.percentile(widths[8:38], 90))
    bottom_forefoot = float(np.percentile(widths[62:92], 90))
    if bottom_forefoot > top_forefoot:
        widths = widths[::-1]
    forefoot_width = float(np.percentile(widths[12:38], 90))
    heel_width = float(np.percentile(widths[90:99], 75))
    heel_forefoot_ratio = heel_width / max(1.0, forefoot_width)
    profile_width = max(1.0, float(np.percentile(widths, 95)))
    endpoint_width_ratio = max(
        float(np.percentile(widths[:4], 75)),
        float(np.percentile(widths[-4:], 75)),
    ) / profile_width
    max_heel_forefoot_ratio = float(
        getattr(settings, "pressed_region_max_heel_forefoot_ratio", 0.90)
    )
    max_endpoint_width_ratio = float(
        getattr(settings, "pressed_region_max_endpoint_width_ratio", 0.80)
    )

    valid = (
        1.25 <= elongation <= 5.0
        and 0.20 <= short_long_ratio <= 0.80
        and heel_forefoot_ratio >= 0.28
        and heel_forefoot_ratio <= max_heel_forefoot_ratio
        and endpoint_width_ratio <= max_endpoint_width_ratio
    )
    shape_score = float(
        np.clip(
            0.55 * ((heel_forefoot_ratio - 0.28) / 0.42)
            + 0.45 * (1.0 - abs(short_long_ratio - 0.50) / 0.30),
            0.0,
            1.0,
        )
    )
    return valid, shape_score


def _predict_small_sole(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fallback for apparatus photos where the reflected sole is very small."""
    h, w = image.shape[:2]
    tile_specs: list[tuple[int, int, int, int, bool]] = []

    def add_centered_tile(
        width_ratio: float,
        height_ratio: float,
        center_x_ratio: float,
        center_y_ratio: float,
        is_panel: bool,
    ) -> None:
        tile_w = int(np.clip(round(w * width_ratio), 256, w))
        tile_h = int(np.clip(round(h * height_ratio), 256, h))
        center_x = round(w * center_x_ratio)
        center_y = round(h * center_y_ratio)
        x = int(np.clip(center_x - tile_w // 2, 0, w - tile_w))
        y = int(np.clip(center_y - tile_h // 2, 0, h - tile_h))
        spec = (y, x, tile_h, tile_w, is_panel)
        if spec not in tile_specs:
            tile_specs.append(spec)

    panel_width = float(settings.pressed_region_panel_crop_width_ratio)
    panel_height = float(settings.pressed_region_panel_crop_height_ratio)
    panel_center_x = float(settings.pressed_region_panel_center_x_ratio)
    panel_center_y = float(settings.pressed_region_panel_center_y_ratio)
    # A close scan-panel crop makes a reflected plantar sole large enough for the
    # existing YOLO model. A slightly expanded/up-shifted view adds tolerance for
    # camera placement while the later ranker penalizes border-truncated masks.
    add_centered_tile(
        panel_width,
        panel_height,
        panel_center_x,
        panel_center_y,
        True,
    )
    add_centered_tile(
        min(1.0, panel_width * 1.17),
        min(1.0, panel_height * 1.17),
        panel_center_x,
        max(0.0, panel_center_y - 0.03),
        True,
    )

    tile_h, tile_w = max(256, round(h * 0.55)), max(256, round(w * 0.55))
    y_starts = sorted({0, max(0, (h - tile_h) // 2), max(0, h - tile_h)})
    x_starts = sorted({0, max(0, (w - tile_w) // 2), max(0, w - tile_w)})
    for y in y_starts:
        for x in x_starts:
            spec = (y, x, tile_h, tile_w, False)
            if spec not in tile_specs:
                tile_specs.append(spec)
    tiles = [image[y : y + th, x : x + tw] for y, x, th, tw, _ in tile_specs]
    results = _foot_model().predict(
        source=tiles,
        conf=max(0.1, settings.foot_confidence * 0.5),
        imgsz=1280,
        retina_masks=True,
        save=False,
        verbose=False,
    )
    candidates: list[tuple[float, float, int, np.ndarray]] = []
    for (y, x, current_h, current_w, is_panel), result in zip(
        tile_specs,
        results,
        strict=True,
    ):
        if result.masks is None or result.masks.data is None:
            continue
        scores = (
            result.boxes.conf.detach().cpu().numpy()
            if result.boxes is not None and result.boxes.conf is not None
            else np.ones(len(result.masks.data), np.float32)
        )
        for raw, score in zip(
            result.masks.data.detach().cpu().numpy(),
            scores,
            strict=False,
        ):
            local = cv2.resize(
                raw,
                (current_w, current_h),
                interpolation=cv2.INTER_LINEAR,
            ) > 0.5
            local = _remove_rectangular_top_protrusions(local)
            area = int(local.sum())
            if area < settings.foot_min_component_area:
                continue
            valid_shape, shape_score = _plantar_mask_shape_quality(local)
            if not valid_shape:
                continue
            local_y, local_x = np.where(local)
            if not len(local_x):
                continue
            candidate_bbox = (
                x + int(local_x.min()),
                y + int(local_y.min()),
                x + int(local_x.max()) + 1,
                y + int(local_y.max()) + 1,
            )
            full = np.zeros((h, w), np.uint8)
            full[y : y + current_h, x : x + current_w][local] = 1
            rank = _small_sole_candidate_score(
                confidence=float(score),
                area=area,
                candidate_bbox=candidate_bbox,
                tile_bbox=(x, y, x + current_w, y + current_h),
                image_shape=(h, w),
                panel_tile=is_panel,
                shape_score=shape_score,
            )
            candidates.append((rank, float(score), area, full))
    if not candidates:
        return np.zeros((h, w), np.uint8), False
    _, _, _, best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return best, True


def _small_sole_candidate_score(
    confidence: float,
    area: int,
    candidate_bbox: tuple[int, int, int, int],
    tile_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    panel_tile: bool,
    shape_score: float,
) -> float:
    """Rank fallback masks toward a complete sole in the configured scan panel."""
    image_h, image_w = image_shape
    x1, y1, x2, y2 = candidate_bbox
    tx1, ty1, tx2, ty2 = tile_bbox
    center_x = ((x1 + x2) / 2.0) / max(1, image_w)
    center_y = ((y1 + y2) / 2.0) / max(1, image_h)
    target_x = float(settings.pressed_region_panel_center_x_ratio)
    target_y = float(settings.pressed_region_panel_center_y_ratio)
    distance = math.hypot(center_x - target_x, center_y - target_y)
    centrality = float(np.clip(1.0 - distance / 0.75, 0.0, 1.0))
    relative_area = area / max(1, image_h * image_w)
    area_support = float(np.clip(math.sqrt(relative_area / 0.02), 0.0, 1.0))
    edge_margin = min(x1 - tx1, y1 - ty1, tx2 - x2, ty2 - y2)
    border_threshold = max(3, round(min(tx2 - tx1, ty2 - ty1) * 0.015))
    border_penalty = 0.20 if edge_margin <= border_threshold else 0.0
    return float(
        confidence
        + 0.15 * centrality
        + 0.06 * area_support
        + (0.05 if panel_tile else 0.0)
        + 0.10 * float(np.clip(shape_score, 0.0, 1.0))
        - border_penalty
    )


def prepare_foot(image_bytes: bytes) -> PreparedFoot:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise PressedRegionAnalysisError("Uploaded file is not a decodable image.")

    mask, found = predict_foot_mask(
        _foot_model(),
        image,
        settings.foot_confidence,
        settings.foot_min_component_area,
        settings.foot_iou_duplicate_threshold,
    )
    mask_bool = (
        _remove_rectangular_top_protrusions(_select_primary_foot_cluster(mask))
        if found
        else np.zeros(image.shape[:2], bool)
    )
    full_shape_valid, _ = (
        _plantar_mask_shape_quality(mask_bool) if found else (False, 0.0)
    )
    if not found or not full_shape_valid:
        fallback_mask, fallback_found = _predict_small_sole(image)
        if fallback_found:
            mask_bool = _remove_rectangular_top_protrusions(
                _select_primary_foot_cluster(fallback_mask)
            )
            found = True
        elif not found:
            raise PressedRegionAnalysisError("The existing model could not segment a sole.")
        else:
            raise PressedRegionAnalysisError(
                "The detected foot is not a complete plantar-facing sole."
            )

    # Retain an original-pixel evidence layer, but transform it with geometry derived
    # from the exact same YOLO mask as the coordinate-bearing cutout.
    context_image = image.copy()
    source_mask = mask_bool.copy()
    image = image.copy()
    # Apply the YOLO cutout before any interpolation so apparatus/background pixels
    # cannot bleed back into the sole boundary during rotation or resizing.
    image[~source_mask] = 255
    image, mask_bool = _canonicalize_sole(image, source_mask)
    context_image, context_mask = _canonicalize_sole_impl(context_image, source_mask)
    if not np.array_equal(mask_bool, context_mask):
        raise PressedRegionAnalysisError(
            "Coordinate and context images could not be aligned to the same sole mask."
        )
    image, mask_bool = _fit_clipped_sole_canvas(image, mask_bool)
    context_image = _fit_context_sole_canvas(context_image, context_mask)
    if context_image.shape != image.shape:
        raise PressedRegionAnalysisError(
            "Coordinate and context analysis canvases have different shapes."
        )
    height, width = mask_bool.shape
    return PreparedFoot(
        image_bgr=image,
        foot_mask=mask_bool,
        # GPT and OpenCV receive the exact same pixels and coordinate system.
        gpt_image_png=_encode_png(image),
        analysis_bbox=(0, 0, width, height),
        context_image_png=_encode_png(context_image),
    )


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _guidance_schema() -> dict[str, Any]:
    coordinate = {"type": "integer", "minimum": 0, "maximum": 1000}
    point = {
        "type": "array",
        "items": coordinate,
        "minItems": 2,
        "maxItems": 2,
    }
    roi = {
        "type": "object",
        "properties": {
            name: {"type": "integer", "minimum": 0, "maximum": 1000}
            for name in ("x_min", "y_min", "x_max", "y_max")
        },
        "required": ["x_min", "y_min", "x_max", "y_max"],
        "additionalProperties": False,
    }
    region = {
        "type": "object",
        "properties": {
            "roi": roi,
            "coarse_polygon": {
                "type": "array",
                "items": point,
                "minItems": 3,
                "maxItems": 20,
            },
            "positive_points": {
                "type": "array",
                "items": point,
                "minItems": 2,
                "maxItems": 8,
            },
            "negative_points": {
                "type": "array",
                "items": point,
                "minItems": 2,
                "maxItems": 12,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "roi",
            "coarse_polygon",
            "positive_points",
            "negative_points",
            "confidence",
        ],
        "additionalProperties": False,
    }
    foot = {
        "type": "object",
        "properties": {
            "toe_direction": {
                "type": "string",
                "enum": ["top", "bottom", "left", "right", "unknown"],
            },
            "orientation_uncertain": {"type": "boolean"},
            "orientation_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "global_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "regions": {"type": "array", "items": region, "maxItems": 16},
        },
        "required": [
            "toe_direction",
            "orientation_uncertain",
            "orientation_confidence",
            "global_confidence",
            "regions",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"left": foot, "right": foot},
        "required": ["left", "right"],
        "additionalProperties": False,
    }


def build_contact_guidance_request(left: PreparedFoot, right: PreparedFoot) -> dict[str, Any]:
    left_context = left.context_image_png or left.gpt_image_png
    right_context = right.context_image_png or right.gpt_image_png
    reference_montage = build_reference_montage(
        getattr(settings, "pressed_region_reference_dataset_dir", None),
        getattr(settings, "pressed_region_reference_annotation_dir", None),
        int(getattr(settings, "pressed_region_reference_max_pairs", 0)),
    )
    user_content: list[dict[str, Any]] = []
    if reference_montage is not None:
        user_content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        "REFERENCE_EXAMPLES only. Each row shows SOURCE on the left "
                        "and its human-annotated magenta PRESS MASK on the right. "
                        "Rows are ordered top=CAVUS, middle=NORMAL, bottom=FLAT. "
                        "Magenta is annotation only and is never a patient color cue. "
                        "Learn the expected plantar contact topology and separation "
                        "of pressure islands, but never reuse these coordinates for "
                        "the patient images that follow. Output only patient LEFT "
                        "and RIGHT guidance."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": _data_url(reference_montage),
                    # Preserve the small disconnected islands in the ground-truth masks.
                    "detail": "high",
                },
            ]
        )
    user_content.extend(
        [
            {
                "type": "input_text",
                "text": (
                    "SOURCE_FIELD=leftFootImage; ANATOMICAL_SIDE=LEFT. "
                    "The next two images are, in order, the LEFT COORDINATE "
                    "CUTOUT and its pixel-aligned LEFT CONTEXT EVIDENCE. "
                    "Return every LEFT coordinate on the first image only. "
                    "Bind it only to JSON key `left`; never swap sides."
                ),
            },
            {
                "type": "input_image",
                "image_url": _data_url(left.gpt_image_png),
                "detail": settings.openai_pressed_region_image_detail,
            },
            {
                "type": "input_image",
                "image_url": _data_url(left_context),
                "detail": "high",
            },
            {
                "type": "input_text",
                "text": (
                    "SOURCE_FIELD=rightFootImage; ANATOMICAL_SIDE=RIGHT. "
                    "The next two images are, in order, the RIGHT COORDINATE "
                    "CUTOUT and its pixel-aligned RIGHT CONTEXT EVIDENCE. "
                    "Return every RIGHT coordinate on the first image only. "
                    "Bind it only to JSON key `right`; never swap sides."
                ),
            },
            {
                "type": "input_image",
                "image_url": _data_url(right.gpt_image_png),
                "detail": settings.openai_pressed_region_image_detail,
            },
            {
                "type": "input_image",
                "image_url": _data_url(right_context),
                "detail": "high",
            },
        ]
    )
    return {
        "model": settings.openai_pressed_region_model,
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "reasoning": {"effort": settings.openai_pressed_region_reasoning_effort},
        "input": [
            {"role": "system", "content": build_contact_guidance_prompt()},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pressed_region_guidance_v4",
                "strict": True,
                "schema": _guidance_schema(),
            }
        },
        "max_output_tokens": settings.openai_pressed_region_max_output_tokens,
        "store": False,
    }


def _response_text(data: dict[str, Any], request_id: str = "missing") -> str:
    if not isinstance(data, dict):
        raise PressedRegionAnalysisError(
            f"OpenAI returned a non-object response (request_id={request_id})."
        )
    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
        raise PressedRegionAnalysisError(
            f"OpenAI response incomplete: {reason} (request_id={request_id})"
        )
    if data.get("status") != "completed":
        raise PressedRegionAnalysisError(
            "Unexpected OpenAI response status: "
            f"{data.get('status')} (request_id={request_id})"
        )
    texts: list[str] = []
    output = data.get("output", [])
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            continue
        content_items = item.get("content", [])
        for content in content_items if isinstance(content_items, list) else []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                refusal = str(content.get("refusal", ""))[:240]
                raise PressedRegionAnalysisError(
                    "OpenAI refused the image analysis request "
                    f"(request_id={request_id}): {refusal}"
                )
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    # Kept only for compatibility with gateways that synthesize the SDK's
    # convenience field. Refusal content above is always inspected first.
    if not texts and isinstance(data.get("output_text"), str):
        texts.append(data["output_text"])
    if len(texts) != 1:
        raise PressedRegionAnalysisError(
            "OpenAI response did not contain exactly one structured output."
        )
    return texts[0]


def _sanitize_point(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return [int(np.clip(x, 0, 1000)), int(np.clip(y, 0, 1000))]


def _sanitize_points(values: Any) -> list[list[int]]:
    points: list[list[int]] = []
    for value in values if isinstance(values, list) else []:
        point = _sanitize_point(value)
        if point is not None and point not in points:
            points.append(point)
    return points


def _sanitize_polygon(values: Any) -> list[list[int]]:
    points = _sanitize_points(values)
    if len(points) < 3:
        return []
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(contour))) < 20.0:
        return []
    return points


def _parse_foot_guidance(raw: Any) -> FootContactGuidance:
    if not isinstance(raw, dict):
        raise PressedRegionAnalysisError("OpenAI returned invalid foot guidance.")
    regions: list[ContactRegionGuidance] = []
    for value in raw.get("regions", []):
        if not isinstance(value, dict):
            continue
        polygon = _sanitize_polygon(value.get("coarse_polygon"))
        positive = _sanitize_points(value.get("positive_points"))
        negative = _sanitize_points(value.get("negative_points"))
        negative = [
            point
            for point in negative
            if all(
                (point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2 >= 15**2
                for seed in positive
            )
        ]
        roi_raw = value.get("roi") if isinstance(value.get("roi"), dict) else {}
        roi = tuple(
            int(np.clip(roi_raw.get(name, 0 if "min" in name else 1000), 0, 1000))
            for name in ("x_min", "y_min", "x_max", "y_max")
        )
        if len(polygon) < 3 or len(positive) < 2 or len(negative) < 2:
            continue
        try:
            confidence = float(value.get("confidence", 0.0))
            confidence = float(np.clip(confidence, 0.0, 1.0)) if math.isfinite(confidence) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        regions.append(
            ContactRegionGuidance(
                roi=roi,
                coarse_polygon=polygon,
                positive_points=positive,
                negative_points=negative,
                confidence=confidence,
            )
        )
    try:
        orientation_confidence = float(raw.get("orientation_confidence", 0.0))
        global_confidence = float(raw.get("global_confidence", 0.0))
        orientation_confidence = (
            float(np.clip(orientation_confidence, 0.0, 1.0))
            if math.isfinite(orientation_confidence)
            else 0.0
        )
        global_confidence = (
            float(np.clip(global_confidence, 0.0, 1.0))
            if math.isfinite(global_confidence)
            else 0.0
        )
    except (TypeError, ValueError):
        orientation_confidence = 0.0
        global_confidence = 0.0
    toe_direction = str(raw.get("toe_direction", "unknown"))
    if toe_direction not in {"top", "bottom", "left", "right", "unknown"}:
        toe_direction = "unknown"
    raw_uncertain = raw.get("orientation_uncertain", True)
    orientation_uncertain = (
        (raw_uncertain if isinstance(raw_uncertain, bool) else True)
        or toe_direction == "unknown"
        or orientation_confidence < 0.60
    )
    return FootContactGuidance(
        toe_direction=toe_direction,
        orientation_uncertain=orientation_uncertain,
        orientation_confidence=orientation_confidence,
        global_confidence=global_confidence,
        regions=regions,
    )


def parse_contact_guidance_document(
    raw: Any,
) -> dict[str, FootContactGuidance]:
    if not isinstance(raw, dict):
        raise PressedRegionAnalysisError("Saved GPT guidance is not a JSON object.")
    try:
        return {
            "left": _parse_foot_guidance(raw["left"]),
            "right": _parse_foot_guidance(raw["right"]),
        }
    except KeyError as exc:
        raise PressedRegionAnalysisError(
            "Saved GPT guidance must contain left and right entries."
        ) from exc


async def detect_contact_guidance(
    left: PreparedFoot,
    right: PreparedFoot,
) -> dict[str, FootContactGuidance]:
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        raise PressedRegionAnalysisError("OPENAI_API_KEY is not configured.")
    body = build_contact_guidance_request(left, right)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=settings.openai_pressed_region_timeout_seconds) as client:
        try:
            response = await client.post(OPENAI_RESPONSES_URL, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise PressedRegionAnalysisError(
                "OpenAI Responses API transport failed "
                f"({type(exc).__name__}): {exc!r}"
            ) from exc
    request_id = response.headers.get("x-request-id", "missing")
    if response.status_code >= 400:
        raise PressedRegionAnalysisError(
            "OpenAI Responses API returned "
            f"{response.status_code} (request_id={request_id}): {response.text[:500]}"
        )
    try:
        response_data = response.json()
        usage = response_data.get("usage") if isinstance(response_data, dict) else None
        if isinstance(usage, dict):
            logger.info(
                "pressed-region GPT request_id=%s model=%s input_tokens=%s "
                "cached_input_tokens=%s output_tokens=%s reasoning_tokens=%s",
                request_id,
                settings.openai_pressed_region_model,
                usage.get("input_tokens"),
                (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                usage.get("output_tokens"),
                (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            )
        parsed = json.loads(_response_text(response_data, request_id=request_id))
        return parse_contact_guidance_document(parsed)
    except PressedRegionAnalysisError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PressedRegionAnalysisError(
            f"OpenAI returned invalid contact guidance (request_id={request_id})."
        ) from exc


def polygons_to_mask(
    shape: tuple[int, int],
    polygons: list[list[list[int]]],
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = bbox or (0, 0, w, h)
    region_width = max(1, x2 - x1)
    region_height = max(1, y2 - y1)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(
            [
                [
                    x1 + round(p[0] * (region_width - 1) / 1000),
                    y1 + round(p[1] * (region_height - 1) / 1000),
                ]
                for p in polygon
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
    return mask


def _point_to_pixel(point: list[int], width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(round(point[0] * (width - 1) / 1000), 0, width - 1)),
        int(np.clip(round(point[1] * (height - 1) / 1000), 0, height - 1)),
    )


def _seed_mask(
    shape: tuple[int, int],
    points: list[list[int]],
    radius: int,
    foot_mask: np.ndarray,
    max_snap_distance: int = 0,
) -> np.ndarray:
    h, w = shape
    result = np.zeros((h, w), np.uint8)
    foot_y, foot_x = np.where(foot_mask)
    for point in points:
        center_x, center_y = _point_to_pixel(point, w, h)
        if not foot_mask[center_y, center_x]:
            if max_snap_distance <= 0 or not len(foot_x):
                continue
            distances = (foot_x - center_x) ** 2 + (foot_y - center_y) ** 2
            nearest = int(np.argmin(distances))
            if int(distances[nearest]) > max_snap_distance**2:
                continue
            center_x, center_y = int(foot_x[nearest]), int(foot_y[nearest])
        cv2.circle(result, (center_x, center_y), radius, 255, -1, cv2.LINE_AA)
    result[~foot_mask] = 0
    return result


def _adaptive_odd(value: float, minimum: int = 3, maximum: int = 31) -> int:
    size = int(np.clip(round(value), minimum, maximum))
    return size if size % 2 else size + 1


def _glare_mask(image: np.ndarray, foot_mask: np.ndarray, foot_length: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    sigma = max(1.0, foot_length * 0.012)
    local_mean = cv2.GaussianBlur(value.astype(np.float32), (0, 0), sigmaX=sigma)
    local_peak = value.astype(np.float32) - local_mean
    glare = (
        (value >= settings.pressed_region_glare_value_threshold)
        & (saturation <= settings.pressed_region_glare_saturation_threshold)
        & (local_peak >= settings.pressed_region_glare_local_contrast)
        & foot_mask
    )
    return glare


def _grabcut_views(image: np.ndarray, foot_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    broad = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(1.0, foot_length * 0.035))
    normalized_l = np.clip(128.0 + lightness - broad, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, foot_length * 0.008))
    local_sq = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=max(1.0, foot_length * 0.008))
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean * local_mean))
    std_low, std_high = np.percentile(local_std, (2, 98))
    local_std = np.clip(
        (local_std - std_low) * (255.0 / max(1.0, std_high - std_low)),
        0,
        255,
    ).astype(np.uint8)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    high = float(np.percentile(gradient, 99))
    gradient_u8 = np.clip(gradient * (255.0 / max(1.0, high)), 0, 255).astype(np.uint8)

    photometric = cv2.merge((normalized_l, lab[:, :, 1], lab[:, :, 2]))
    texture = cv2.merge((normalized_l, local_std, gradient_u8))
    return photometric, texture, gradient_u8


def _run_grabcut(view: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = initial.copy()
    if view.dtype != np.uint8 or view.ndim != 3 or view.shape[:2] != mask.shape:
        return np.zeros(mask.shape, np.uint8), True
    foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    background = np.isin(mask, [cv2.GC_BGD, cv2.GC_PR_BGD])
    if not foreground.any() or not background.any():
        return np.zeros(mask.shape, np.uint8), True
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            view,
            mask,
            None,
            background_model,
            foreground_model,
            max(1, int(settings.pressed_region_grabcut_iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return np.zeros(mask.shape, np.uint8), True
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8), False


def _keep_seeded_components(
    mask: np.ndarray,
    positive: np.ndarray,
    min_area: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros(mask.shape, np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area >= min_area and np.any(component & (positive > 0)):
            kept[component] = 1
    return kept


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = mask.copy()
    h, w = mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + width == w or y + height == h
        if not touches_border and area <= max_area:
            result[labels == label] = 1
    return result


def _boundary_support(mask: np.ndarray, gradient: np.ndarray, foot_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or not np.any(foot_mask):
        return 0.0
    boundary = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, 2)
    foot_gradient = gradient[foot_mask].astype(np.float32)
    low, threshold, high = np.percentile(foot_gradient, (20, 60, 95))
    # A flat image has no measurable contact transition. In particular, do not
    # count zero-valued pixels as supported merely because the percentile is zero.
    if high < 8.0 or high - low < 4.0:
        return 0.0
    values = gradient[boundary > 0]
    return float(np.mean(values > threshold)) if len(values) else 0.0


def _feature_separation(
    mask: np.ndarray,
    photometric: np.ndarray,
    texture: np.ndarray,
    foot_mask: np.ndarray,
    foot_length: int,
) -> float:
    """Measure multi-scale photometric *and* texture change across the result edge."""
    binary = (mask > 0).astype(np.uint8)
    radii = sorted(
        {
            max(2, min(32, round(foot_length * ratio)))
            for ratio in (0.004, 0.008)
        }
    )

    def channel_effect(channel: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> float:
        reference = channel[foot_mask].astype(np.float32)
        if not len(reference):
            return 0.0
        q25, q75 = np.percentile(reference, (25, 75))
        scale = max(12.0, float(q75 - q25))
        difference = abs(
            float(np.median(channel[inner])) - float(np.median(channel[outer]))
        )
        return difference / scale

    scores: list[float] = []
    for radius in radii:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        inner = (binary > 0) & (cv2.erode(binary, kernel, iterations=1) == 0)
        outer = (
            (cv2.dilate(binary, kernel, iterations=1) > 0)
            & (binary == 0)
            & foot_mask
        )
        if np.count_nonzero(inner) < 12 or np.count_nonzero(outer) < 12:
            continue
        photometric_strength = max(
            channel_effect(photometric[:, :, index], inner, outer)
            for index in range(3)
        )
        texture_strength = max(
            channel_effect(texture[:, :, index], inner, outer)
            for index in (1, 2)
        )
        # A single modality is insufficient: color-only and texture-only edges are
        # both common in skin, glare, and the glass plate.
        scores.append(float(np.sqrt(photometric_strength * texture_strength)))
    return float(np.clip(max(scores, default=0.0), 0.0, 1.0))


def _local_region_bounds(
    region: ContactRegionGuidance,
    width: int,
    height: int,
    foot_length: int,
) -> tuple[int, int, int, int]:
    geometry = region.coarse_polygon + region.positive_points + region.negative_points
    pixels = [_point_to_pixel(point, width, height) for point in geometry]
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    declared = (
        round(region.roi[0] * (width - 1) / 1000),
        round(region.roi[1] * (height - 1) / 1000),
        round(region.roi[2] * (width - 1) / 1000),
        round(region.roi[3] * (height - 1) / 1000),
    )
    margin = max(8, round(foot_length * settings.pressed_region_support_dilation_ratio * 1.5))
    gx1, gy1 = min(xs) - margin, min(ys) - margin
    gx2, gy2 = max(xs) + margin + 1, max(ys) + margin + 1
    dx1, dy1, dx2, dy2 = declared
    declared_valid = dx1 < dx2 and dy1 < dy2
    if declared_valid:
        geometry_area = max(1, (gx2 - gx1) * (gy2 - gy1))
        declared_area = max(1, (dx2 - dx1) * (dy2 - dy1))
        if declared_area <= geometry_area * 4:
            gx1, gy1 = min(gx1, dx1), min(gy1, dy1)
            gx2, gy2 = max(gx2, dx2), max(gy2, dy2)
    return (
        int(np.clip(gx1, 0, width - 1)),
        int(np.clip(gy1, 0, height - 1)),
        int(np.clip(gx2, 1, width)),
        int(np.clip(gy2, 1, height)),
    )


def _refine_region(
    image: np.ndarray,
    foot_mask: np.ndarray,
    region: ContactRegionGuidance,
    foot_area: int,
    foot_length: int,
) -> tuple[np.ndarray, np.ndarray, RegionRefinementMetrics]:
    h, w = foot_mask.shape
    empty = np.zeros((h, w), np.uint8)
    reject_reason: str | None = None

    coarse = polygons_to_mask((h, w), [region.coarse_polygon]) > 0
    coarse &= foot_mask
    seed_radius = max(3, min(24, round(foot_length * settings.pressed_region_seed_radius_ratio)))
    snap_distance = max(2, round(foot_length * 0.02))
    positive = _seed_mask(
        (h, w),
        region.positive_points,
        seed_radius,
        foot_mask,
        snap_distance,
    )
    negative = _seed_mask(
        (h, w),
        region.negative_points,
        max(2, seed_radius - 1),
        foot_mask,
        snap_distance,
    )
    negative[positive > 0] = 0
    if not np.any(coarse):
        reject_reason = reject_reason or "empty-coarse-region"
    if not np.any(positive):
        reject_reason = reject_reason or "no-positive-seed-on-foot"
    if not np.any(negative):
        reject_reason = reject_reason or "no-negative-seed-on-foot"
    if reject_reason is not None:
        metrics = RegionRefinementMetrics(
            confidence=region.confidence,
            area_ratio=0.0,
            support_fill_ratio=0.0,
            coarse_retention_ratio=0.0,
            stability_iou=0.0,
            boundary_support=0.0,
            feature_separation=0.0,
            positive_coverage=0.0,
            negative_leakage=0.0,
            glare_ratio=0.0,
            quality_score=0.0,
            accepted=False,
            reason=reject_reason,
        )
        return empty, empty, metrics

    x1, y1, x2, y2 = _local_region_bounds(region, w, h, foot_length)
    if x2 <= x1 or y2 <= y1:
        metrics = RegionRefinementMetrics(
            region.confidence,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            "invalid-roi",
        )
        return empty, empty, metrics

    local_image = image[y1:y2, x1:x2]
    local_foot = foot_mask[y1:y2, x1:x2]
    local_coarse = coarse[y1:y2, x1:x2]
    local_positive = positive[y1:y2, x1:x2]
    local_negative = negative[y1:y2, x1:x2]
    dilation = max(3, round(foot_length * settings.pressed_region_support_dilation_ratio))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation * 2 + 1, dilation * 2 + 1),
    )
    support = cv2.dilate(local_coarse.astype(np.uint8), support_kernel, iterations=1) > 0
    support |= cv2.dilate((local_positive > 0).astype(np.uint8), support_kernel, iterations=1) > 0
    support &= local_foot

    glare = _glare_mask(local_image, local_foot, foot_length)
    initial = np.full(local_foot.shape, cv2.GC_BGD, np.uint8)
    initial[local_foot] = cv2.GC_PR_BGD
    initial[support] = cv2.GC_PR_BGD
    initial[local_coarse] = cv2.GC_PR_FGD
    hard_background = ((local_negative > 0) | glare) & (local_positive == 0)
    initial[hard_background] = cv2.GC_BGD
    # A positive seed wins if GPT accidentally places opposite seed disks too close.
    initial[local_positive > 0] = cv2.GC_FGD

    # Provide definite local background samples even when GPT negatives were unusable.
    far_background = local_foot & ~cv2.dilate(
        support.astype(np.uint8), support_kernel, iterations=1
    ).astype(bool)
    initial[far_background] = cv2.GC_BGD

    photometric, texture, gradient = _grabcut_views(local_image, foot_length)
    first, first_failed = _run_grabcut(photometric, initial)
    # The second pass must not inherit GPT's entire coarse polygon as probable
    # foreground. It retains hard positive/negative seeds but must rediscover the
    # region from texture. Intersecting it with the coarse-guided photometric pass
    # prevents two identical priors from merely agreeing with themselves.
    independent_initial = initial.copy()
    independent_initial[independent_initial == cv2.GC_PR_FGD] = cv2.GC_PR_BGD
    second, second_failed = _run_grabcut(texture, independent_initial)
    grabcut_failed = first_failed or second_failed
    first &= support.astype(np.uint8)
    second &= support.astype(np.uint8)
    union = (first | second).astype(np.uint8)
    intersection = (first & second).astype(np.uint8)
    union_area = int(union.sum())
    stability = float(intersection.sum() / union_area) if union_area else 0.0
    refined = (
        intersection
        if stability >= settings.pressed_region_min_stability_iou
        else np.zeros(local_foot.shape, np.uint8)
    )

    open_size = _adaptive_odd(foot_length * 0.0015)
    close_size = _adaptive_odd(foot_length * 0.0025)
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    min_area = max(12, round(foot_area * settings.pressed_region_min_component_ratio))
    refined = _keep_seeded_components(refined, local_positive, min_area)
    refined = _fill_small_holes(
        refined,
        max(8, round(foot_area * settings.pressed_region_max_hole_ratio)),
    )
    glare_candidate_area = int(np.count_nonzero((refined > 0) & glare))
    # Hole filling must never erase explicit non-contact or reflection evidence.
    protected_background = ((local_negative > 0) | glare) & (local_positive == 0)
    refined[protected_background] = 0
    refined &= local_foot.astype(np.uint8)
    refined &= support.astype(np.uint8)

    area = int(refined.sum())
    area_ratio = float(area / foot_area) if foot_area else 0.0
    positive_count = int((local_positive > 0).sum())
    negative_count = int((local_negative > 0).sum())
    positive_coverage = (
        float(np.count_nonzero(refined & (local_positive > 0)) / positive_count)
        if positive_count
        else 0.0
    )
    negative_leakage = (
        float(np.count_nonzero(refined & (local_negative > 0)) / negative_count)
        if negative_count
        else 0.0
    )
    glare_ratio = float(
        glare_candidate_area / max(1, area + glare_candidate_area)
    )
    positive_disk_area = int(np.count_nonzero(local_positive))
    seed_expansion = float(area / positive_disk_area) if positive_disk_area else 0.0
    support_area = int(np.count_nonzero(support))
    support_fill = float(area / support_area) if support_area else 0.0
    coarse_area = int(np.count_nonzero(local_coarse))
    coarse_retention = (
        float(np.count_nonzero((refined > 0) & local_coarse) / coarse_area)
        if coarse_area
        else 0.0
    )
    boundary = _boundary_support(refined, gradient, local_foot)
    feature_separation = _feature_separation(
        refined,
        photometric,
        texture,
        local_foot,
        foot_length,
    )
    quality = float(
        np.clip(
            0.35 * stability
            + 0.30 * boundary
            + 0.25 * feature_separation
            + 0.05 * positive_coverage
            + 0.05 * (1.0 - negative_leakage)
            - 0.20 * glare_ratio,
            0.0,
            1.0,
        )
    )
    reason = None
    if grabcut_failed:
        reason = "grabcut-error"
    elif area == 0:
        reason = "empty-refinement"
    elif area_ratio > settings.pressed_region_max_region_ratio:
        reason = "region-area-too-large"
    elif (
        area_ratio >= settings.pressed_region_prior_dominance_min_area_ratio
        and coarse_retention
        >= settings.pressed_region_prior_dominance_min_coarse_retention
        and support_fill
        >= settings.pressed_region_prior_dominance_min_support_fill
    ):
        # Two GrabCut views share the same coarse initialization. If both simply
        # reproduce a broad GPT polygon, their agreement is not independent image
        # evidence. Require OpenCV to move at least one boundary materially.
        reason = "coarse-prior-not-refined"
    elif seed_expansion < settings.pressed_region_min_seed_expansion_ratio:
        reason = "seed-only-refinement"
    elif support_fill > settings.pressed_region_max_support_fill_ratio:
        reason = "support-filled-without-boundary"
    elif boundary < settings.pressed_region_min_boundary_support:
        reason = "weak-image-boundary"
    elif feature_separation < settings.pressed_region_min_feature_separation:
        reason = "weak-image-separation"
    elif positive_coverage < 0.75:
        reason = "positive-seed-not-retained"
    elif negative_leakage > 0.20:
        reason = "negative-seed-leakage"
    elif glare_ratio > 0.25:
        reason = "glare-dominated"
    elif quality < settings.pressed_region_min_quality_score:
        reason = "quality-score"
    accepted = reason is None

    full_mask = np.zeros((h, w), np.uint8)
    full_score = np.zeros((h, w), np.uint8)
    if accepted:
        full_mask[y1:y2, x1:x2] = refined * 255
    consensus = np.clip(first.astype(np.uint16) * 96 + second.astype(np.uint16) * 96, 0, 192)
    consensus[local_positive > 0] = 255
    full_score[y1:y2, x1:x2] = consensus.astype(np.uint8)
    metrics = RegionRefinementMetrics(
        confidence=round(region.confidence, 4),
        area_ratio=round(area_ratio, 6),
        support_fill_ratio=round(support_fill, 4),
        coarse_retention_ratio=round(coarse_retention, 4),
        stability_iou=round(stability, 4),
        boundary_support=round(boundary, 4),
        feature_separation=round(feature_separation, 4),
        positive_coverage=round(positive_coverage, 4),
        negative_leakage=round(negative_leakage, 4),
        glare_ratio=round(glare_ratio, 4),
        quality_score=round(quality, 4),
        accepted=accepted,
        reason=reason,
    )
    return full_mask, full_score, metrics


def refine_contact_mask(
    prepared: PreparedFoot,
    guidance: FootContactGuidance,
) -> RefinementResult:
    if prepared.image_bgr.ndim != 3 or prepared.image_bgr.shape[2] != 3:
        raise PressedRegionAnalysisError("Prepared image must be a 3-channel BGR image.")
    if prepared.foot_mask.shape != prepared.image_bgr.shape[:2]:
        raise PressedRegionAnalysisError("Prepared image and sole mask shapes differ.")
    non_upright_guidance = (
        guidance.toe_direction in {"bottom", "left", "right"}
        and guidance.orientation_confidence >= 0.75
    )
    x1, y1, x2, y2 = prepared.analysis_bbox
    height, width = prepared.foot_mask.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise PressedRegionAnalysisError("Prepared analysis bounds are invalid.")
    image = prepared.image_bgr[y1:y2, x1:x2]
    foot = np.asarray(prepared.foot_mask[y1:y2, x1:x2] > 0, dtype=bool)
    h, w = foot.shape
    if non_upright_guidance:
        # A semantic orientation disagreement is an inconclusive analysis, not a
        # transport or processing failure. Never refine coordinates against a sole
        # that GPT believes is still non-upright, but keep the request forwardable.
        empty = np.zeros((h, w), np.uint8)
        return RefinementResult(mask=empty, score_map=empty.copy(), metrics=[])
    combined = np.zeros((h, w), np.uint8)
    score = np.zeros((h, w), np.uint8)
    metrics: list[RegionRefinementMetrics] = []
    accepted_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    foot_area = int(foot.sum())
    foot_bbox = _mask_bbox(foot, 0)
    foot_length = max(
        1,
        foot_bbox[2] - foot_bbox[0],
        foot_bbox[3] - foot_bbox[1],
    )
    for region in guidance.regions:
        region_mask, region_score, region_metrics = _refine_region(
            image,
            foot,
            region,
            foot_area,
            foot_length,
        )
        metrics.append(region_metrics)
        if region_metrics.accepted and np.any(region_mask):
            accepted_candidates.append(
                (len(metrics) - 1, region_mask, region_score)
            )

    # Broad GPT proposals can overlap or collectively cover too much of the sole.
    # Preserve the strongest OpenCV-supported regions instead of invalidating every
    # otherwise reliable region when the total-area cap is crossed.
    accepted_candidates.sort(
        key=lambda item: metrics[item[0]].quality_score,
        reverse=True,
    )
    for metric_index, region_mask, region_score in accepted_candidates:
        proposed = cv2.bitwise_or(combined, region_mask)
        proposed_ratio = (
            float(np.count_nonzero(proposed) / foot_area) if foot_area else 0.0
        )
        if proposed_ratio <= settings.pressed_region_max_total_ratio:
            combined = proposed
            score = np.maximum(score, region_score)
            continue
        metrics[metric_index] = RegionRefinementMetrics(
            **{
                **asdict(metrics[metric_index]),
                "accepted": False,
                "reason": "total-area-cap-exceeded",
            }
        )

    combined[~foot] = 0
    return RefinementResult(mask=combined, score_map=score, metrics=metrics)


def save_result(
    prepared: PreparedFoot,
    guidance: FootContactGuidance,
    side: str,
    request_dir: Path,
) -> SavedPressedRegion:
    refinement = refine_contact_mask(prepared, guidance)
    x1, y1, x2, y2 = prepared.analysis_bbox
    foot = prepared.foot_mask > 0
    request_dir.mkdir(parents=True, exist_ok=True)
    visualization_path = request_dir / f"{side}_pressed_overlay.png"
    mask_path = request_dir / f"{side}_pressed_mask.png"
    score_path = request_dir / f"{side}_contact_score.png"
    guidance_path = request_dir / f"{side}_guidance.json"
    analysis_input_path = request_dir / f"{side}_analysis_input.png"
    foot_mask_path = request_dir / f"{side}_foot_mask.png"
    diagnostic_payload = {
        "guidance": asdict(guidance),
        "refinement": [asdict(metric) for metric in refinement.metrics],
    }
    alpha_mask = cv2.GaussianBlur(foot.astype(np.float32), (0, 0), sigmaX=1.0)
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)[:, :, None]
    white = np.full_like(prepared.image_bgr, 255)
    cutout = (
        prepared.image_bgr.astype(np.float32) * alpha_mask
        + white.astype(np.float32) * (1.0 - alpha_mask)
    ).astype(np.uint8)
    overlay = cutout[y1:y2, x1:x2].copy()
    region = refinement.mask > 0
    has_contact = bool(np.any(region))
    if has_contact:
        alpha = float(np.clip(settings.pressed_region_overlay_alpha, 0.0, 1.0))
        color = np.array([0, 0, 255], np.float32)
        overlay[region] = (
            (1 - alpha) * overlay[region] + alpha * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            refinement.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)
        score_to_save = refinement.score_map
        diagnostic_payload["status"] = "accepted"
    else:
        # Lack of sufficient visual evidence is a valid inconclusive analysis, not a
        # transport/processing failure. Preserve the sole-only visualization and make
        # both output maps explicitly empty so no rejected coarse guidance leaks out.
        contours = []
        score_to_save = np.zeros_like(refinement.score_map)
        reasons = sorted(
            {metric.reason or "unknown" for metric in refinement.metrics}
        )
        if (
            guidance.toe_direction in {"bottom", "left", "right"}
            and guidance.orientation_confidence >= 0.75
        ):
            detail = f"non-upright guidance ({guidance.toe_direction})"
        else:
            detail = ", ".join(reasons) if reasons else "no GPT contact candidates"
        diagnostic_payload["status"] = "inconclusive"
        diagnostic_payload["reason"] = detail

    if not cv2.imwrite(str(visualization_path), overlay):
        raise PressedRegionAnalysisError("Failed to save visualization.")
    if not cv2.imwrite(str(mask_path), refinement.mask):
        raise PressedRegionAnalysisError("Failed to save pressed-region mask.")
    if not cv2.imwrite(str(score_path), score_to_save):
        raise PressedRegionAnalysisError("Failed to save contact score map.")
    if not cv2.imwrite(str(analysis_input_path), prepared.image_bgr[y1:y2, x1:x2]):
        raise PressedRegionAnalysisError("Failed to save the analysis input.")
    if not cv2.imwrite(
        str(foot_mask_path),
        foot[y1:y2, x1:x2].astype(np.uint8) * 255,
    ):
        raise PressedRegionAnalysisError("Failed to save the sole mask.")
    guidance_path.write_text(
        json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    foot_area = int(np.count_nonzero(foot))
    pressed_area = int(np.count_nonzero(refinement.mask))
    return SavedPressedRegion(
        side=side,
        visualization_path=visualization_path,
        mask_path=mask_path,
        score_path=score_path,
        guidance_path=guidance_path,
        analysis_input_path=analysis_input_path,
        foot_mask_path=foot_mask_path,
        pressed_area_ratio=round(pressed_area / foot_area, 6) if foot_area else 0.0,
        region_count=len(contours),
    )


def combine_visualizations(
    left_path: Path,
    right_path: Path,
    output_path: Path | None = None,
) -> bytes:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise PressedRegionAnalysisError("Failed to read a saved visualization.")
    target_h = min(
        max(left.shape[0], right.shape[0]),
        settings.combined_image_max_height,
    )

    def fit(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_h:
            return image
        width = max(1, round(image.shape[1] * target_h / image.shape[0]))
        interpolation = cv2.INTER_AREA if target_h < image.shape[0] else cv2.INTER_LINEAR
        return cv2.resize(image, (width, target_h), interpolation=interpolation)

    fitted = [fit(left), fit(right)]
    cell_width = max(image.shape[1] for image in fitted)

    def center_in_cell(image: np.ndarray) -> np.ndarray:
        cell = np.full((target_h, cell_width, 3), 255, np.uint8)
        offset = (cell_width - image.shape[1]) // 2
        cell[:, offset : offset + image.shape[1]] = image
        return cell

    gap = np.full((target_h, settings.combined_image_gap_pixels, 3), 255, np.uint8)
    combined = np.concatenate(
        [center_in_cell(fitted[0]), gap, center_in_cell(fitted[1])],
        axis=1,
    )
    ok, encoded = cv2.imencode(".png", combined)
    if not ok:
        raise PressedRegionAnalysisError("Failed to encode the combined visualization.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), combined):
            raise PressedRegionAnalysisError("Failed to save the combined visualization.")
    return encoded.tobytes()


def new_request_dir() -> tuple[str, Path]:
    request_id = uuid4().hex
    return request_id, settings.pressed_region_output_dir / request_id
