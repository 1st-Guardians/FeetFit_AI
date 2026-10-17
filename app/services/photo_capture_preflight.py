"""Bounded capture validation for the three hardware photo uploads.

This module stops after decoding, inexpensive image checks, ArUco geometry,
and the production YOLO foot-presence gates.  It never runs foot measurement,
contact segmentation, skin/hallux models, GPT generation, or backend writes.
Original upload bytes remain available to the background pipeline. Plantar
validation also carries its selected preprocessing and YOLO ROI into contact
analysis so the accepted pixels and mask are not recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.services.aruco_foot_occlusion import (
    FootMarkerOcclusionDiagnostic,
    diagnose_foot_marker_occlusion,
)
from app.plantar_full_sam.foot_roi import FootRoiResult
from app.plantar_full_sam.preprocessing import (
    PreparedPlantarFootInput,
    iter_plantar_foot_inputs,
    prepare_plantar_foot_input,
)
from app.services.integrated_foot_analysis import (
    FootGeometryResult,
    IntegratedAnalysisError,
    _TwoMarkerCapture,
    _count_two_marker_anchors,
    _prepare_two_marker_capture,
    _segment_two_marker_feet,
    prepare_foot_geometry,
)
from app.services.photo_failure_guidance import dorsal_failure_diagnostic
from app.services.plantar_full_sam_analysis import (
    PlantarFullSamAnalysisError,
    extract_plantar_foot_roi,
)
from app.services.two_marker_foot_geometry import (
    REQUIRED_MARKER_IDS,
    TwoMarkerGeometryError,
)


MAX_DECODED_PIXELS = 40_000_000
MIN_PLANTAR_EDGE_PIXELS = 64
WHITE_CHANNEL_THRESHOLD = 245
MAX_NEAR_WHITE_PIXEL_RATIO = 0.985
MIN_PLANTAR_GRAYSCALE_STANDARD_DEVIATION = 6.0
BACKGROUND_DIFFERENCE_THRESHOLD = 18
MIN_FOREGROUND_COMPONENT_RATIO = 0.05
# ChArUco 36b4db1 uses the eight outer corners from IDs 1 and 4. Both markers
# must pass before hardware may proceed to pressure measurement.
REQUIRED_CAPTURE_ARUCO_IDS = frozenset(REQUIRED_MARKER_IDS)
REQUIRED_KNOWN_ARUCO_ANCHORS = len(REQUIRED_CAPTURE_ARUCO_IDS)

PhotoField = Literal[
    "dorsalFootImage",
    "leftPlantarFootImage",
    "rightPlantarFootImage",
]
PreflightDiagnosticCode = Literal[
    "DORSAL_DECODE_FAILED",
    "DORSAL_DIMENSIONS_MISMATCH",
    "DORSAL_MARKERS_INSUFFICIENT",
    "DORSAL_MARKER_OCCLUDED_BY_FOOT",
    "DORSAL_ARUCO_GEOMETRY_INVALID",
    "DORSAL_FEET_NOT_DETECTED",
    "DORSAL_FOOT_POSITION_INVALID",
    "DORSAL_FOOT_OUTLINE_INCOMPLETE",
    "DORSAL_BALL_WIDTH_ZONE_NOT_VISIBLE",
    "DORSAL_BALL_WIDTH_UNSTABLE",
    "PLANTAR_DECODE_FAILED",
    "PLANTAR_DIMENSIONS_UNSAFE",
    "PLANTAR_CAPTURE_BLANK",
    "PLANTAR_CAPTURE_NEAR_WHITE",
    "LEFT_PLANTAR_FOOT_NOT_DETECTED",
    "RIGHT_PLANTAR_FOOT_NOT_DETECTED",
]


@dataclass(frozen=True, slots=True)
class PhotoCapturePreflightError(ValueError):
    """Safe capture-validation failure metadata suitable for a 422 response."""

    diagnostic_code: PreflightDiagnosticCode
    image_field: PhotoField
    actual_width: int | None = None
    actual_height: int | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    detected_marker_count: int | None = None
    required_marker_count: int | None = None
    detected_foot_count: int | None = None
    required_foot_count: int | None = None
    marker_occlusion_detected: bool = False
    occluded_marker_ids: tuple[int, ...] = ()
    marker_occlusion_overlap_ratio: float | None = None
    marker_occlusion_diagnostic: FootMarkerOcclusionDiagnostic | None = None

    def __str__(self) -> str:
        # Never include filenames, image bytes, filesystem paths, or nested
        # decoder/runtime exception text in the operator/client boundary.
        return self.diagnostic_code


@dataclass(frozen=True, slots=True)
class PhotoCapturePreflightResult:
    dorsal_width: int
    dorsal_height: int
    detected_marker_count: int
    left_plantar_width: int
    left_plantar_height: int
    right_plantar_width: int
    right_plantar_height: int
    dorsal_detected_foot_count: int = 2
    left_plantar_foot_confidence: float | None = None
    right_plantar_foot_confidence: float | None = None
    left_plantar_prepared_input: PreparedPlantarFootInput | None = field(
        default=None, repr=False, compare=False,
    )
    right_plantar_prepared_input: PreparedPlantarFootInput | None = field(
        default=None, repr=False, compare=False,
    )


@dataclass(slots=True)
class _DecodedCapture:
    rgb: np.ndarray
    bgr: np.ndarray
    width: int
    height: int
    source_sha256: str = ""
    prepared_input: PreparedPlantarFootInput | None = field(
        default=None, repr=False, compare=False,
    )


@dataclass(frozen=True, slots=True)
class _PreparedDorsalCapture:
    capture: _TwoMarkerCapture
    detected_marker_count: int


def _open_supported_pillow_image(image_bytes: bytes):
    """Open only supported formats without Ultralytics' HEIF auto-installer."""

    configured_opener = Image.open
    opener_globals = getattr(configured_opener, "__globals__", {})
    captured_original = (
        opener_globals.get("_image_open")
        if isinstance(opener_globals, dict)
        else None
    )
    opener = captured_original if callable(captured_original) else configured_opener
    return opener(
        BytesIO(image_bytes),
        formats=("JPEG", "PNG", "WEBP"),
    )


def _decode_capture(
    image_bytes: bytes,
    *,
    image_field: PhotoField,
    decode_code: PreflightDiagnosticCode,
) -> _DecodedCapture:
    try:
        with _open_supported_pillow_image(image_bytes) as opened:
            # Inspect header dimensions before EXIF transpose or pixel loading.
            # This keeps a valid compressed container with hostile dimensions
            # from allocating its full decoded raster during preflight.
            encoded_width, encoded_height = (int(value) for value in opened.size)
            if (
                encoded_width <= 0
                or encoded_height <= 0
                or encoded_width * encoded_height > MAX_DECODED_PIXELS
            ):
                unsafe_code: PreflightDiagnosticCode = (
                    "PLANTAR_DIMENSIONS_UNSAFE"
                    if image_field != "dorsalFootImage"
                    else decode_code
                )
                raise PhotoCapturePreflightError(
                    diagnostic_code=unsafe_code,
                    image_field=image_field,
                    actual_width=max(0, encoded_width),
                    actual_height=max(0, encoded_height),
                )
            transposed = ImageOps.exif_transpose(opened)
            width, height = (int(value) for value in transposed.size)
            rgba = np.asarray(transposed.convert("RGBA"), dtype=np.uint8)
    except PhotoCapturePreflightError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        raise PhotoCapturePreflightError(
            diagnostic_code=decode_code,
            image_field=image_field,
        ) from None

    alpha = rgba[..., 3].astype(np.float32)[..., None] / 255.0
    rgb = np.rint(
        rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    )
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return _DecodedCapture(
        rgb=rgb,
        bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        width=width,
        height=height,
        source_sha256=sha256(image_bytes).hexdigest(),
    )


def _prepare_production_aruco_geometry(
    dorsal_bgr: np.ndarray,
) -> _PreparedDorsalCapture:
    """Run the ID 1/4 lens, orientation, and eight-corner Homography gates."""

    try:
        capture = _prepare_two_marker_capture(dorsal_bgr)
    except TwoMarkerGeometryError as exc:
        detected_anchor_count = _count_two_marker_anchors(dorsal_bgr)
        safe_reason = str(exc)
        markers_missing = (
            detected_anchor_count < REQUIRED_KNOWN_ARUCO_ANCHORS
            and "duplicate=" not in safe_reason
        )
        # A missing marker does not prove that a foot obscured it. This optional
        # diagnostic requires a localized missing marker AND a foot mask at that
        # location. Uncertain geometry/no feet keeps the existing failure code.
        diagnostic = diagnose_foot_marker_occlusion(dorsal_bgr) if markers_missing else None
        occlusion = (
            diagnostic.evidence if diagnostic is not None and diagnostic.status == "foot_occlusion"
            else None
        )
        if diagnostic is not None:
            detected_anchor_count = max(detected_anchor_count, len(diagnostic.detected_marker_ids))
        raise PhotoCapturePreflightError(
            diagnostic_code=(
                "DORSAL_MARKER_OCCLUDED_BY_FOOT" if occlusion is not None
                else "DORSAL_ARUCO_GEOMETRY_INVALID" if diagnostic is not None and diagnostic.status == "markers_visible"
                else "DORSAL_MARKERS_INSUFFICIENT" if markers_missing
                else "DORSAL_ARUCO_GEOMETRY_INVALID"
            ),
            image_field="dorsalFootImage",
            actual_width=int(dorsal_bgr.shape[1]),
            actual_height=int(dorsal_bgr.shape[0]),
            detected_marker_count=detected_anchor_count,
            required_marker_count=REQUIRED_KNOWN_ARUCO_ANCHORS,
            marker_occlusion_detected=occlusion is not None,
            occluded_marker_ids=occlusion.occluded_marker_ids if occlusion is not None else (),
            marker_occlusion_overlap_ratio=occlusion.overlap_ratio if occlusion is not None else None,
            marker_occlusion_diagnostic=diagnostic,
        ) from None

    detected_anchor_count = len(capture.corners_by_id)
    return _PreparedDorsalCapture(
        capture=capture,
        detected_marker_count=detected_anchor_count,
    )


def _validate_production_aruco_geometry(dorsal_bgr: np.ndarray) -> int:
    """Compatibility seam returning the production-known marker count."""

    return _prepare_production_aruco_geometry(dorsal_bgr).detected_marker_count


def _detected_dorsal_foot_count(error_message: str) -> int:
    missing = {
        side
        for side in {"image_left", "image_right"}
        if side in error_message
    }
    if not missing:
        return 0
    return max(0, REQUIRED_DORSAL_FEET - len(missing))


REQUIRED_DORSAL_FEET = 2


def _validate_dorsal_foot_presence(prepared: _PreparedDorsalCapture) -> int:
    """Require YOLO masks for both ID 1/4 calibrated board bays."""

    try:
        segmentation = _segment_two_marker_feet(prepared.capture)
    except TwoMarkerGeometryError as exc:
        safe_message = str(exc)
        raise PhotoCapturePreflightError(
            diagnostic_code="DORSAL_FEET_NOT_DETECTED",
            image_field="dorsalFootImage",
            actual_width=int(prepared.capture.canonical_bgr.shape[1]),
            actual_height=int(prepared.capture.canonical_bgr.shape[0]),
            detected_foot_count=_detected_dorsal_foot_count(safe_message),
            required_foot_count=REQUIRED_DORSAL_FEET,
        ) from None

    detected_sides = set(segmentation.analysis_masks_by_side)
    detected_count = len(
        detected_sides.intersection({"image_left", "image_right"})
    )
    if detected_count != REQUIRED_DORSAL_FEET:
        raise PhotoCapturePreflightError(
            diagnostic_code="DORSAL_FEET_NOT_DETECTED",
            image_field="dorsalFootImage",
            actual_width=int(prepared.capture.canonical_bgr.shape[1]),
            actual_height=int(prepared.capture.canonical_bgr.shape[0]),
            detected_foot_count=detected_count,
            required_foot_count=REQUIRED_DORSAL_FEET,
        )
    return detected_count


def _validate_plantar_capture(
    image_bytes: bytes,
    *,
    image_field: Literal["leftPlantarFootImage", "rightPlantarFootImage"],
) -> _DecodedCapture:
    decoded = _decode_capture(
        image_bytes,
        image_field=image_field,
        decode_code="PLANTAR_DECODE_FAILED",
    )
    if (
        decoded.width < MIN_PLANTAR_EDGE_PIXELS
        or decoded.height < MIN_PLANTAR_EDGE_PIXELS
    ):
        raise PhotoCapturePreflightError(
            diagnostic_code="PLANTAR_DIMENSIONS_UNSAFE",
            image_field=image_field,
            actual_width=decoded.width,
            actual_height=decoded.height,
        )

    first_error: PhotoCapturePreflightError | None = None
    for _, _, candidate_rgb in iter_plantar_foot_inputs(decoded.rgb):
        try:
            _validate_plantar_pixel_quality(
                candidate_rgb,
                image_field=image_field,
                actual_width=decoded.width,
                actual_height=decoded.height,
            )
        except PhotoCapturePreflightError as exc:
            if first_error is None:
                first_error = exc
        else:
            # Keep the original pixels for independently derived YOLO trials
            # and retain original upload dimensions in error metadata.
            return decoded
    assert first_error is not None
    raise first_error


def _validate_plantar_pixel_quality(
    rgb: np.ndarray,
    *,
    image_field: Literal["leftPlantarFootImage", "rightPlantarFootImage"],
    actual_width: int,
    actual_height: int,
) -> None:
    """Require usable contrast and foreground in one preprocessed candidate."""

    height, width = rgb.shape[:2]
    inspection = rgb
    longest_edge = max(width, height)
    if longest_edge > 512:
        scale = 512.0 / float(longest_edge)
        inspection = cv2.resize(
            rgb,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    white_ratio = float(
        np.mean(np.all(inspection >= WHITE_CHANNEL_THRESHOLD, axis=2))
    )
    if white_ratio >= MAX_NEAR_WHITE_PIXEL_RATIO:
        raise PhotoCapturePreflightError(
            diagnostic_code="PLANTAR_CAPTURE_NEAR_WHITE",
            image_field=image_field,
            actual_width=actual_width,
            actual_height=actual_height,
        )
    gray = cv2.cvtColor(inspection, cv2.COLOR_RGB2GRAY)
    if float(np.std(gray)) < MIN_PLANTAR_GRAYSCALE_STANDARD_DEVIATION:
        raise PhotoCapturePreflightError(
            diagnostic_code="PLANTAR_CAPTURE_BLANK",
            image_field=image_field,
            actual_width=actual_width,
            actual_height=actual_height,
        )

    sample_height, sample_width = inspection.shape[:2]
    border_size = max(1, int(round(min(sample_height, sample_width) * 0.05)))
    border_pixels = np.concatenate(
        (
            inspection[:border_size].reshape(-1, 3),
            inspection[-border_size:].reshape(-1, 3),
            inspection[:, :border_size].reshape(-1, 3),
            inspection[:, -border_size:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_rgb = np.median(border_pixels, axis=0)
    difference = np.max(
        np.abs(inspection.astype(np.int16) - background_rgb.astype(np.int16)),
        axis=2,
    )
    foreground = np.asarray(
        difference >= BACKGROUND_DIFFERENCE_THRESHOLD,
        dtype=np.uint8,
    )
    kernel = np.ones((7, 7), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    largest_component = (
        0
        if component_count <= 1
        else int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    )
    largest_component_ratio = largest_component / float(
        sample_width * sample_height
    )
    if largest_component_ratio < MIN_FOREGROUND_COMPONENT_RATIO:
        raise PhotoCapturePreflightError(
            diagnostic_code="PLANTAR_CAPTURE_BLANK",
            image_field=image_field,
            actual_width=actual_width,
            actual_height=actual_height,
        )


def _validate_plantar_foot_presence(
    decoded: _DecodedCapture,
    *,
    image_field: Literal["leftPlantarFootImage", "rightPlantarFootImage"],
) -> float:
    """Preprocess, validate foot presence, and retain the accepted model input."""

    decoded.prepared_input = None

    def extract_usable_candidate(candidate_rgb: np.ndarray) -> FootRoiResult:
        try:
            _validate_plantar_pixel_quality(
                candidate_rgb,
                image_field=image_field,
                actual_width=decoded.width,
                actual_height=decoded.height,
            )
        except PhotoCapturePreflightError as exc:
            height, width = candidate_rgb.shape[:2]
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            return FootRoiResult(
                found=False,
                mask=empty_mask,
                bbox=(0, 0, width, height),
                crop_image=candidate_rgb,
                crop_mask=empty_mask,
                confidence=0.0,
                instance_count=0,
                reason=exc.diagnostic_code,
                detection_pass="quality_rejected",
            )
        return extract_plantar_foot_roi(candidate_rgb)

    try:
        prepared = prepare_plantar_foot_input(
            decoded.rgb, extract=extract_usable_candidate,
        )
        roi = prepared.roi
    except PlantarFullSamAnalysisError as exc:
        # Missing weights, unavailable devices, and inference crashes are AI
        # server problems.  A normal no-foot prediction is returned as
        # ``found=False`` and is classified below as recapturable input.
        raise RuntimeError("Plantar foot-presence detector could not run.") from exc
    if not bool(roi.found):
        raise PhotoCapturePreflightError(
            diagnostic_code=(
                "LEFT_PLANTAR_FOOT_NOT_DETECTED"
                if image_field == "leftPlantarFootImage"
                else "RIGHT_PLANTAR_FOOT_NOT_DETECTED"
            ),
            image_field=image_field,
            actual_width=decoded.width,
            actual_height=decoded.height,
            detected_foot_count=0,
            required_foot_count=1,
        )
    decoded.prepared_input = replace(
        prepared,
        audit={**prepared.audit, "source_sha256": decoded.source_sha256},
    )
    return round(float(roi.confidence), 6)


def validate_dorsal_measurement_preflight(
    dorsal_bytes: bytes,
    filename: str | None,
) -> FootGeometryResult:
    """Require usable length and ball-width geometry before returning HTTP 202."""

    try:
        geometry = prepare_foot_geometry(dorsal_bytes, filename)
    except IntegratedAnalysisError as exc:
        diagnostic_code = dorsal_failure_diagnostic(exc)
        if diagnostic_code is None:
            # Model, calibration-asset, and configuration failures are server
            # failures. Do not tell the user to keep retaking valid photos.
            raise
        raise PhotoCapturePreflightError(
            diagnostic_code=diagnostic_code,
            image_field="dorsalFootImage",
        ) from None

    if not geometry.measurement_valid:
        raise PhotoCapturePreflightError(
            diagnostic_code=(
                "DORSAL_MARKERS_INSUFFICIENT"
                if geometry.missing_marker_ids
                else "DORSAL_ARUCO_GEOMETRY_INVALID"
            ),
            image_field="dorsalFootImage",
            detected_marker_count=len(geometry.detected_marker_ids),
            required_marker_count=REQUIRED_KNOWN_ARUCO_ANCHORS,
        )
    return geometry


def validate_photo_capture_preflight(
    dorsal_bytes: bytes,
    left_plantar_bytes: bytes,
    right_plantar_bytes: bytes,
) -> PhotoCapturePreflightResult:
    """Validate captures through YOLO presence gates without downstream I/O."""

    dorsal = _decode_capture(
        dorsal_bytes,
        image_field="dorsalFootImage",
        decode_code="DORSAL_DECODE_FAILED",
    )
    expected_width = int(settings.aruco_expected_image_width)
    expected_height = int(settings.aruco_expected_image_height)
    if (
        not settings.aruco_allow_calibration_resize
        and (dorsal.width, dorsal.height) != (expected_width, expected_height)
    ):
        raise PhotoCapturePreflightError(
            diagnostic_code="DORSAL_DIMENSIONS_MISMATCH",
            image_field="dorsalFootImage",
            actual_width=dorsal.width,
            actual_height=dorsal.height,
            expected_width=expected_width,
            expected_height=expected_height,
        )

    # Reject malformed/blank captures before loading either YOLO model.
    left_plantar = _validate_plantar_capture(
        left_plantar_bytes,
        image_field="leftPlantarFootImage",
    )
    right_plantar = _validate_plantar_capture(
        right_plantar_bytes,
        image_field="rightPlantarFootImage",
    )

    prepared_dorsal = _prepare_production_aruco_geometry(dorsal.bgr)
    detected_marker_count = prepared_dorsal.detected_marker_count
    if detected_marker_count < REQUIRED_KNOWN_ARUCO_ANCHORS:
        raise PhotoCapturePreflightError(
            diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
            image_field="dorsalFootImage",
            actual_width=dorsal.width,
            actual_height=dorsal.height,
            detected_marker_count=detected_marker_count,
            required_marker_count=REQUIRED_KNOWN_ARUCO_ANCHORS,
        )
    dorsal_foot_count = _validate_dorsal_foot_presence(prepared_dorsal)
    left_plantar_confidence = _validate_plantar_foot_presence(
        left_plantar,
        image_field="leftPlantarFootImage",
    )
    right_plantar_confidence = _validate_plantar_foot_presence(
        right_plantar,
        image_field="rightPlantarFootImage",
    )
    return PhotoCapturePreflightResult(
        dorsal_width=dorsal.width,
        dorsal_height=dorsal.height,
        detected_marker_count=detected_marker_count,
        left_plantar_width=left_plantar.width,
        left_plantar_height=left_plantar.height,
        right_plantar_width=right_plantar.width,
        right_plantar_height=right_plantar.height,
        dorsal_detected_foot_count=dorsal_foot_count,
        left_plantar_foot_confidence=left_plantar_confidence,
        right_plantar_foot_confidence=right_plantar_confidence,
        left_plantar_prepared_input=getattr(left_plantar, "prepared_input", None),
        right_plantar_prepared_input=getattr(right_plantar, "prepared_input", None),
    )
