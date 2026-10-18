from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme
from app.plantar_full_sam.preprocessing import PreparedPlantarFootInput
from app.schemas.photo_analysis import (
    PhotoAnalysisErrorDetail,
    PhotoAnalysisErrorResponse,
    PhotoMetricSummaryRequest,
)
from app.schemas.reports import (
    FootMetricsReportRequest,
    HalluxValgusReportRequest,
    TineaPedisReportRequest,
)
from app.services.integrated_foot_analysis import (
    FootGeometryResult,
    IntegratedFootAnalysisResult,
    analyze_integrated_foot_photo,
    analyze_prepared_foot_geometry,
)
from app.services.measurement_failure_notifier import notify_measurement_failed
from app.services.aruco_foot_occlusion import FOOT_MARKER_DIAGNOSTIC_REASON_CODES
from app.services.photo_capture_status import (
    PhotoCaptureStatusOutcome,
    is_photo_capture_attempt_conflict,
    notify_photo_capture_passed,
)
from app.services.measurement_recapture_notifier import (
    ARUCO_RECAPTURE_DIAGNOSTIC_CODES,
    MeasurementRecaptureOutcome,
    notify_measurement_recapture_required,
)
from app.services.photo_capture_preflight import (
    PhotoCapturePreflightError,
    REQUIRED_KNOWN_ARUCO_ANCHORS,
    validate_dorsal_measurement_preflight,
    validate_photo_capture_preflight,
)
from app.services.photo_failure_guidance import (
    PhotoFailureGuidance,
    dorsal_failure_diagnostic,
    marker_occlusion_explanation,
    photo_failure_guidance,
)
from app.services.plantar_full_sam_analysis import (
    PlantarFullSamAnalysisError,
    analyze_plantar_image,
    new_request_dir,
)
from app.services.report_text_generation import (
    FullPhotoOutputValidationError,
    generate_full_photo_analysis_texts,
    inflammation_report_uses_unannotated_images,
)
from app.services.tinea_analysis import (
    AnalysisError,
    build_plantar_skin_result,
    combine_png_images_side_by_side,
)
from app.services.tinea_policy import aggregate_tinea_safety_scores


router = APIRouter()
PIPELINE_VERSION = "full-photo-analysis-v1"
SKIPPED_METRIC_TYPES = ("PRESSURE_BALANCE", "FOOT_ENVIRONMENT")
PHOTO_ANALYSIS_MAX_PENDING_JOBS = 3
logger = logging.getLogger(__name__)
# Uvicorn does not emit INFO records from arbitrary application loggers with its
# default configuration.  Use its error logger for operator-facing progress
# events while retaining the module logger above for the existing warnings and
# errors.
progress_logger = logging.getLogger("uvicorn.error")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "token",
    "secret",
    "password",
    "apikey",
    "api_key",
    "credential",
)
_SAFE_UPSTREAM_BODY_LIMIT = 2000

# Progress details are deliberately a closed set of scalar operational facts.
# This prevents future callers from accidentally attaching credentials, source
# images, generated prose, or upstream response bodies to an INFO event.
_PROGRESS_DETAIL_KEYS = frozenset(
    {
        "adviceMetricCount",
        "allReportsOk",
        "analysisImageCount",
        "artifactCount",
        "attemptedSaveCount",
        "byteCount",
        "completedStageCount",
        "actualHeight",
        "actualWidth",
        "captureValidationStatus",
        "photoCaptureAttempt",
        "contactAreaRatio",
        "connectTimeoutSeconds",
        "coreReportCount",
        "descriptionCount",
        "detectedMarkerCount",
        "markerOcclusionStatus",
        "markerOcclusionReason",
        "markerOcclusionFootDetected",
        "markerOcclusionOverlapRatio",
        "markerOcclusionDetected",
        "markerOcclusionMethod",
        "markerVisibleCornerCount",
        "markerCoveredCornerCount",
        "markerPatternScore",
        "markerProjectionReason",
        "markerPartialCheckReason",
        "markerPartialCandidateCount",
        "markerPartialEdgeCount",
        "markerPartialCoveredCandidateCount",
        "markerPartialPatternCandidateCount",
        "detectedMarkerId1",
        "detectedMarkerId4",
        "missingMarkerId1",
        "missingMarkerId4",
        "detectedFootCount",
        "diagnosticCode",
        "expectedUploadCount",
        "expectedHeight",
        "expectedWidth",
        "errorCode",
        "elapsedSeconds",
        "queueWaitSeconds",
        "failedStageCount",
        "footprintStatus",
        "fungalSafetyScore",
        "generationSource",
        "generationTimeoutSeconds",
        "imageCount",
        "leftFootLengthMm",
        "leftFootWidthMm",
        "leftPlantarFootConfidence",
        "leftPlantarRotationDegreesClockwise",
        "leftPlantarBrightnessGain",
        "leftToeAngleDegree",
        "measurementStatus",
        "measurementValid",
        "missingMarkerCount",
        "outcome",
        "partialSave",
        "regionCount",
        "reportCount",
        "responseTimeoutDisabled",
        "retryable",
        "rightFootLengthMm",
        "rightFootWidthMm",
        "rightPlantarFootConfidence",
        "rightPlantarRotationDegreesClockwise",
        "rightPlantarBrightnessGain",
        "rightToeAngleDegree",
        "requiredMarkerCount",
        "requiredFootCount",
        "skinReactionSafetyScore",
        "statusCode",
        "successfulSaveCount",
        "summaryCount",
        "poolTimeoutSeconds",
    }
)
_PROGRESS_DETAIL_ENUMS = frozenset(
    {
        *FOOT_MARKER_DIAGNOSTIC_REASON_CODES,
        "foot_occlusion",
        "calibrated_single_marker_projection_yolo_overlap",
        "visible_partial_marker_corners_yolo_overlap",
        "no_foot_overlap",
        "unknown",
        "markers_visible",
        "detected",
        "http-error",
        "inconclusive",
        "invalid",
        "OPENAI",
        "DETERMINISTIC_FALLBACK",
        "PASSED",
        "PHOTO_UPLOAD_EMPTY",
        "PHOTO_UPLOAD_TOO_LARGE",
        "DORSAL_DECODE_FAILED",
        "DORSAL_DIMENSIONS_MISMATCH",
        "DORSAL_MARKERS_INSUFFICIENT",
        "DORSAL_MARKER_OCCLUDED_BY_FOOT",
        "PHOTO_CAPTURE_ATTEMPT_CONFLICT",
        "PHOTO_CAPTURE_STATUS_UNCONFIRMED",
        "DORSAL_ARUCO_GEOMETRY_INVALID",
        "DORSAL_FEET_NOT_DETECTED",
        "DORSAL_FOOT_POSITION_INVALID",
        "DORSAL_FOOT_OUTLINE_INCOMPLETE",
        "DORSAL_BALL_WIDTH_ZONE_NOT_VISIBLE",
        "DORSAL_BALL_WIDTH_UNSTABLE",
        "GPT_REPORT_RESPONSE_INVALID",
        "GPT_REPORT_GENERATION_FAILED",
        "PLANTAR_DECODE_FAILED",
        "PLANTAR_DIMENSIONS_UNSAFE",
        "PLANTAR_CAPTURE_BLANK",
        "PLANTAR_CAPTURE_NEAR_WHITE",
        "LEFT_PLANTAR_FOOT_NOT_DETECTED",
        "RIGHT_PLANTAR_FOOT_NOT_DETECTED",
        "saved",
        "skipped",
        "transport-error",
        "valid",
    }
)

_CORE_SAVE_STAGES = {
    "halluxValgus": "core-save-hallux-valgus",
    "metrics": "core-save-metrics",
    "tineaPedis": "core-save-tinea-pedis",
    "plantarFootprint": "core-save-plantar-footprint",
}
_CORE_REPORT_KOREAN_LABELS = {
    "halluxValgus": "무지외반 분석 결과",
    "metrics": "발 길이·발볼 측정 결과",
    "tineaPedis": "무좀·피부 분석 결과",
    "plantarFootprint": "발바닥 눌림 분석 결과",
}
_SUMMARY_KOREAN_LABELS = {
    "ATHLETES_FOOT": "무좀 위험도 지표 리포트",
    "SKIN_IRRITATION": "피부 자극도 지표 리포트",
    "HALLUX_VALGUS": "무지외반 지표 리포트",
}
_PHOTO_RESPONSE_HEADERS = {
    "X-Feetfit-Pipeline": {
        "description": "Photo-analysis pipeline version.",
        "schema": {"type": "string"},
    },
    "X-Feetfit-Request-Id": {
        "description": "Safe per-request correlation identifier.",
        "schema": {"type": "string"},
    },
}


@dataclass(frozen=True, slots=True, repr=False)
class PhotoAnalysisWorkItem:
    """Request-independent inputs owned by one background photo job."""

    request_id: str
    measurement_session_id: int
    photo_capture_attempt: int
    output_dir: Path
    dorsal_bytes: bytes
    left_plantar_bytes: bytes
    right_plantar_bytes: bytes
    dorsal_filename: str
    authorization: str
    completed_stages: tuple[str, ...]
    prepared_geometry: FootGeometryResult | None = None
    left_plantar_prepared_input: PreparedPlantarFootInput | None = None
    right_plantar_prepared_input: PreparedPlantarFootInput | None = None


# Full-photo analyzers share process-wide model instances. Serialize complete
# photo jobs even though individual blocking stages already use a thread pool.
_photo_analysis_worker_semaphore = asyncio.Semaphore(1)
_photo_analysis_reservation_lock = threading.Lock()
_active_photo_analysis_sessions: set[int] = set()


def _safe_generation_failure_code(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value) else None


def _safe_progress_details(
    details: Mapping[str, object] | None,
) -> dict[str, bool | int | float | str]:
    safe: dict[str, bool | int | float | str] = {}
    for key, original_value in (details or {}).items():
        if key not in _PROGRESS_DETAIL_KEYS:
            continue
        value = (
            original_value.item()
            if isinstance(original_value, np.generic)
            else original_value
        )
        if key == "errorCode":
            code = _safe_generation_failure_code(value)
            if code is not None:
                safe[key] = code
        elif isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int):
            safe[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            safe[key] = value
        elif isinstance(value, str) and value in _PROGRESS_DETAIL_ENUMS:
            safe[key] = value
    return safe


def _log_photo_progress(
    *,
    request_id: str,
    measurement_session_id: int,
    stage: str,
    event: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "pipeline": PIPELINE_VERSION,
        "requestId": str(request_id),
        "measurementSessionId": int(measurement_session_id),
        "event": event,
        "stage": stage,
        "message": message,
    }
    safe_details = _safe_progress_details(details)
    if safe_details:
        payload["details"] = safe_details
    display_message = message
    elapsed_seconds = safe_details.get("elapsedSeconds")
    queue_wait_seconds = safe_details.get("queueWaitSeconds")
    if isinstance(elapsed_seconds, (int, float)):
        display_message += f" (소요 {elapsed_seconds:.3f}초)"
    if isinstance(queue_wait_seconds, (int, float)):
        display_message += f" (대기 {queue_wait_seconds:.3f}초)"
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # Keep one concise operator-facing line while retaining the exact JSON
    # suffix for log collectors. The request id is shortened only in the human
    # prefix; the complete correlation id remains in the JSON payload.
    progress_logger.info(
        "[사진 분석][세션 %s][요청 %s] %s | pipeline_progress %s",
        int(measurement_session_id),
        str(request_id)[:8],
        display_message,
        serialized_payload,
    )


def _backend_save_progress_message(
    label: str,
    result: httpx.Response | BaseException,
) -> str:
    if isinstance(result, httpx.Response):
        if _is_2xx(result):
            return f"{label} 저장을 완료했습니다."
        return f"{label} 저장에 실패했습니다. (HTTP {result.status_code})"
    return f"{label} 저장에 실패했습니다. (백엔드 연결 오류)"


def _append_completed(completed_stages: list[str], stage: str) -> None:
    if stage not in completed_stages:
        completed_stages.append(stage)


def _mask_bearer(value: str) -> str:
    return _BEARER_PATTERN.sub("Bearer [REDACTED]", value)


def _redact_upstream_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            text_key = str(key)
            normalized = text_key.lower().replace("-", "").replace("_", "")
            if any(
                part.replace("_", "") in normalized
                for part in _SENSITIVE_KEY_PARTS
            ):
                redacted[text_key] = "[REDACTED]"
            else:
                redacted[text_key] = _redact_upstream_value(item, depth=depth + 1)
        return redacted
    if isinstance(value, (list, tuple)):
        return [
            _redact_upstream_value(item, depth=depth + 1)
            for item in list(value)[:64]
        ]
    if isinstance(value, str):
        return _mask_bearer(value)[:1000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _mask_bearer(str(value))[:500]


def _bounded_safe_upstream_body(value: Any) -> Any:
    safe = _redact_upstream_value(value)
    try:
        encoded = json.dumps(safe, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = _mask_bearer(str(safe))
    if len(encoded) <= _SAFE_UPSTREAM_BODY_LIMIT:
        return safe
    return {
        "truncated": True,
        "preview": _mask_bearer(encoded[:_SAFE_UPSTREAM_BODY_LIMIT]),
    }


async def _read_upload(upload: UploadFile, field_name: str) -> bytes:
    # Read only one byte beyond the configured limit. Starlette may already
    # have spooled a large multipart part to disk, but this prevents the AI
    # process from copying an arbitrarily large upload into memory before it
    # can return 413.
    data = await upload.read(settings.max_upload_size_bytes + 1)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is empty.",
        )
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field_name} exceeds {settings.max_upload_size_bytes} bytes.",
        )
    return data


def _read_png(path: Path, label: str, *, require_alpha: bool = False) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PlantarFullSamAnalysisError(f"Could not read {label}.") from exc
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise PlantarFullSamAnalysisError(f"{label} is not a valid PNG image.")
    if require_alpha and (decoded.ndim != 3 or decoded.shape[2] != 4):
        raise PlantarFullSamAnalysisError(f"{label} must preserve transparency.")
    return data


def _read_mask(path: Path, label: str) -> np.ndarray:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PlantarFullSamAnalysisError(f"Could not read {label}.") from exc
    mask = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or not bool(np.any(mask > 0)):
        raise PlantarFullSamAnalysisError(f"{label} is empty or invalid.")
    return np.ascontiguousarray(mask)


def _plantar_facts(result: Any) -> dict[str, object]:
    count = max(0, int(getattr(result, "region_count", 0)))
    ratio = float(getattr(result, "contact_area_ratio", 0.0))
    if not math.isfinite(ratio):
        ratio = 0.0
    ratio = min(1.0, max(0.0, ratio))
    return {
        "region_count": count,
        "contact_area_ratio": ratio,
        "status": "detected" if count > 0 else "inconclusive",
    }


def _is_2xx(response: httpx.Response) -> bool:
    return (
        200 <= int(response.status_code) < 300
        and not is_photo_capture_attempt_conflict(response)
    )


def _response_summary(response: httpx.Response) -> dict[str, object]:
    try:
        body: object = response.json() if response.content else None
    except (json.JSONDecodeError, ValueError):
        body = response.text
    return {
        "statusCode": int(response.status_code),
        "contentType": response.headers.get("content-type") or None,
        "body": _bounded_safe_upstream_body(body),
    }


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code < 600


def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _safe_attempt_summary(result: httpx.Response | BaseException) -> dict[str, object]:
    if isinstance(result, httpx.Response):
        status_code = int(result.status_code)
        return {
            "status": "saved" if _is_2xx(result) else "http-error",
            "statusCode": status_code,
            "retryable": (
                False if _is_2xx(result) else _is_retryable_http_status(status_code)
            ),
        }
    return {
        "status": "transport-error",
        "errorType": type(result).__name__,
        "retryable": _is_retryable_exception(result),
    }


def _progress_attempt_details(
    result: httpx.Response | BaseException,
) -> dict[str, object]:
    if isinstance(result, httpx.Response):
        status_code = int(result.status_code)
        return {
            "outcome": "saved" if _is_2xx(result) else "http-error",
            "statusCode": status_code,
            "retryable": (
                False if _is_2xx(result) else _is_retryable_http_status(status_code)
            ),
        }
    return {
        "outcome": "transport-error",
        "retryable": _is_retryable_exception(result),
    }


def _partial_save_occurred(
    *result_groups: dict[str, httpx.Response | BaseException],
) -> bool:
    return any(
        isinstance(result, httpx.Response) and _is_2xx(result)
        for group in result_groups
        for result in group.values()
    )


def _structured_photo_error(
    *,
    status_code: int,
    stage: str,
    completed_stages: list[str],
    request_id: str,
    error_type: str,
    retryable: bool,
    message: str,
    photo_capture_attempt: int | None = None,
    recommended_action: str | None = None,
    diagnostic_code: str | None = None,
    image_field: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    actual_width: int | None = None,
    actual_height: int | None = None,
    detected_marker_count: int | None = None,
    required_marker_count: int | None = None,
    detected_foot_count: int | None = None,
    required_foot_count: int | None = None,
    recapture_required: bool | None = None,
    same_session_recapture_allowed: bool | None = None,
    status_callback_confirmed: bool | None = None,
    next_measurement_status: str | None = None,
    should_disconnect: bool | None = None,
    photo_recapture_count: int | None = None,
    max_photo_recaptures: int | None = None,
    failure_reason: str | None = None,
    marker_occlusion_detected: bool | None = None,
    occluded_marker_ids: list[int] | None = None,
    marker_occlusion_overlap_ratio: float | None = None,
    marker_occlusion_status: str | None = None,
    marker_occlusion_reason: str | None = None,
    marker_occlusion_foot_detected: bool | None = None,
    marker_occlusion_method: str | None = None,
    marker_visible_corner_count: int | None = None,
    marker_covered_corner_count: int | None = None,
    marker_pattern_score: float | None = None,
    marker_projection_reason: str | None = None,
    marker_partial_check_reason: str | None = None,
    marker_partial_candidate_count: int | None = None,
    marker_partial_edge_count: int | None = None,
    marker_partial_covered_candidate_count: int | None = None,
    marker_partial_pattern_candidate_count: int | None = None,
    may_start_pressure: bool | None = None,
    reports: dict[str, Any] | None = None,
    partial_save: bool = False,
    measurement_status: str | None = None,
    measurement_invalid_reasons: list[str] | None = None,
    detected_marker_ids: list[int] | None = None,
    missing_marker_ids: list[int] | None = None,
    validation_field: str | None = None,
    validation_rule: str | None = None,
) -> HTTPException:
    detail = PhotoAnalysisErrorDetail(
        pipeline=PIPELINE_VERSION,
        stage=stage,
        completed_stages=list(completed_stages),
        request_id=request_id,
        photo_capture_attempt=photo_capture_attempt,
        error_type=error_type,
        diagnostic_code=diagnostic_code,
        image_field=image_field,
        expected_width=expected_width,
        expected_height=expected_height,
        actual_width=actual_width,
        actual_height=actual_height,
        detected_marker_count=detected_marker_count,
        required_marker_count=required_marker_count,
        detected_foot_count=detected_foot_count,
        required_foot_count=required_foot_count,
        recapture_required=recapture_required,
        same_session_recapture_allowed=same_session_recapture_allowed,
        status_callback_confirmed=status_callback_confirmed,
        next_measurement_status=next_measurement_status,
        should_disconnect=should_disconnect,
        photo_recapture_count=photo_recapture_count,
        max_photo_recaptures=max_photo_recaptures,
        failure_reason=failure_reason,
        marker_occlusion_detected=marker_occlusion_detected,
        occluded_marker_ids=occluded_marker_ids,
        marker_occlusion_overlap_ratio=marker_occlusion_overlap_ratio,
        marker_occlusion_status=marker_occlusion_status,
        marker_occlusion_reason=marker_occlusion_reason,
        marker_occlusion_foot_detected=marker_occlusion_foot_detected,
        marker_occlusion_method=marker_occlusion_method,
        marker_visible_corner_count=marker_visible_corner_count,
        marker_covered_corner_count=marker_covered_corner_count,
        marker_pattern_score=marker_pattern_score,
        marker_projection_reason=marker_projection_reason,
        marker_partial_check_reason=marker_partial_check_reason,
        marker_partial_candidate_count=marker_partial_candidate_count,
        marker_partial_edge_count=marker_partial_edge_count,
        marker_partial_covered_candidate_count=marker_partial_covered_candidate_count,
        marker_partial_pattern_candidate_count=marker_partial_pattern_candidate_count,
        may_start_pressure=may_start_pressure,
        validation_field=validation_field,
        validation_rule=validation_rule,
        retryable=retryable,
        partial_save=partial_save,
        reports=reports or {},
        message=message,
        recommended_action=recommended_action,
        measurement_status=measurement_status,
        measurement_invalid_reasons=measurement_invalid_reasons,
        detected_marker_ids=detected_marker_ids,
        missing_marker_ids=missing_marker_ids,
    )
    log_method = logger.warning if status_code < 500 else logger.error
    operator_message = (
        f"{message} {recommended_action}"
        if recommended_action is not None
        else message
    )
    marker_context = ""
    if marker_occlusion_status is not None:
        marker_context = " | " + json.dumps({
            "markerOcclusionStatus": marker_occlusion_status,
            "markerOcclusionReason": marker_occlusion_reason,
            "detectedMarkerIds": detected_marker_ids,
            "missingMarkerIds": missing_marker_ids,
            "markerOcclusionFootDetected": marker_occlusion_foot_detected,
            "markerOcclusionOverlapRatio": marker_occlusion_overlap_ratio,
            "markerOcclusionMethod": marker_occlusion_method,
            "markerVisibleCornerCount": marker_visible_corner_count,
            "markerCoveredCornerCount": marker_covered_corner_count,
            "markerPatternScore": marker_pattern_score,
            "markerProjectionReason": marker_projection_reason,
            "markerPartialCheckReason": marker_partial_check_reason,
            "markerPartialCandidateCount": marker_partial_candidate_count,
            "markerPartialEdgeCount": marker_partial_edge_count,
            "markerPartialCoveredCandidateCount": marker_partial_covered_candidate_count,
            "markerPartialPatternCandidateCount": marker_partial_pattern_candidate_count,
        }, ensure_ascii=False, separators=(",", ":"))
    log_method(
        "[사진 분석][요청 %s] %s | pipeline=%s request_id=%s stage=%s "
        "error_type=%s retryable=%s partial_save=%s status_code=%s%s",
        request_id[:8],
        operator_message,
        PIPELINE_VERSION,
        request_id,
        stage,
        error_type,
        retryable,
        partial_save,
        status_code,
        marker_context,
    )
    return HTTPException(
        status_code=status_code,
        detail=detail.model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
        ),
        headers={
            "X-Feetfit-Pipeline": PIPELINE_VERSION,
            "X-Feetfit-Request-Id": request_id,
        },
    )


def _photo_capture_status_error(
    *,
    outcome: PhotoCaptureStatusOutcome,
    stage: str,
    completed_stages: list[str],
    request_id: str,
    photo_capture_attempt: int,
) -> HTTPException:
    conflict = outcome.attempt_conflict
    return _structured_photo_error(
        status_code=status.HTTP_409_CONFLICT if conflict else status.HTTP_502_BAD_GATEWAY,
        stage=stage,
        completed_stages=completed_stages,
        request_id=request_id,
        photo_capture_attempt=photo_capture_attempt,
        error_type="PhotoCaptureAttemptConflict" if conflict else "PhotoCaptureStatusUnconfirmed",
        diagnostic_code="PHOTO_CAPTURE_ATTEMPT_CONFLICT" if conflict else "PHOTO_CAPTURE_STATUS_UNCONFIRMED",
        retryable=False,
        message=(
            "촬영 회차가 현재 측정 세션과 일치하지 않아 처리를 중단했습니다."
            if conflict else "백엔드의 촬영 검증 완료 상태를 확인하지 못해 분석을 시작하지 않았습니다."
        ),
        recommended_action="회차를 변경해 재전송하거나 FAILED로 변경하지 말고 백엔드의 현재 세션 상태를 확인해 주세요.",
        recapture_required=False,
        may_start_pressure=False,
        status_callback_confirmed=False,
        same_session_recapture_allowed=False,
        should_disconnect=False,
    )


async def _reserve_photo_analysis_session(
    *,
    measurement_session_id: int,
    request_id: str,
    completed_stages: list[str],
) -> None:
    with _photo_analysis_reservation_lock:
        if measurement_session_id in _active_photo_analysis_sessions:
            raise _structured_photo_error(
                status_code=status.HTTP_409_CONFLICT,
                stage="request-reservation",
                completed_stages=completed_stages,
                request_id=request_id,
                error_type="DuplicatePhotoAnalysisError",
                retryable=False,
                message="Photo analysis is already active for this measurement session.",
            )
        if len(_active_photo_analysis_sessions) >= PHOTO_ANALYSIS_MAX_PENDING_JOBS:
            error = _structured_photo_error(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                stage="request-reservation",
                completed_stages=completed_stages,
                request_id=request_id,
                error_type="PhotoAnalysisQueueFullError",
                retryable=True,
                message="The photo analysis queue is full. Retry later.",
            )
            assert error.headers is not None
            error.headers["Retry-After"] = "5"
            raise error
        _active_photo_analysis_sessions.add(measurement_session_id)


async def _release_photo_analysis_session(measurement_session_id: int) -> None:
    with _photo_analysis_reservation_lock:
        _active_photo_analysis_sessions.discard(measurement_session_id)


def _contains_upstream_status_code(value: object, target_status_code: int) -> bool:
    """Inspect a safe response/error payload without relying on its exact shape."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"statusCode", "status_code"}:
                try:
                    if int(item) == target_status_code:
                        return True
                except (TypeError, ValueError):
                    pass
            if _contains_upstream_status_code(item, target_status_code):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_upstream_status_code(item, target_status_code)
            for item in value
        )
    return False


_MARKER_CAPTURE_FAILURE_CODES = frozenset({
    "DORSAL_MARKERS_INSUFFICIENT", "DORSAL_ARUCO_GEOMETRY_INVALID",
    "DORSAL_MARKER_OCCLUDED_BY_FOOT",
})
_MARKER_DIAGNOSTIC_SERVER_ERRORS = frozenset({
    "ARUCO_RUNTIME_UNAVAILABLE", "MARKER_DETECTION_FAILED", "CALIBRATION_UNAVAILABLE",
    "CALIBRATION_QUALITY_INVALID", "LENS_CORRECTION_FAILED", "FOOT_MODEL_UNAVAILABLE",
    "FOOT_MODEL_INFERENCE_FAILED", "FOOT_MODEL_NO_RESULTS", "FOOT_MODEL_RESULT_INVALID",
    "FOOT_MASK_REMAP_FAILED", "DIAGNOSTIC_PROCESSING_FAILED",
    "PARTIAL_MARKER_INPUT_INVALID", "PARTIAL_MARKER_PROCESSING_FAILED",
})


def _photo_marker_diagnostic_kwargs(exc: PhotoCapturePreflightError) -> dict[str, Any]:
    diagnostic = exc.marker_occlusion_diagnostic
    if diagnostic is None:
        return {}
    diagnostic_status = diagnostic.status
    reason = diagnostic.reason_code
    if not isinstance(reason, str) or reason not in FOOT_MARKER_DIAGNOSTIC_REASON_CODES or diagnostic_status not in {
        "foot_occlusion", "no_foot_overlap", "unknown", "markers_visible",
    }:
        diagnostic_status, reason = "unknown", "DIAGNOSTIC_PROCESSING_FAILED"
    if diagnostic_status == "foot_occlusion" and not (
        exc.marker_occlusion_detected
        and exc.diagnostic_code == "DORSAL_MARKER_OCCLUDED_BY_FOOT"
        and diagnostic.evidence is not None
    ):
        diagnostic_status, reason = "unknown", "DIAGNOSTIC_PROCESSING_FAILED"
    result: dict[str, Any] = {
        "marker_occlusion_status": diagnostic_status,
        "marker_occlusion_reason": reason,
        "marker_occlusion_foot_detected": diagnostic.foot_detected,
        "detected_marker_ids": [value for value in diagnostic.detected_marker_ids
                                if type(value) is int and value in (1, 4)],
        "missing_marker_ids": [value for value in diagnostic.missing_marker_ids
                               if type(value) is int and value in (1, 4)],
        "marker_occlusion_overlap_ratio": diagnostic.overlap_ratio,
    }
    for field, attribute in (
        ("marker_projection_reason", "projection_reason_code"),
        ("marker_partial_check_reason", "partial_marker_reason_code"),
    ):
        value = getattr(diagnostic, attribute, None)
        if isinstance(value, str) and value in FOOT_MARKER_DIAGNOSTIC_REASON_CODES:
            result[field] = value
    for field, attribute in (
        ("marker_partial_candidate_count", "partial_marker_candidate_count"),
        ("marker_partial_edge_count", "partial_marker_edge_count"),
        ("marker_partial_covered_candidate_count", "partial_marker_covered_candidate_count"),
        ("marker_partial_pattern_candidate_count", "partial_marker_pattern_candidate_count"),
    ):
        value = getattr(diagnostic, attribute, None)
        if type(value) is int and 0 <= value <= 1000000:
            result[field] = value
    # Unknown is not a negative result. Omit the legacy boolean in that case.
    if diagnostic_status == "foot_occlusion":
        result["marker_occlusion_detected"] = True
        evidence = diagnostic.evidence
        if evidence is not None and evidence.method in {
            "calibrated_single_marker_projection_yolo_overlap", "visible_partial_marker_corners_yolo_overlap",
        }:
            result.update(
                marker_occlusion_method=evidence.method,
                marker_visible_corner_count=evidence.visible_corner_count,
                marker_covered_corner_count=evidence.covered_corner_count,
                marker_pattern_score=evidence.pattern_score,
            )
    elif diagnostic_status == "no_foot_overlap":
        result["marker_occlusion_detected"] = False
    return result


def _photo_failure_reason_for_status(status_code: int) -> str:
    if status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    }:
        # Keep generic legacy upload handling separate from the explicit
        # marker-validation INVALID_CAPTURE_DATA classification at preflight.
        return "CAMERA_ERROR"
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return "TOKEN_EXPIRED"
    if status_code in {
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_504_GATEWAY_TIMEOUT,
    }:
        return "NETWORK_ERROR"
    if status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "AI_SERVER_ERROR"
    return "UNKNOWN"


def _http_exception_failure_reason(exc: HTTPException) -> str:
    """Classify a terminal exception using status and safe upstream metadata."""

    detail = exc.detail if isinstance(exc.detail, Mapping) else {}
    if int(exc.status_code) == status.HTTP_401_UNAUTHORIZED or (
        _contains_upstream_status_code(detail, status.HTTP_401_UNAUTHORIZED)
    ):
        return "TOKEN_EXPIRED"
    if detail.get("diagnosticCode") in _MARKER_CAPTURE_FAILURE_CODES:
        return "INVALID_CAPTURE_DATA"
    return _photo_failure_reason_for_status(int(exc.status_code))


def _http_exception_failure_metadata(
    exc: HTTPException,
) -> tuple[
    str,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    detail = exc.detail if isinstance(exc.detail, Mapping) else {}
    stage = detail.get("stage")
    error_type = detail.get("errorType")
    diagnostic_code = detail.get("diagnosticCode")
    image_field = detail.get("imageField")
    validation_field = detail.get("validationField")
    validation_rule = detail.get("validationRule")
    if photo_failure_guidance(diagnostic_code, image_field) is None:
        diagnostic_code = None
        image_field = None
    if diagnostic_code != "GPT_REPORT_RESPONSE_INVALID":
        validation_field = None
        validation_rule = None
    return (
        stage if isinstance(stage, str) and stage else "background-worker",
        (
            error_type
            if isinstance(error_type, str) and error_type
            else type(exc).__name__
        ),
        diagnostic_code if isinstance(diagnostic_code, str) else None,
        image_field if isinstance(image_field, str) else None,
        validation_field if isinstance(validation_field, str) else None,
        validation_rule if isinstance(validation_rule, str) else None,
    )


def _partial_outcome_failure_metadata(
    response: JSONResponse,
) -> tuple[str, str, str]:
    try:
        payload = json.loads(response.body)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    raw_errors = payload.get("errors") if isinstance(payload, Mapping) else None
    errors = raw_errors if isinstance(raw_errors, list) else []
    safe_errors = [item for item in errors if isinstance(item, Mapping)]
    token_error = next(
        (
            item
            for item in safe_errors
            if _contains_upstream_status_code(item, status.HTTP_401_UNAUTHORIZED)
        ),
        None,
    )
    first = token_error or (safe_errors[0] if safe_errors else {})
    stage = first.get("stage")
    error_type = first.get("errorType")
    return (
        stage if isinstance(stage, str) and stage else "response",
        (
            error_type
            if isinstance(error_type, str) and error_type
            else "PartialPhotoAnalysisError"
        ),
        (
            "TOKEN_EXPIRED"
            if _contains_upstream_status_code(
                payload,
                status.HTTP_401_UNAUTHORIZED,
            )
            else "REPORT_SAVE_ERROR"
        ),
    )


async def _notify_photo_failure(
    *,
    measurement_session_id: int,
    photo_capture_attempt: int,
    authorization: str,
    request_id: str,
    failure_reason: str,
    stage: str,
    error_type: str,
    diagnostic_code: str | None = None,
    image_field: str | None = None,
    validation_field: str | None = None,
    validation_rule: str | None = None,
    marker_occlusion_reason: str | None = None,
) -> bool:
    try:
        notification_kwargs: dict[str, Any] = {
            "measurement_session_id": measurement_session_id,
            "photo_capture_attempt": photo_capture_attempt,
            "authorization": authorization,
            "failure_reason": failure_reason,
            "pipeline": PIPELINE_VERSION,
            "stage": stage,
            "error_type": error_type,
            "request_id": request_id,
        }
        if photo_failure_guidance(diagnostic_code, image_field) is not None:
            notification_kwargs.update(
                diagnostic_code=diagnostic_code,
                image_field=image_field,
            )
        if diagnostic_code == "GPT_REPORT_RESPONSE_INVALID":
            notification_kwargs.update(
                validation_field=validation_field,
                validation_rule=validation_rule,
            )
        if marker_occlusion_reason in FOOT_MARKER_DIAGNOSTIC_REASON_CODES:
            notification_kwargs["marker_occlusion_reason"] = marker_occlusion_reason
        notified = await notify_measurement_failed(
            **notification_kwargs,
        )
    except Exception:
        # The notifier is expected to absorb its own transport failures, but do
        # not let an integration regression leak out of the background worker.
        logger.error(
            "[사진 분석][요청 %s] 백엔드 실패 상태 전달 중 오류가 발생했습니다. | "
            "pipeline=%s request_id=%s stage=%s error_type=%s",
            request_id[:8],
            PIPELINE_VERSION,
            request_id,
            stage,
            error_type,
        )
        return False
    if not notified:
        logger.error(
            "[사진 분석][요청 %s] 백엔드에 실패 상태를 전달하지 못했습니다. | "
            "pipeline=%s request_id=%s stage=%s error_type=%s",
            request_id[:8],
            PIPELINE_VERSION,
            request_id,
            stage,
            error_type,
        )
    return notified is True


async def _notify_photo_job_failure(
    work_item: PhotoAnalysisWorkItem,
    *,
    failure_reason: str,
    stage: str,
    error_type: str,
    diagnostic_code: str | None = None,
    image_field: str | None = None,
    validation_field: str | None = None,
    validation_rule: str | None = None,
) -> None:
    await _notify_photo_failure(
        measurement_session_id=work_item.measurement_session_id,
        photo_capture_attempt=work_item.photo_capture_attempt,
        authorization=work_item.authorization,
        request_id=work_item.request_id,
        failure_reason=failure_reason,
        stage=stage,
        error_type=error_type,
        diagnostic_code=diagnostic_code,
        image_field=image_field,
        validation_field=validation_field,
        validation_rule=validation_rule,
    )


def _safe_backend_reports(
    core_results: dict[str, httpx.Response | BaseException],
    summary_results: dict[str, httpx.Response | BaseException] | None = None,
) -> dict[str, Any]:
    return {
        **{
            report_name: _safe_attempt_summary(result)
            for report_name, result in core_results.items()
        },
        "summaries": {
            metric_type: _safe_attempt_summary(result)
            for metric_type, result in (summary_results or {}).items()
        },
    }


def _dorsal_failure_diagnostic(exc: AnalysisError) -> str | None:
    """Compatibility seam for the shared allowlisted diagnostic classifier."""

    return dorsal_failure_diagnostic(exc)


def _is_dorsal_input_failure(exc: AnalysisError) -> bool:
    """Return True only when a stable capture diagnostic is available."""

    return _dorsal_failure_diagnostic(exc) is not None


def _plantar_failure_diagnostic(
    exc: PlantarFullSamAnalysisError,
    stage: str,
) -> str | None:
    """Classify a plantar capture fault using safe text patterns and stage."""

    message = str(exc).casefold()
    if any(
        marker in message
        for marker in (
            "empty image",
            "invalid image",
            "could not decode",
            "unsupported image",
            "not a supported decodable image",
        )
    ):
        return "PLANTAR_DECODE_FAILED"
    if "decoded image dimensions are unsafe" in message:
        return "PLANTAR_DIMENSIONS_UNSAFE"
    if any(
        marker in message
        for marker in (
            "no reliable",
            "did not detect",
            "could not detect",
            "no foot",
            "plantar silhouette is empty",
            "plantar silhouette is too small",
            "not a complete plantar-facing sole",
            "toe/heel orientation is ambiguous",
            "could not segment a plantar sole",
        )
    ):
        return (
            "RIGHT_PLANTAR_FOOT_NOT_DETECTED"
            if stage == "plantar-right-analysis"
            else "LEFT_PLANTAR_FOOT_NOT_DETECTED"
        )
    return None


def _is_plantar_input_failure(exc: PlantarFullSamAnalysisError) -> bool:
    """Compatibility helper for capture/model failure discrimination."""

    # Side does not affect whether this is invalid input.
    return _plantar_failure_diagnostic(exc, "plantar-left-analysis") is not None


def _photo_job_failure_log_message(
    *,
    stage: str,
    diagnostic_code: str | None,
    image_field: str | None,
) -> str:
    """Return an operator-friendly message without raw exception content."""

    guidance = photo_failure_guidance(diagnostic_code, image_field)
    if guidance is not None:
        reason = guidance.failure_detail
    elif stage in {"dorsal-integrated-analysis", "measurement-validation"}:
        reason = "발등 사진 전처리 또는 발 인식 과정에서 오류가 발생했습니다."
    elif stage.startswith("plantar-"):
        reason = "발바닥 사진에서 발 영역을 분석하는 과정에서 오류가 발생했습니다."
    elif stage.startswith("sole-tinea-"):
        reason = "발바닥 피부 상태를 분석하는 과정에서 오류가 발생했습니다."
    elif stage == "gpt-report-text":
        reason = "사진 분석 설명 문구를 생성하는 과정에서 오류가 발생했습니다."
    elif stage.startswith("core-save-") or stage.startswith("summary-save-"):
        reason = "사진 분석 결과를 백엔드에 저장하는 과정에서 오류가 발생했습니다."
    elif stage in {
        "analysis-image-composition",
        "analysis-fact-build",
        "backend-payload-build",
    }:
        reason = "사진 분석 결과를 구성하는 과정에서 오류가 발생했습니다."
    else:
        reason = "사진 분석 작업을 처리하는 과정에서 오류가 발생했습니다."
    return f"{reason} 측정 실패로 처리했습니다."


def _hallux_fallback_safety_score(left_angle: float, right_angle: float) -> float:
    score = 100.0 - 2.5 * max(float(left_angle), float(right_angle))
    return round(min(100.0, max(0.0, score)), 1)


def _hallux_summary_score(
    response: httpx.Response,
    left_angle: float,
    right_angle: float,
) -> float:
    try:
        body = response.json()
        value = body["result"]["riskScore"]
        if isinstance(value, bool):
            raise TypeError
        score = float(value)
        if math.isfinite(score) and 0.0 <= score <= 100.0:
            return round(score, 1)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return _hallux_fallback_safety_score(left_angle, right_angle)


def _json_safe_fact(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_fact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_fact(item) for item in value]
    return str(value)


def _tinea_side_facts(result: Any) -> dict[str, object]:
    return {
        "fungal_suspicion_safety_score": int(result.fungal_safety_score),
        "skin_reaction_safety_score": int(result.skin_reaction_safety_score),
        "inflammation_analysis_enabled": _json_safe_fact(
            result.metrics.get("inflammation_analysis_enabled", True)
        ),
        "inflammation_analysis_mode": _json_safe_fact(
            result.metrics.get("inflammation_analysis_mode", "enabled")
        ),
        "skin_reaction_score_label": _json_safe_fact(
            result.metrics.get("inflammation_score_label")
        ),
        "skin_assessment_source": _json_safe_fact(result.metrics.get("wound_model")),
        "fungal_analysis_enabled": _json_safe_fact(result.metrics.get("fungal_analysis_enabled", True)),
        "fungal_assessment_source": _json_safe_fact(result.metrics.get("fungal_model")),
        "sam_inflammation_class_used": False,
        "wound_detected": _json_safe_fact(result.metrics.get("wound_detected")),
        "wound_area_ratio": _json_safe_fact(result.metrics.get("wound_area_ratio")),
        "wound_threshold": _json_safe_fact(result.metrics.get("wound_threshold")),
        "lesion_overlap_policy": _json_safe_fact(result.metrics.get("lesion_overlap_policy")),
        "contained_wound_pixels_reclassified_as_fungal": _json_safe_fact(
            result.metrics.get("contained_wound_pixels_reclassified_as_fungal")
        ),
        "analysis_mode": _json_safe_fact(result.metrics.get("analysis_mode")),
        "score_reliability": _json_safe_fact(
            result.metrics.get("score_reliability")
        ),
        "fungal_regions": _json_safe_fact(result.metrics.get("fungal_regions", [])),
        "skin_reaction_regions": _json_safe_fact(
            result.metrics.get("inflammation_regions", [])
        ),
    }


def _local_analysis_payload(
    analysis: IntegratedFootAnalysisResult,
    *,
    fungal_score: int,
    skin_score: int,
    left_plantar_facts: dict[str, object],
    right_plantar_facts: dict[str, object],
) -> dict[str, object]:
    left_geometry = analysis.geometry.feet["left"]
    right_geometry = analysis.geometry.feet["right"]
    left_hallux = analysis.hallux_valgus["left"]
    right_hallux = analysis.hallux_valgus["right"]
    return {
        "measurementValid": analysis.geometry.measurement_valid,
        "metrics": {
            "measuredLeftFootSizeMm": left_geometry.length_mm,
            "measuredRightFootSizeMm": right_geometry.length_mm,
            "leftFootWidthMm": left_geometry.ball_width_mm,
            "rightFootWidthMm": right_geometry.ball_width_mm,
        },
        "halluxValgus": {
            "leftToeAngleDegree": left_hallux.angle_degree,
            "rightToeAngleDegree": right_hallux.angle_degree,
        },
        "tineaPedis": {
            "fungalSuspicionSafetyScore": fungal_score,
            "skinReactionSafetyScore": skin_score,
        },
        "plantarFootprint": {
            "left": left_plantar_facts,
            "right": right_plantar_facts,
        },
    }


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="상단 사진과 좌우 발바닥 사진 통합 분석 및 저장",
    responses={
        202: {
            "description": "사진 검증 및 백엔드 촬영 회차 상태 확인 후 분석 대기열 등록",
            "headers": _PHOTO_RESPONSE_HEADERS,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": [
                            "accepted",
                            "status",
                            "captureValidationStatus",
                            "mayStartPressure",
                            "measurementSessionId",
                            "photoCaptureAttempt",
                            "measurementStatus",
                            "statusCallbackConfirmed",
                            "pipeline",
                            "requestId",
                        ],
                        "properties": {
                            "accepted": {"type": "boolean"},
                            "status": {
                                "type": "string",
                                "enum": ["QUEUED"],
                            },
                            "captureValidationStatus": {
                                "type": "string",
                                "enum": ["PASSED"],
                            },
                            "mayStartPressure": {
                                "type": "boolean",
                                "enum": [True],
                            },
                            "measurementSessionId": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "photoCaptureAttempt": {
                                "type": "integer", "minimum": 0, "maximum": 3,
                            },
                            "measurementStatus": {
                                "type": "string", "enum": ["WAITING_FOR_ENVIRONMENT"],
                            },
                            "statusCallbackConfirmed": {
                                "type": "boolean", "enum": [True],
                            },
                            "pipeline": {"type": "string"},
                            "requestId": {"type": "string"},
                        },
                    }
                }
            },
        },
        409: {
            "description": "동일 세션 작업 중복 또는 백엔드 촬영 회차 불일치 (MEASUREMENT4009)",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        400: {
            "description": "빈 이미지",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        413: {
            "description": "업로드 용량 초과",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        422: {
            "description": "multipart 필드·촬영 회차·사진 사전검증 실패 또는 재촬영 필요",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        500: {
            "description": "업로드 읽기 또는 백그라운드 작업 등록 실패",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        502: {
            "description": "백엔드 촬영 검증 완료 상태를 확인하지 못해 분석 등록 중단",
            "model": PhotoAnalysisErrorResponse,
            "headers": _PHOTO_RESPONSE_HEADERS,
        },
        503: {
            "description": "사진 분석 대기열이 가득 참",
            "model": PhotoAnalysisErrorResponse,
            "headers": {
                **_PHOTO_RESPONSE_HEADERS,
                "Retry-After": {
                    "description": "재시도 전 최소 대기 시간(초)",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
async def create_photo_analysis(
    background_tasks: BackgroundTasks,
    measurementSessionId: int = Form(..., ge=1),
    photoCaptureAttempt: int = Form(
        ..., ge=0, le=3,
        description="백엔드가 지정한 촬영 회차. 최초 0, 재촬영 1~3을 그대로 전달합니다.",
    ),
    dorsalFootImage: UploadFile = File(...),
    leftPlantarFootImage: UploadFile = File(...),
    rightPlantarFootImage: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> JSONResponse:
    request_id, output_dir = new_request_dir()
    completed_stages: list[str] = []
    authorization = f"{credentials.scheme} {credentials.credentials}"
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage="request",
        event="request_started",
        message="발 사진 3장을 전달받아 파일 읽기를 시작합니다.",
        details={"expectedUploadCount": 3, "photoCaptureAttempt": photoCaptureAttempt},
    )

    upload_specs = (
        (
            "input-read-dorsal",
            dorsalFootImage,
            "dorsalFootImage",
            "발등 사진",
        ),
        (
            "input-read-left-plantar",
            leftPlantarFootImage,
            "leftPlantarFootImage",
            "왼발 발바닥 사진",
        ),
        (
            "input-read-right-plantar",
            rightPlantarFootImage,
            "rightPlantarFootImage",
            "오른발 발바닥 사진",
        ),
    )
    upload_values: list[bytes] = []
    for stage, upload, field_name, image_label in upload_specs:
        try:
            upload_values.append(await _read_upload(upload, field_name))
        except HTTPException as exc:
            diagnostic_code = (
                "PHOTO_UPLOAD_EMPTY"
                if exc.status_code == status.HTTP_400_BAD_REQUEST
                else "PHOTO_UPLOAD_TOO_LARGE"
            )
            guidance = photo_failure_guidance(diagnostic_code, field_name)
            _log_photo_progress(
                request_id=request_id,
                measurement_session_id=measurementSessionId,
                stage=stage,
                event="upload_rejected",
                message=guidance.failure_detail,
                details={
                    "diagnosticCode": diagnostic_code,
                    "statusCode": int(exc.status_code),
                },
            )
            error = _structured_photo_error(
                photo_capture_attempt=photoCaptureAttempt,
                status_code=exc.status_code,
                stage=stage,
                completed_stages=completed_stages,
                request_id=request_id,
                error_type="UploadValidationError",
                retryable=False,
                message=guidance.user_message,
                recommended_action=guidance.recommended_action,
                diagnostic_code=diagnostic_code,
                image_field=field_name,
                recapture_required=True,
                may_start_pressure=False,
            )
            await _notify_photo_failure(
                photo_capture_attempt=photoCaptureAttempt,
                measurement_session_id=measurementSessionId,
                authorization=authorization,
                request_id=request_id,
                failure_reason="CAMERA_ERROR",
                stage=stage,
                error_type="UploadValidationError",
                diagnostic_code=diagnostic_code,
                image_field=field_name,
            )
            raise error from exc
        except Exception as exc:
            error_type = type(exc).__name__
            error = _structured_photo_error(
                photo_capture_attempt=photoCaptureAttempt,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                stage=stage,
                completed_stages=completed_stages,
                request_id=request_id,
                error_type=error_type,
                retryable=True,
                message=f"Could not read {field_name}.",
            )
            await _notify_photo_failure(
                photo_capture_attempt=photoCaptureAttempt,
                measurement_session_id=measurementSessionId,
                authorization=authorization,
                request_id=request_id,
                failure_reason="AI_SERVER_ERROR",
                stage=stage,
                error_type=error_type,
            )
            raise error from exc
        _append_completed(completed_stages, stage)
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=stage,
            event="input_read_completed",
            message=f"{image_label} 파일 읽기를 완료했습니다.",
            details={"byteCount": len(upload_values[-1])},
        )
    dorsal_bytes, left_plantar_bytes, right_plantar_bytes = upload_values
    dorsal_filename = str(dorsalFootImage.filename or "dorsal-foot-image")

    preflight_stage = "capture-preflight"
    await _reserve_photo_analysis_session(
        measurement_session_id=measurementSessionId,
        request_id=request_id,
        completed_stages=completed_stages,
    )
    try:
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=preflight_stage,
            event="capture_preflight_started",
            message="사진 3장의 촬영 품질 사전검증을 시작합니다.",
        )
        capture_started_at = perf_counter()
        preflight = await run_in_threadpool(
            validate_photo_capture_preflight,
            dorsal_bytes,
            left_plantar_bytes,
            right_plantar_bytes,
        )
        left_plantar_prepared_input = getattr(
            preflight, "left_plantar_prepared_input", None
        )
        right_plantar_prepared_input = getattr(
            preflight, "right_plantar_prepared_input", None
        )
        _append_completed(completed_stages, preflight_stage)
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=preflight_stage,
            event="capture_preflight_passed",
            message="발등·왼발바닥·오른발바닥 사진 검증을 모두 통과했습니다.",
            details={
                "elapsedSeconds": round(perf_counter() - capture_started_at, 3),
                "captureValidationStatus": "PASSED",
                "actualWidth": preflight.dorsal_width,
                "actualHeight": preflight.dorsal_height,
                "detectedMarkerCount": preflight.detected_marker_count,
                "requiredMarkerCount": REQUIRED_KNOWN_ARUCO_ANCHORS,
                "detectedFootCount": preflight.dorsal_detected_foot_count,
                "requiredFootCount": 2,
                "leftPlantarFootConfidence": (
                    preflight.left_plantar_foot_confidence
                ),
                "rightPlantarFootConfidence": (
                    preflight.right_plantar_foot_confidence
                ),
                "leftPlantarRotationDegreesClockwise": getattr(
                    left_plantar_prepared_input, "rotation_degrees_clockwise", None
                ),
                "leftPlantarBrightnessGain": getattr(
                    left_plantar_prepared_input, "brightness_gain", None
                ),
                "rightPlantarRotationDegreesClockwise": getattr(
                    right_plantar_prepared_input, "rotation_degrees_clockwise", None
                ),
                "rightPlantarBrightnessGain": getattr(
                    right_plantar_prepared_input, "brightness_gain", None
                ),
            },
        )

        preflight_stage = "dorsal-measurement-preflight"
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=preflight_stage,
            event="measurement_preflight_started",
            message="사진 3장 검증 완료 후 발등 치수 사전검증을 시작합니다.",
        )
        measurement_started_at = perf_counter()
        prepared_geometry = await run_in_threadpool(
            validate_dorsal_measurement_preflight,
            dorsal_bytes,
            dorsal_filename,
        )
    except asyncio.CancelledError:
        await _release_photo_analysis_session(measurementSessionId)
        raise
    except PhotoCapturePreflightError as exc:
        marker_diagnostic_kwargs = _photo_marker_diagnostic_kwargs(exc)
        if exc.diagnostic_code == "DORSAL_MARKER_OCCLUDED_BY_FOOT" and not (
            exc.marker_occlusion_detected and (
                exc.marker_occlusion_diagnostic is None
                or marker_diagnostic_kwargs.get("marker_occlusion_status") == "foot_occlusion"
            )
        ):
            # Contradictory diagnostic metadata must not send a foot-cover
            # explanation with a terminal callback or grant a retake.
            exc = replace(exc, diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
                          marker_occlusion_detected=False, occluded_marker_ids=())
        guidance = photo_failure_guidance(exc.diagnostic_code, exc.image_field)
        marker_reason = marker_diagnostic_kwargs.get("marker_occlusion_reason")
        marker_explanation = marker_occlusion_explanation(marker_reason)
        if guidance is not None and marker_explanation is not None:
            guidance = PhotoFailureGuidance(
                diagnostic_code=guidance.diagnostic_code,
                user_message=f"{guidance.user_message} {marker_explanation}",
                recommended_action=(
                    "AI 서버의 마커·발 검출 모델과 카메라 보정 설정을 확인한 뒤 새 측정을 시작해 주세요."
                    if marker_reason in _MARKER_DIAGNOSTIC_SERVER_ERRORS
                    else guidance.recommended_action
                ),
            )
        safe_details = {
            key: value
            for key, value in {
                "diagnosticCode": exc.diagnostic_code,
                "actualWidth": exc.actual_width,
                "actualHeight": exc.actual_height,
                "expectedWidth": exc.expected_width,
                "expectedHeight": exc.expected_height,
                "detectedMarkerCount": exc.detected_marker_count,
                "requiredMarkerCount": exc.required_marker_count,
                "detectedFootCount": exc.detected_foot_count,
                "requiredFootCount": exc.required_foot_count,
                "markerOcclusionStatus": marker_diagnostic_kwargs.get("marker_occlusion_status"),
                "markerOcclusionReason": marker_reason,
                "markerOcclusionFootDetected": marker_diagnostic_kwargs.get("marker_occlusion_foot_detected"),
                "markerOcclusionOverlapRatio": marker_diagnostic_kwargs.get("marker_occlusion_overlap_ratio"),
                "markerOcclusionDetected": marker_diagnostic_kwargs.get("marker_occlusion_detected"),
                "markerOcclusionMethod": marker_diagnostic_kwargs.get("marker_occlusion_method"),
                "markerVisibleCornerCount": marker_diagnostic_kwargs.get("marker_visible_corner_count"),
                "markerCoveredCornerCount": marker_diagnostic_kwargs.get("marker_covered_corner_count"),
                "markerPatternScore": marker_diagnostic_kwargs.get("marker_pattern_score"),
                "markerProjectionReason": marker_diagnostic_kwargs.get("marker_projection_reason"),
                "markerPartialCheckReason": marker_diagnostic_kwargs.get("marker_partial_check_reason"),
                "markerPartialCandidateCount": marker_diagnostic_kwargs.get("marker_partial_candidate_count"),
                "markerPartialEdgeCount": marker_diagnostic_kwargs.get("marker_partial_edge_count"),
                "markerPartialCoveredCandidateCount": marker_diagnostic_kwargs.get("marker_partial_covered_candidate_count"),
                "markerPartialPatternCandidateCount": marker_diagnostic_kwargs.get("marker_partial_pattern_candidate_count"),
            }.items()
            if value is not None
        }
        if marker_diagnostic_kwargs:
            for marker_id in (1, 4):
                safe_details[f"detectedMarkerId{marker_id}"] = marker_id in marker_diagnostic_kwargs["detected_marker_ids"]
                safe_details[f"missingMarkerId{marker_id}"] = marker_id in marker_diagnostic_kwargs["missing_marker_ids"]
        if settings.photo_capture_diagnostics_enabled and exc.image_field == "dorsalFootImage" and exc.diagnostic_code in _MARKER_CAPTURE_FAILURE_CODES:
            # Preserve the exact rejected input locally so the next unknown
            # diagnosis can be reproduced. No image bytes enter logs or callbacks.
            try:
                from app.services.photo_capture_diagnostic_capture import save_marker_capture_diagnostic

                capture_directory = await run_in_threadpool(
                    save_marker_capture_diagnostic, dorsal_bytes,
                    measurement_session_id=measurementSessionId,
                    photo_capture_attempt=photoCaptureAttempt,
                    request_id=request_id,
                    diagnostic_details=safe_details,
                    enabled=True,
                )
                if capture_directory is not None:
                    progress_logger.info(
                        "[사진 분석][세션 %s][요청 %s] 마커 진단 원본과 검사 결과를 로컬에 저장했습니다. | diagnostic_directory=%s",
                        measurementSessionId, request_id[:8], capture_directory,
                    )
                else:
                    progress_logger.warning(
                        "[사진 분석][세션 %s][요청 %s] 로컬 마커 진단 자료를 저장하지 못했습니다. | code=LOCAL_DIAGNOSTIC_CAPTURE_UNAVAILABLE",
                        measurementSessionId, request_id[:8],
                    )
            except asyncio.CancelledError:
                await _release_photo_analysis_session(measurementSessionId)
                raise
            except Exception:
                # Local diagnostics must never change the backend transition.
                progress_logger.warning(
                    "[사진 분석][세션 %s][요청 %s] 로컬 마커 진단 자료를 저장하지 못했습니다. | code=LOCAL_DIAGNOSTIC_CAPTURE_UNAVAILABLE",
                    measurementSessionId, request_id[:8],
                )
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=preflight_stage,
            event=(
                "measurement_preflight_rejected"
                if preflight_stage == "dorsal-measurement-preflight"
                else "capture_preflight_rejected"
            ),
            message=(
                guidance.failure_detail
                if guidance is not None
                else "사진 사전검증을 통과하지 못했습니다. 다시 촬영해 주세요."
            ),
            details=safe_details,
        )
        if (
            exc.image_field == "dorsalFootImage"
            and exc.diagnostic_code in ARUCO_RECAPTURE_DIAGNOSTIC_CODES
            and exc.marker_occlusion_detected
            and (
                exc.marker_occlusion_diagnostic is None
                or marker_diagnostic_kwargs.get("marker_occlusion_status") == "foot_occlusion"
            )
        ):
            # The backend owns the persisted retry count and final transition.
            # Hold the local reservation until the callback finishes so another
            # upload cannot race the state change. Never fall back to FAILED
            # merely because this best-effort callback is unavailable.
            try:
                outcome = await notify_measurement_recapture_required(
                    measurement_session_id=measurementSessionId,
                    authorization=authorization,
                    diagnostic_code=exc.diagnostic_code,
                    request_id=request_id,
                    photo_capture_attempt=photoCaptureAttempt,
                )
            except Exception as callback_exc:
                logger.warning(
                    "ArUco recapture callback failed: request_id=%s error_type=%s",
                    request_id, type(callback_exc).__name__,
                )
                outcome = MeasurementRecaptureOutcome()
            finally:
                await _release_photo_analysis_session(measurementSessionId)
            if outcome.attempt_conflict:
                raise _photo_capture_status_error(
                    outcome=outcome,
                    stage="capture-status-transition",
                    completed_stages=completed_stages,
                    request_id=request_id,
                    photo_capture_attempt=photoCaptureAttempt,
                ) from exc
            terminal = outcome.confirmed and outcome.status == "FAILED"
            waiting = outcome.confirmed and outcome.status == "WAITING_FOR_RECAPTURE"
            raise _structured_photo_error(
                photo_capture_attempt=photoCaptureAttempt,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                stage=preflight_stage,
                completed_stages=completed_stages,
                request_id=request_id,
                error_type=type(exc).__name__,
                retryable=waiting,
                message=(
                    "사진 촬영을 완료하지 못했습니다."
                    if terminal else guidance.user_message
                ),
                recommended_action=(
                    "기준 ArUco 마커를 반복적으로 인식하지 못했습니다. 새 측정을 시작해 주세요."
                    if terminal else guidance.recommended_action if waiting else
                    "재촬영 대기 상태 반영을 확인하지 못했습니다. 세션 상태를 확인한 뒤 다시 촬영해 주세요."
                ),
                diagnostic_code=exc.diagnostic_code,
                image_field=exc.image_field,
                actual_width=exc.actual_width,
                actual_height=exc.actual_height,
                detected_marker_count=exc.detected_marker_count,
                required_marker_count=exc.required_marker_count,
                recapture_required=not terminal,
                may_start_pressure=False,
                measurement_status=outcome.status if outcome.confirmed else None,
                same_session_recapture_allowed=waiting,
                status_callback_confirmed=outcome.confirmed,
                next_measurement_status="READY_FOR_RECAPTURE" if waiting else None,
                should_disconnect=terminal,
                photo_recapture_count=outcome.recapture_count,
                max_photo_recaptures=outcome.max_recaptures,
                failure_reason="INVALID_CAPTURE_DATA" if terminal else None,
                occluded_marker_ids=list(exc.occluded_marker_ids),
                **{
                    "marker_occlusion_detected": True,
                    "marker_occlusion_overlap_ratio": exc.marker_occlusion_overlap_ratio,
                    **marker_diagnostic_kwargs,
                },
            ) from exc
        await _release_photo_analysis_session(measurementSessionId)
        # Marker visibility/geometry failure does not establish camera damage.
        marker_failure = exc.diagnostic_code in _MARKER_CAPTURE_FAILURE_CODES
        failure_reason = (
            "AI_SERVER_ERROR" if marker_failure and marker_reason in _MARKER_DIAGNOSTIC_SERVER_ERRORS
            else "INVALID_CAPTURE_DATA" if marker_failure else "CAMERA_ERROR"
        )
        callback_confirmed = await _notify_photo_failure(
            photo_capture_attempt=photoCaptureAttempt,
            measurement_session_id=measurementSessionId,
            authorization=authorization,
            request_id=request_id,
            failure_reason=failure_reason,
            stage=preflight_stage,
            error_type=type(exc).__name__,
            diagnostic_code=exc.diagnostic_code,
            image_field=exc.image_field,
            **({"marker_occlusion_reason": marker_reason} if marker_reason is not None else {}),
        )
        raise _structured_photo_error(
            photo_capture_attempt=photoCaptureAttempt,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            stage=preflight_stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=False,
            message=(
                guidance.user_message
                if guidance is not None
                else "사진 촬영 사전 검증을 통과하지 못했습니다."
            ),
            recommended_action=(
                guidance.recommended_action if guidance is not None else None
            ),
            diagnostic_code=exc.diagnostic_code,
            image_field=exc.image_field,
            expected_width=exc.expected_width,
            expected_height=exc.expected_height,
            actual_width=exc.actual_width,
            actual_height=exc.actual_height,
            detected_marker_count=exc.detected_marker_count,
            required_marker_count=exc.required_marker_count,
            detected_foot_count=exc.detected_foot_count,
            required_foot_count=exc.required_foot_count,
            recapture_required=True,
            may_start_pressure=False,
            failure_reason=failure_reason if marker_failure else None,
            measurement_status="FAILED" if marker_failure and callback_confirmed is True else None,
            status_callback_confirmed=(callback_confirmed is True) if marker_failure else None,
            same_session_recapture_allowed=False if marker_failure else None,
            should_disconnect=True if marker_failure and callback_confirmed is True else None,
            **marker_diagnostic_kwargs,
        ) from exc
    except Exception as exc:
        await _release_photo_analysis_session(measurementSessionId)
        error_type = type(exc).__name__
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=preflight_stage,
            event=(
                "measurement_preflight_failed"
                if preflight_stage == "dorsal-measurement-preflight"
                else "capture_preflight_failed"
            ),
            message=(
                "발등 치수 사전검증 중 내부 오류가 발생했습니다."
                if preflight_stage == "dorsal-measurement-preflight"
                else "사진 3장 사전검증 중 내부 오류가 발생했습니다."
            ),
            details={"statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR},
        )
        await _notify_photo_failure(
            photo_capture_attempt=photoCaptureAttempt,
            measurement_session_id=measurementSessionId,
            authorization=authorization,
            request_id=request_id,
            failure_reason="AI_SERVER_ERROR",
            stage=preflight_stage,
            error_type=error_type,
        )
        raise _structured_photo_error(
            photo_capture_attempt=photoCaptureAttempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=preflight_stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=error_type,
            retryable=True,
            message=(
                "Dorsal measurement preflight could not run."
                if preflight_stage == "dorsal-measurement-preflight"
                else "Photo capture preflight could not run."
            ),
            recapture_required=False,
            may_start_pressure=False,
        ) from exc
    _append_completed(completed_stages, preflight_stage)
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=preflight_stage,
        event="measurement_preflight_passed",
        message="발등 치수 사전검증을 통과해 본 분석 대기열 등록이 가능합니다.",
        details={
            "elapsedSeconds": round(perf_counter() - measurement_started_at, 3),
            "measurementValid": prepared_geometry.measurement_valid,
            "leftFootLengthMm": prepared_geometry.feet["left"].length_mm,
            "rightFootLengthMm": prepared_geometry.feet["right"].length_mm,
            "leftFootWidthMm": prepared_geometry.feet["left"].ball_width_mm,
            "rightFootWidthMm": prepared_geometry.feet["right"].ball_width_mm,
        },
    )

    # Persist the accepted capture state before the job can analyze or save.
    # Failed callbacks never enter the generic terminal-failure handler.
    transition_stage = "capture-status-transition"
    try:
        outcome = await notify_photo_capture_passed(
            measurement_session_id=measurementSessionId,
            authorization=authorization,
            photo_capture_attempt=photoCaptureAttempt,
            request_id=request_id,
        )
    except asyncio.CancelledError:
        await _release_photo_analysis_session(measurementSessionId)
        raise
    except Exception:
        outcome = PhotoCaptureStatusOutcome()
    if not outcome.confirmed or outcome.status != "WAITING_FOR_ENVIRONMENT" or outcome.attempt_conflict:
        await _release_photo_analysis_session(measurementSessionId)
        raise _photo_capture_status_error(
            outcome=outcome,
            stage=transition_stage,
            completed_stages=completed_stages,
            request_id=request_id,
            photo_capture_attempt=photoCaptureAttempt,
        )
    _append_completed(completed_stages, transition_stage)
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=transition_stage,
        event="capture_status_confirmed",
        message="백엔드가 촬영 검증 완료 상태를 확인했습니다. 사진 분석을 등록합니다.",
        details={"photoCaptureAttempt": photoCaptureAttempt},
    )

    work_item = PhotoAnalysisWorkItem(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        photo_capture_attempt=photoCaptureAttempt,
        output_dir=output_dir,
        dorsal_bytes=dorsal_bytes,
        left_plantar_bytes=left_plantar_bytes,
        right_plantar_bytes=right_plantar_bytes,
        dorsal_filename=dorsal_filename,
        authorization=authorization,
        completed_stages=tuple(completed_stages),
        prepared_geometry=prepared_geometry,
        left_plantar_prepared_input=left_plantar_prepared_input,
        right_plantar_prepared_input=right_plantar_prepared_input,
    )
    try:
        background_tasks.add_task(_run_photo_analysis_job, work_item)
    except Exception as exc:
        await _release_photo_analysis_session(measurementSessionId)
        error_type = type(exc).__name__
        error = _structured_photo_error(
            photo_capture_attempt=photoCaptureAttempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage="background-schedule",
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=error_type,
            retryable=True,
            message="Could not schedule the photo analysis job.",
        )
        await _notify_photo_failure(
            photo_capture_attempt=photoCaptureAttempt,
            measurement_session_id=measurementSessionId,
            authorization=authorization,
            request_id=request_id,
            failure_reason="AI_SERVER_ERROR",
            stage="background-schedule",
            error_type=error_type,
        )
        raise error from exc

    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage="queue",
        event="job_queued",
        message="사진 분석 요청을 접수했습니다. 백그라운드 분석 대기열에 등록했습니다.",
        details={"photoCaptureAttempt": photoCaptureAttempt},
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "accepted": True,
            "status": "QUEUED",
            "captureValidationStatus": "PASSED",
            "mayStartPressure": True,
            "measurementSessionId": measurementSessionId,
            "photoCaptureAttempt": photoCaptureAttempt,
            "measurementStatus": "WAITING_FOR_ENVIRONMENT",
            "statusCallbackConfirmed": True,
            "pipeline": PIPELINE_VERSION,
            "requestId": request_id,
        },
        headers={
            "X-Feetfit-Pipeline": PIPELINE_VERSION,
            "X-Feetfit-Request-Id": request_id,
        },
    )


async def _execute_photo_pipeline(
    work_item: PhotoAnalysisWorkItem,
) -> JSONResponse:
    pipeline_started_at = perf_counter()
    request_id = work_item.request_id
    output_dir = work_item.output_dir
    measurementSessionId = work_item.measurement_session_id
    completed_stages = list(work_item.completed_stages)
    dorsal_bytes = work_item.dorsal_bytes
    left_plantar_bytes = work_item.left_plantar_bytes
    right_plantar_bytes = work_item.right_plantar_bytes

    stage = "dorsal-integrated-analysis"
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="analysis_started",
        message="발등 사진에서 기준 마커·발 치수·무지외반·피부 분석을 시작합니다.",
    )
    try:
        if work_item.prepared_geometry is None:
            # Compatibility path for directly constructed legacy work items.
            integrated = await run_in_threadpool(
                analyze_integrated_foot_photo,
                dorsal_bytes,
                work_item.dorsal_filename,
            )
        else:
            integrated = await run_in_threadpool(
                analyze_prepared_foot_geometry,
                work_item.prepared_geometry,
            )
    except AnalysisError as exc:
        diagnostic_code = _dorsal_failure_diagnostic(exc)
        client_input_error = diagnostic_code is not None
        guidance = photo_failure_guidance(
            diagnostic_code,
            "dorsalFootImage",
        )
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
                if client_input_error
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=not client_input_error,
            message=(
                guidance.user_message
                if guidance is not None
                else "발등 사진 분석 중 AI 서버 오류가 발생했습니다."
            ),
            recommended_action=(
                guidance.recommended_action if guidance is not None else None
            ),
            diagnostic_code=diagnostic_code,
            image_field=("dorsalFootImage" if diagnostic_code is not None else None),
            recapture_required=(True if diagnostic_code is not None else None),
            may_start_pressure=(False if diagnostic_code is not None else None),
        ) from exc
    except Exception as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=True,
            message="발등 사진 분석 중 AI 서버 오류가 발생했습니다.",
        ) from exc
    _append_completed(completed_stages, stage)

    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="analysis_completed",
        message="발등 무좀·피부·무지외반 통합 분석을 완료했습니다.",
        details={"elapsedSeconds": round(perf_counter() - pipeline_started_at, 3)},
    )
    geometry = integrated.geometry
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="aruco_analyzed",
        message="발등 사진의 기준 마커와 촬영 배율 검사를 완료했습니다.",
        details={
            "measurementValid": geometry.measurement_valid,
            "measurementStatus": (
                "valid" if geometry.measurement_valid else "invalid"
            ),
            "detectedMarkerCount": len(geometry.detected_marker_ids),
            "missingMarkerCount": len(geometry.missing_marker_ids),
        },
    )
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="foot_metrics_analyzed",
        message="왼발과 오른발의 발 길이·발볼 측정을 완료했습니다.",
        details={
            "leftFootLengthMm": geometry.feet["left"].length_mm,
            "rightFootLengthMm": geometry.feet["right"].length_mm,
            "leftFootWidthMm": geometry.feet["left"].ball_width_mm,
            "rightFootWidthMm": geometry.feet["right"].ball_width_mm,
        },
    )
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="hallux_valgus_analyzed",
        message="왼발과 오른발의 무지외반 각도 분석을 완료했습니다.",
        details={
            "leftToeAngleDegree": integrated.hallux_valgus["left"].angle_degree,
            "rightToeAngleDegree": integrated.hallux_valgus["right"].angle_degree,
        },
    )
    for dorsal_side in ("left", "right"):
        dorsal_tinea_result = integrated.tinea[dorsal_side]
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=f"dorsal-tinea-{dorsal_side}-analysis",
            event="dorsal_tinea_analyzed",
            message=(
                "왼발 발등의 무좀·피부 상태 분석을 완료했습니다."
                if dorsal_side == "left"
                else "오른발 발등의 무좀·피부 상태 분석을 완료했습니다."
            ),
            details={
                "fungalSafetyScore": dorsal_tinea_result.fungal_safety_score,
                "skinReactionSafetyScore": (
                    dorsal_tinea_result.skin_reaction_safety_score
                ),
            },
        )
    stage = "measurement-validation"
    if not geometry.measurement_valid:
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=stage,
            event="measurement_rejected",
            message="발 치수 측정값을 검증했지만 저장 기준을 통과하지 못했습니다.",
            details={
                "measurementValid": False,
                "measurementStatus": "invalid",
                "detectedMarkerCount": len(geometry.detected_marker_ids),
                "missingMarkerCount": len(geometry.missing_marker_ids),
            },
        )
        diagnostic_code = (
            "DORSAL_MARKERS_INSUFFICIENT"
            if geometry.missing_marker_ids
            else "DORSAL_ARUCO_GEOMETRY_INVALID"
        )
        guidance = photo_failure_guidance(diagnostic_code, "dorsalFootImage")
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type="MeasurementValidationError",
            retryable=False,
            message=guidance.user_message,
            recommended_action=guidance.recommended_action,
            diagnostic_code=diagnostic_code,
            image_field="dorsalFootImage",
            recapture_required=True,
            may_start_pressure=False,
            measurement_status=geometry.measurement_status,
            measurement_invalid_reasons=list(geometry.measurement_invalid_reasons),
            detected_marker_ids=list(geometry.detected_marker_ids),
            missing_marker_ids=list(geometry.missing_marker_ids),
        )
    _append_completed(completed_stages, stage)
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="measurement_validated",
        message="발 치수 측정값 검증을 완료했으며 저장 가능한 상태입니다.",
        details={"measurementValid": True, "measurementStatus": "valid"},
    )

    stage = "plantar-left-analysis"
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="plantar_segmentation_started",
        message="왼발 발바닥 눌림 영역 분할 분석을 시작합니다.",
    )
    try:
        left_plantar_kwargs = (
            {"prepared_input": work_item.left_plantar_prepared_input}
            if work_item.left_plantar_prepared_input is not None
            else {}
        )
        left_plantar_started_at = perf_counter()
        left_plantar = await run_in_threadpool(
            analyze_plantar_image,
            left_plantar_bytes,
            "left",
            output_dir,
            **left_plantar_kwargs,
        )
        _append_completed(completed_stages, stage)
        left_progress_facts = _plantar_facts(left_plantar)
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=stage,
            event="plantar_segmentation_completed",
            message="왼발 발바닥 눌림 영역 분할 분석을 완료했습니다.",
            details={
                "regionCount": left_progress_facts["region_count"],
                "elapsedSeconds": round(perf_counter() - left_plantar_started_at, 3),
                "contactAreaRatio": left_progress_facts["contact_area_ratio"],
                "footprintStatus": left_progress_facts["status"],
            },
        )
        stage = "plantar-right-analysis"
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=stage,
            event="plantar_segmentation_started",
            message="오른발 발바닥 눌림 영역 분할 분석을 시작합니다.",
        )
        right_plantar_kwargs = (
            {"prepared_input": work_item.right_plantar_prepared_input}
            if work_item.right_plantar_prepared_input is not None
            else {}
        )
        right_plantar_started_at = perf_counter()
        right_plantar = await run_in_threadpool(
            analyze_plantar_image,
            right_plantar_bytes,
            "right",
            output_dir,
            **right_plantar_kwargs,
        )
        _append_completed(completed_stages, stage)
        right_progress_facts = _plantar_facts(right_plantar)
        _log_photo_progress(
            request_id=request_id,
            measurement_session_id=measurementSessionId,
            stage=stage,
            event="plantar_segmentation_completed",
            message="오른발 발바닥 눌림 영역 분할 분석을 완료했습니다.",
            details={
                "regionCount": right_progress_facts["region_count"],
                "elapsedSeconds": round(perf_counter() - right_plantar_started_at, 3),
                "contactAreaRatio": right_progress_facts["contact_area_ratio"],
                "footprintStatus": right_progress_facts["status"],
            },
        )

        stage = "plantar-artifact-read"
        left_footprint_png, right_footprint_png, left_sole_input, right_sole_input = await asyncio.gather(
            run_in_threadpool(_read_png, left_plantar.visualization_path,
                             "left plantar footprint visualization", require_alpha=True),
            run_in_threadpool(_read_png, right_plantar.visualization_path,
                             "right plantar footprint visualization", require_alpha=True),
            run_in_threadpool(_read_png, left_plantar.tinea_input_path, "left unassessed plantar photo"),
            run_in_threadpool(_read_png, right_plantar.tinea_input_path, "right unassessed plantar photo"),
        )
        _append_completed(completed_stages, stage)
        dorsal_left_tinea = integrated.tinea["left"]
        dorsal_right_tinea = integrated.tinea["right"]
        stage = "analysis-image-composition"
        dorsal_suspicion_map, dorsal_photo_overlay, dorsal_clean_original, sole_clean_original = await asyncio.gather(
            run_in_threadpool(combine_png_images_side_by_side,
                              dorsal_left_tinea.suspicion_map_png, dorsal_right_tinea.suspicion_map_png),
            run_in_threadpool(combine_png_images_side_by_side,
                              dorsal_left_tinea.photo_overlay_png, dorsal_right_tinea.photo_overlay_png),
            run_in_threadpool(combine_png_images_side_by_side,
                              dorsal_left_tinea.clean_original_png, dorsal_right_tinea.clean_original_png),
            run_in_threadpool(combine_png_images_side_by_side, left_sole_input, right_sole_input),
        )
        # Historical multipart fields are retained for backend compatibility.
        # Their plantar content is an unannotated photograph, never a skin prediction.
        sole_suspicion_map = sole_clean_original
        sole_photo_overlay = sole_clean_original
        _append_completed(completed_stages, stage)
        _log_photo_progress(
            request_id=request_id, measurement_session_id=measurementSessionId,
            stage=stage, event="analysis_images_composed",
            message="발등 피부 분석 이미지와 발바닥 눌림 이미지를 준비했습니다.",
            details={"artifactCount": 4},
        )
    except PlantarFullSamAnalysisError as exc:
        diagnostic_code = _plantar_failure_diagnostic(exc, stage)
        input_failure = diagnostic_code is not None
        image_field = (
            "rightPlantarFootImage"
            if stage == "plantar-right-analysis"
            else "leftPlantarFootImage"
        )
        guidance = photo_failure_guidance(diagnostic_code, image_field)
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
                if input_failure
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=not input_failure,
            message=(
                guidance.user_message
                if guidance is not None
                else "발바닥 사진 분석 중 AI 서버 오류가 발생했습니다."
            ),
            recommended_action=(
                guidance.recommended_action if guidance is not None else None
            ),
            diagnostic_code=diagnostic_code,
            image_field=(image_field if diagnostic_code is not None else None),
            recapture_required=(True if diagnostic_code is not None else None),
            may_start_pressure=(False if diagnostic_code is not None else None),
        ) from exc
    except AnalysisError as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=True,
            message="Local skin-image analysis failed.",
        ) from exc
    except Exception as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=True,
            message="Local plantar or skin-image analysis failed.",
        ) from exc

    stage = "analysis-fact-build"
    try:
        all_tinea_results = (
            dorsal_left_tinea,
            dorsal_right_tinea,
        )
        fungal_score = aggregate_tinea_safety_scores(
            item.fungal_safety_score for item in all_tinea_results
        )
        skin_score = aggregate_tinea_safety_scores(
            item.skin_reaction_safety_score for item in all_tinea_results
        )
        left_plantar_facts = _plantar_facts(left_plantar)
        right_plantar_facts = _plantar_facts(right_plantar)
        left_hallux = integrated.hallux_valgus["left"]
        right_hallux = integrated.hallux_valgus["right"]
        local_hallux_safety_score = _hallux_fallback_safety_score(
            left_hallux.angle_degree,
            right_hallux.angle_degree,
        )
        left_geometry = geometry.feet["left"]
        right_geometry = geometry.feet["right"]
        facts = {
            "hallux_valgus": {
                "left_angle_degree": left_hallux.angle_degree,
                "right_angle_degree": right_hallux.angle_degree,
                "safety_score": local_hallux_safety_score,
            },
            "tinea_pedis": {
                "dorsal": {
                    "left": _tinea_side_facts(dorsal_left_tinea),
                    "right": _tinea_side_facts(dorsal_right_tinea),
                },
                "analysis_scope": "dorsal_only",
                "excluded_surfaces": ["plantar"],
                "final_scores": {
                    "fungal_suspicion_safety_score": fungal_score,
                    "skin_reaction_safety_score": skin_score,
                },
            },
            "plantar_footprint": {
                "left": left_plantar_facts,
                "right": right_plantar_facts,
            },
        }
        if inflammation_report_uses_unannotated_images(facts):
            tinea_report_images = {
                "TINEA_DORSAL_CLEAN_UNANNOTATED_PHOTO": dorsal_clean_original,
            }
        else:
            tinea_report_images = {
                "DORSAL_TINEA_SUSPICION_MAP": dorsal_suspicion_map,
                "DORSAL_TINEA_PHOTO_OVERLAY": dorsal_photo_overlay,
            }
        related_analysis_images = {
            "HALLUX_LEFT_RESULT": left_hallux.analysis_png,
            "HALLUX_RIGHT_RESULT": right_hallux.analysis_png,
            **tinea_report_images,
            "LEFT_PLANTAR_FOOTPRINT": left_footprint_png,
            "RIGHT_PLANTAR_FOOTPRINT": right_footprint_png,
        }
    except Exception as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=False,
            message="Could not build deterministic photo-analysis facts.",
        ) from exc
    _append_completed(completed_stages, stage)
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="analysis_facts_built",
        message="분석 수치와 최종 안전 점수 계산을 완료했습니다.",
        details={
            "fungalSafetyScore": fungal_score,
            "skinReactionSafetyScore": skin_score,
            "leftToeAngleDegree": left_hallux.angle_degree,
            "rightToeAngleDegree": right_hallux.angle_degree,
        },
    )

    # This is the route's only GPT service call. It completes before any write.
    stage = "gpt-report-text"
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="gpt_generation_started",
        message="분석 이미지와 수치를 바탕으로 리포트 문구 생성을 시작합니다.",
        details={
            "imageCount": 3,
            "analysisImageCount": len(related_analysis_images),
            "responseTimeoutDisabled": False,
            "generationTimeoutSeconds": settings.openai_report_full_photo_timeout_seconds,
            "connectTimeoutSeconds": settings.openai_report_connect_timeout_seconds,
            "poolTimeoutSeconds": settings.openai_report_pool_timeout_seconds,
        },
    )
    try:
        generation_started_at = perf_counter()
        generated = await generate_full_photo_analysis_texts(
            dorsal_raw_image_bytes=dorsal_bytes,
            left_plantar_raw_image_bytes=left_plantar_bytes,
            right_plantar_raw_image_bytes=right_plantar_bytes,
            facts=facts,
            related_analysis_images=related_analysis_images,
        )
        if generated.generation_source != "OPENAI" and not (
            settings.openai_report_fallback_enabled
            and generated.generation_source == "DETERMINISTIC_FALLBACK"
        ):
            raise RuntimeError("Required report text generation source was not accepted.")
    except Exception as exc:
        output_validation_failed = isinstance(
            exc,
            FullPhotoOutputValidationError,
        )
        validation_field = exc.field if output_validation_failed else None
        validation_rule = exc.rule_code if output_validation_failed else None
        diagnostic_code = (
            "GPT_REPORT_RESPONSE_INVALID"
            if output_validation_failed
            else "GPT_REPORT_GENERATION_FAILED"
        )
        guidance = photo_failure_guidance(diagnostic_code)
        assert guidance is not None
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            diagnostic_code=diagnostic_code,
            validation_field=validation_field,
            validation_rule=validation_rule,
            retryable=True,
            message=guidance.user_message,
            recommended_action=guidance.recommended_action,
            recapture_required=False,
            may_start_pressure=False,
        ) from exc
    _append_completed(completed_stages, stage)
    generation_failure_code = _safe_generation_failure_code(
        getattr(generated, "generation_failure_code", None)
    )
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="gpt_generation_completed",
        message=(
            "GPT 리포트 문구 생성을 완료했습니다."
            if generated.generation_source == "OPENAI"
            else "서버 분석 사실을 바탕으로 기본 리포트 문구 생성을 완료했습니다."
        ),
        details={
            "descriptionCount": 5,
            "elapsedSeconds": round(perf_counter() - generation_started_at, 3),
            "adviceMetricCount": len(generated.metric_advice),
            "generationSource": generated.generation_source,
            "errorCode": generation_failure_code,
        },
    )

    stage = "backend-payload-build"
    try:
        hallux_request = HalluxValgusReportRequest(
            measurement_session_id=measurementSessionId,
            left_toe_angle_degree=left_hallux.angle_degree,
            right_toe_angle_degree=right_hallux.angle_degree,
            score_analysis_text=generated.hallux_score_analysis_text,
        )
        metrics_request = FootMetricsReportRequest(
            measurement_session_id=measurementSessionId,
            measured_left_foot_size_mm=left_geometry.length_mm,
            measured_right_foot_size_mm=right_geometry.length_mm,
            left_foot_width_mm=left_geometry.ball_width_mm,
            right_foot_width_mm=right_geometry.ball_width_mm,
        )
        tinea_request = TineaPedisReportRequest(
            measurement_session_id=measurementSessionId,
            fungal_suspicion_safety_score=fungal_score,
            skin_reaction_safety_score=skin_score,
            fungal_suspicion_safety_description=(
                generated.fungal_suspicion_safety_description
            ),
            skin_reaction_safety_description=(
                generated.skin_reaction_safety_description
            ),
            total_score_description=generated.total_score_description,
        )
        plantar_request_json = json.dumps(
            {
                "measurementSessionId": measurementSessionId,
                "plantarFootprintAnalysisText": (
                    generated.plantar_footprint_analysis_text
                ),
            },
            ensure_ascii=False,
        )
        headers = {"accept": "*/*", "Authorization": work_item.authorization}
    except Exception as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=False,
            message="Could not build backend report payloads.",
        ) from exc
    _append_completed(completed_stages, stage)
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="backend_payloads_built",
        message="백엔드에 저장할 핵심 분석 결과 데이터 준비를 완료했습니다.",
        details={"coreReportCount": len(_CORE_SAVE_STAGES)},
    )

    core_results: dict[str, httpx.Response | BaseException] = {}
    summary_results: dict[str, httpx.Response | BaseException] = {}
    summary_skip_reasons: dict[str, str] = {}
    stage = "core-report-saves"
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage=stage,
        event="core_saves_started",
        message="무지외반·발 치수·무좀·발바닥 눌림 분석 결과를 백엔드에 저장하는 중입니다.",
        details={"reportCount": len(_CORE_SAVE_STAGES)},
    )
    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            core_operations = {
                "halluxValgus": client.post(
                    settings.hallux_valgus_report_endpoint,
                    headers=headers,
                    files={
                        "request": (
                            None,
                            json.dumps(
                                hallux_request.model_dump(by_alias=True),
                                ensure_ascii=False,
                            ),
                            "application/json",
                        ),
                        "leftFootImage": (
                            "hallux_valgus_left.png",
                            left_hallux.analysis_png,
                            "image/png",
                        ),
                        "rightFootImage": (
                            "hallux_valgus_right.png",
                            right_hallux.analysis_png,
                            "image/png",
                        ),
                    },
                ),
                "metrics": client.post(
                    settings.daily_metrics_report_endpoint,
                    headers=headers,
                    json=metrics_request.model_dump(by_alias=True),
                ),
                "tineaPedis": client.post(
                    settings.tinea_report_endpoint,
                    headers=headers,
                    files={
                        "request": (
                            None,
                            json.dumps(
                                tinea_request.model_dump(by_alias=True),
                                ensure_ascii=False,
                            ),
                            "application/json",
                        ),
                        "suspiciousAreaMapImage": (
                            "dorsal_suspicion_map.png",
                            dorsal_suspicion_map,
                            "image/png",
                        ),
                        # The backend field name is historical. The frontend
                        # compares this image with the suspicion map, so it
                        # must receive the matching annotated photo overlay.
                        "originalFootImage": (
                            "dorsal_analysis_overlay.png",
                            dorsal_photo_overlay,
                            "image/png",
                        ),
                        "soleSuspiciousAreaMapImage": (
                            "sole_not_assessed_photo.png",
                            sole_suspicion_map,
                            "image/png",
                        ),
                        "soleOriginalFootImage": (
                            "sole_not_assessed_photo.png",
                            sole_photo_overlay,
                            "image/png",
                        ),
                    },
                ),
                "plantarFootprint": client.post(
                    settings.plantar_footprint_report_endpoint,
                    headers=headers,
                    files={
                        "request": (None, plantar_request_json, "application/json"),
                        "leftPlantarFootprintImage": (
                            "left_plantar_footprint.png",
                            left_footprint_png,
                            "image/png",
                        ),
                        "rightPlantarFootprintImage": (
                            "right_plantar_footprint.png",
                            right_footprint_png,
                            "image/png",
                        ),
                    },
                ),
            }
            core_values = await asyncio.gather(
                *core_operations.values(),
                return_exceptions=True,
            )
            core_results = dict(zip(core_operations, core_values, strict=True))
            # Core requests already in flight may finish together; once any
            # response rejects this attempt, issue no subsequent summary saves.
            if any(is_photo_capture_attempt_conflict(value) for value in core_results.values()):
                raise _structured_photo_error(
                    status_code=status.HTTP_409_CONFLICT,
                    stage="core-report-saves",
                    completed_stages=completed_stages,
                    request_id=request_id,
                    photo_capture_attempt=work_item.photo_capture_attempt,
                    error_type="PhotoCaptureAttemptConflict",
                    diagnostic_code="PHOTO_CAPTURE_ATTEMPT_CONFLICT",
                    retryable=False,
                    message="백엔드가 촬영 회차 충돌을 반환해 추가 결과 저장을 중단했습니다.",
                    reports=_safe_backend_reports(core_results),
                    partial_save=_partial_save_occurred(core_results),
                )
            for report_name, result in core_results.items():
                if isinstance(result, httpx.Response) and _is_2xx(result):
                    _append_completed(
                        completed_stages,
                        _CORE_SAVE_STAGES[report_name],
                    )
                _log_photo_progress(
                    request_id=request_id,
                    measurement_session_id=measurementSessionId,
                    stage=_CORE_SAVE_STAGES[report_name],
                    event="core_save_result",
                    message=_backend_save_progress_message(
                        _CORE_REPORT_KOREAN_LABELS[report_name],
                        result,
                    ),
                    details=_progress_attempt_details(result),
                )

            core_transport_errors = [
                (report_name, result)
                for report_name, result in core_results.items()
                if isinstance(result, BaseException)
            ]
            if core_transport_errors:
                report_name, transport_error = core_transport_errors[0]
                stage = _CORE_SAVE_STAGES[report_name]
                raise _structured_photo_error(
                    photo_capture_attempt=work_item.photo_capture_attempt,
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    stage=stage,
                    completed_stages=completed_stages,
                    request_id=request_id,
                    error_type=type(transport_error).__name__,
                    retryable=_is_retryable_exception(transport_error),
                    message="One or more core report transport requests failed.",
                    reports=_safe_backend_reports(core_results),
                    partial_save=_partial_save_occurred(core_results),
                )

            hallux_response = core_results["halluxValgus"]
            metrics_response = core_results["metrics"]
            tinea_response = core_results["tineaPedis"]
            plantar_response = core_results["plantarFootprint"]
            assert isinstance(hallux_response, httpx.Response)
            assert isinstance(metrics_response, httpx.Response)
            assert isinstance(tinea_response, httpx.Response)
            assert isinstance(plantar_response, httpx.Response)

            summary_requests: dict[str, PhotoMetricSummaryRequest] = {}
            if _is_2xx(tinea_response):
                summary_requests["ATHLETES_FOOT"] = PhotoMetricSummaryRequest(
                    measurement_session_id=measurementSessionId,
                    metric_type="ATHLETES_FOOT",
                    score=float(fungal_score),
                    advice=tuple(generated.metric_advice["ATHLETES_FOOT"]),
                )
                summary_requests["SKIN_IRRITATION"] = PhotoMetricSummaryRequest(
                    measurement_session_id=measurementSessionId,
                    metric_type="SKIN_IRRITATION",
                    score=float(skin_score),
                    advice=tuple(generated.metric_advice["SKIN_IRRITATION"]),
                )
            else:
                reason = "tineaPedis core report did not return 2xx"
                summary_skip_reasons["ATHLETES_FOOT"] = reason
                summary_skip_reasons["SKIN_IRRITATION"] = reason
            if _is_2xx(hallux_response):
                summary_requests["HALLUX_VALGUS"] = PhotoMetricSummaryRequest(
                    measurement_session_id=measurementSessionId,
                    metric_type="HALLUX_VALGUS",
                    score=_hallux_summary_score(
                        hallux_response,
                        left_hallux.angle_degree,
                        right_hallux.angle_degree,
                    ),
                    advice=tuple(generated.metric_advice["HALLUX_VALGUS"]),
                )
            else:
                summary_skip_reasons["HALLUX_VALGUS"] = (
                    "halluxValgus core report did not return 2xx"
                )

            for metric_type in summary_skip_reasons:
                _log_photo_progress(
                    request_id=request_id,
                    measurement_session_id=measurementSessionId,
                    stage=f"summary-save-{metric_type}",
                    event="summary_save_result",
                    message=(
                        f"{_SUMMARY_KOREAN_LABELS[metric_type]} 저장을 건너뛰었습니다. "
                        "먼저 저장해야 하는 핵심 분석 결과가 실패했습니다."
                    ),
                    details={"outcome": "skipped"},
                )

            stage = "summary-report-saves"
            _log_photo_progress(
                request_id=request_id,
                measurement_session_id=measurementSessionId,
                stage=stage,
                event="summary_saves_started",
                message="무좀 위험도·피부 자극도·무지외반 지표 리포트를 백엔드에 순서대로 저장하는 중입니다.",
                details={"summaryCount": len(summary_requests)},
            )

            # A summary save can create the parent report for a measurement session.
            # Sending multiple metric saves concurrently can therefore race in the
            # report server and create duplicate parent rows. Keep these requests
            # strictly ordered while still recording every individual outcome.
            summary_results = {}
            for metric_type, summary_request in summary_requests.items():
                try:
                    result = await client.post(
                        settings.summary_report_endpoint,
                        headers=headers,
                        json=summary_request.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                    )
                except Exception as exc:
                    # Preserve the existing per-metric failure matrix and continue
                    # with later metrics instead of failing the sequence early.
                    result = exc
                summary_results[metric_type] = result
                if is_photo_capture_attempt_conflict(result):
                    raise _structured_photo_error(
                        status_code=status.HTTP_409_CONFLICT,
                        stage=f"summary-save-{metric_type}",
                        completed_stages=completed_stages,
                        request_id=request_id,
                        photo_capture_attempt=work_item.photo_capture_attempt,
                        error_type="PhotoCaptureAttemptConflict",
                        diagnostic_code="PHOTO_CAPTURE_ATTEMPT_CONFLICT",
                        retryable=False,
                        message="백엔드가 촬영 회차 충돌을 반환해 이후 요약 저장을 중단했습니다.",
                        reports=_safe_backend_reports(core_results, summary_results),
                        partial_save=_partial_save_occurred(core_results, summary_results),
                    )

            for metric_type, result in summary_results.items():
                if isinstance(result, httpx.Response) and _is_2xx(result):
                    _append_completed(
                        completed_stages,
                        f"summary-save-{metric_type}",
                    )
                _log_photo_progress(
                    request_id=request_id,
                    measurement_session_id=measurementSessionId,
                    stage=f"summary-save-{metric_type}",
                    event="summary_save_result",
                    message=_backend_save_progress_message(
                        _SUMMARY_KOREAN_LABELS[metric_type],
                        result,
                    ),
                    details=_progress_attempt_details(result),
                )

            summary_transport_errors = [
                (metric_type, result)
                for metric_type, result in summary_results.items()
                if isinstance(result, BaseException)
            ]
            if summary_transport_errors:
                metric_type, transport_error = summary_transport_errors[0]
                stage = f"summary-save-{metric_type}"
                safe_reports = _safe_backend_reports(core_results, summary_results)
                safe_reports["summaries"].update(
                    {
                        skipped_type: {
                            "status": "skipped",
                            "reason": reason,
                        }
                        for skipped_type, reason in summary_skip_reasons.items()
                    }
                )
                raise _structured_photo_error(
                    photo_capture_attempt=work_item.photo_capture_attempt,
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    stage=stage,
                    completed_stages=completed_stages,
                    request_id=request_id,
                    error_type=type(transport_error).__name__,
                    retryable=_is_retryable_exception(transport_error),
                    message="One or more summary report transport requests failed.",
                    reports=safe_reports,
                    partial_save=_partial_save_occurred(
                        core_results,
                        summary_results,
                    ),
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise _structured_photo_error(
            photo_capture_attempt=work_item.photo_capture_attempt,
            status_code=status.HTTP_502_BAD_GATEWAY,
            stage=stage,
            completed_stages=completed_stages,
            request_id=request_id,
            error_type=type(exc).__name__,
            retryable=_is_retryable_exception(exc),
            message="The report backend connection could not be completed.",
            reports=_safe_backend_reports(core_results, summary_results),
            partial_save=_partial_save_occurred(core_results, summary_results),
        ) from exc

    core_responses = {
        report_name: result
        for report_name, result in core_results.items()
        if isinstance(result, httpx.Response)
    }
    summary_responses = {
        metric_type: result
        for metric_type, result in summary_results.items()
        if isinstance(result, httpx.Response)
    }
    reports: dict[str, object] = {
        "halluxValgus": _response_summary(hallux_response),
        "metrics": _response_summary(metrics_response),
        "tineaPedis": _response_summary(tinea_response),
        "plantarFootprint": _response_summary(plantar_response),
        "summaries": {
            metric_type: _response_summary(response)
            for metric_type, response in summary_responses.items()
        },
    }
    for metric_type, reason in summary_skip_reasons.items():
        reports["summaries"][metric_type] = {
            "status": "skipped",
            "reason": reason,
        }

    errors: list[dict[str, object]] = []
    for report_name, response in core_responses.items():
        if _is_2xx(response):
            continue
        error_stage = _CORE_SAVE_STAGES[report_name]
        errors.append(
            {
                "stage": error_stage,
                "errorType": "UpstreamHTTPError",
                "retryable": _is_retryable_http_status(response.status_code),
                "statusCode": int(response.status_code),
            }
        )
    for metric_type, response in summary_responses.items():
        if _is_2xx(response):
            continue
        errors.append(
            {
                "stage": f"summary-save-{metric_type}",
                "errorType": "UpstreamHTTPError",
                "retryable": _is_retryable_http_status(response.status_code),
                "statusCode": int(response.status_code),
            }
        )
    failed_stages = [str(item["stage"]) for item in errors]
    has_successful_save = _partial_save_occurred(core_results, summary_results)
    for error in errors:
        logger.warning(
            "[사진 분석][요청 %s] 일부 리포트 저장에 실패했습니다. | "
            "pipeline=%s request_id=%s stage=%s error_type=%s retryable=%s "
            "partial_save=%s status_code=%s",
            request_id[:8],
            PIPELINE_VERSION,
            request_id,
            error["stage"],
            error["errorType"],
            error["retryable"],
            has_successful_save,
            error["statusCode"],
        )

    core_ok = len(core_responses) == len(_CORE_SAVE_STAGES) and all(
        _is_2xx(item) for item in core_responses.values()
    )
    all_expected_summaries_saved = set(summary_responses) == {
        "HALLUX_VALGUS",
        "ATHLETES_FOOT",
        "SKIN_IRRITATION",
    }
    summaries_ok = all(_is_2xx(item) for item in summary_responses.values())
    all_reports_ok = core_ok and all_expected_summaries_saved and summaries_ok
    partial_save = (not all_reports_ok) and has_successful_save
    response_status_code = (
        status.HTTP_200_OK if all_reports_ok else status.HTTP_207_MULTI_STATUS
    )

    response_body = {
        "measurementSessionId": measurementSessionId,
        "pipeline": PIPELINE_VERSION,
        "requestId": request_id,
        "completedStages": completed_stages,
        "partialSave": partial_save,
        "failedStages": failed_stages,
        "errors": errors,
        "analysis": {
            **_local_analysis_payload(
                integrated,
                fungal_score=fungal_score,
                skin_score=skin_score,
                left_plantar_facts=left_plantar_facts,
                right_plantar_facts=right_plantar_facts,
            ),
            "reportTextGeneration": {
                "source": generated.generation_source,
                "model": settings.openai_report_model if generated.generation_source == "OPENAI" else None,
                **({
                    "inflammationAnalysisEnabled": False,
                    "inflammationAnalysisMode": "image_based_with_detector_disabled",
                    "localPolicyFields": list(getattr(generated, "local_policy_fields", ())),
                } if not getattr(generated, "inflammation_analysis_enabled", True) else {}),
                **({"failureCode": generation_failure_code} if generation_failure_code is not None else {}),
            },
        },
        "reports": reports,
        "skippedMetricTypes": list(SKIPPED_METRIC_TYPES),
    }
    successful_save_count = sum(
        isinstance(result, httpx.Response) and _is_2xx(result)
        for result in (*core_results.values(), *summary_results.values())
    )
    _log_photo_progress(
        request_id=request_id,
        measurement_session_id=measurementSessionId,
        stage="response",
        event="response_completed",
        message=(
            "사진 분석과 모든 리포트 저장을 정상적으로 완료했습니다."
            if all_reports_ok
            else "사진 분석은 완료했지만 일부 리포트 저장에 실패했습니다."
        ),
        details={
            "statusCode": response_status_code,
            "allReportsOk": all_reports_ok,
            "partialSave": partial_save,
            "completedStageCount": len(completed_stages),
            "failedStageCount": len(failed_stages),
            "attemptedSaveCount": len(core_results) + len(summary_results),
            "successfulSaveCount": successful_save_count,
            "elapsedSeconds": round(perf_counter() - pipeline_started_at, 3),
        },
    )
    return JSONResponse(
        content=jsonable_encoder(response_body),
        status_code=response_status_code,
        headers={
            "X-Feetfit-Pipeline": PIPELINE_VERSION,
            "X-Feetfit-Request-Id": request_id,
        },
    )


async def _run_photo_analysis_job(work_item: PhotoAnalysisWorkItem) -> None:
    """Run one accepted photo job and convert terminal failures to callbacks."""

    queue_started_at = perf_counter()
    try:
        async with _photo_analysis_worker_semaphore:
            _log_photo_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                stage="queue",
                event="job_started",
                message="대기열에서 사진 분석 작업을 가져와 백그라운드 분석을 시작합니다.",
                details={
                    "photoCaptureAttempt": work_item.photo_capture_attempt,
                    "queueWaitSeconds": round(perf_counter() - queue_started_at, 3),
                },
            )
            try:
                response = await _execute_photo_pipeline(work_item)
            except HTTPException as exc:
                if is_photo_capture_attempt_conflict(exc.detail, int(exc.status_code)):
                    _log_photo_progress(
                        request_id=work_item.request_id,
                        measurement_session_id=work_item.measurement_session_id,
                        stage="capture-attempt-conflict",
                        event="job_stopped",
                        message="촬영 회차 충돌로 작업을 중단했습니다. 세션 실패 상태는 변경하지 않습니다.",
                        details={"photoCaptureAttempt": work_item.photo_capture_attempt},
                    )
                    return
                (
                    stage,
                    error_type,
                    diagnostic_code,
                    image_field,
                    validation_field,
                    validation_rule,
                ) = _http_exception_failure_metadata(exc)
                notification_kwargs: dict[str, Any] = {
                    "failure_reason": _http_exception_failure_reason(exc),
                    "stage": stage,
                    "error_type": error_type,
                }
                if diagnostic_code is not None:
                    notification_kwargs.update(
                        diagnostic_code=diagnostic_code,
                        image_field=image_field,
                    )
                if diagnostic_code == "GPT_REPORT_RESPONSE_INVALID":
                    notification_kwargs.update(
                        validation_field=validation_field,
                        validation_rule=validation_rule,
                    )
                await _notify_photo_job_failure(work_item, **notification_kwargs)
                _log_photo_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage=stage,
                    event="job_failed",
                    message=_photo_job_failure_log_message(
                        stage=stage,
                        diagnostic_code=diagnostic_code,
                        image_field=image_field,
                    ),
                    details={
                        key: value
                        for key, value in {
                            "statusCode": int(exc.status_code),
                            "diagnosticCode": diagnostic_code,
                        }.items()
                        if value is not None
                    },
                )
                return
            except Exception as exc:
                error_type = type(exc).__name__
                logger.error(
                    "[사진 분석][요청 %s] 백그라운드 작업 중 예상하지 못한 오류가 "
                    "발생했습니다. | pipeline=%s request_id=%s stage=%s error_type=%s",
                    work_item.request_id[:8],
                    PIPELINE_VERSION,
                    work_item.request_id,
                    "background-worker",
                    error_type,
                )
                await _notify_photo_job_failure(
                    work_item,
                    failure_reason="AI_SERVER_ERROR",
                    stage="background-worker",
                    error_type=error_type,
                )
                _log_photo_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage="background-worker",
                    event="job_failed",
                    message="사진 분석 작업 중 예상하지 못한 오류가 발생했습니다. 백엔드에 측정 실패 상태를 전달했습니다.",
                    details={"statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR},
                )
                return

            try:
                response_payload = json.loads(response.body)
            except (ValueError, TypeError):
                response_payload = None
            if is_photo_capture_attempt_conflict(response_payload, int(response.status_code)):
                _log_photo_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage="capture-attempt-conflict",
                    event="job_stopped",
                    message="촬영 회차 충돌 응답으로 작업을 중단했습니다. FAILED를 전송하지 않습니다.",
                    details={"photoCaptureAttempt": work_item.photo_capture_attempt},
                )
                return
            if response.status_code == status.HTTP_200_OK:
                _log_photo_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage="queue",
                    event="job_succeeded",
                    message="백그라운드 사진 분석과 결과 저장을 모두 완료했습니다.",
                    details={"statusCode": status.HTTP_200_OK},
                )
                return
            if response.status_code == status.HTTP_207_MULTI_STATUS:
                stage, error_type, failure_reason = (
                    _partial_outcome_failure_metadata(response)
                )
            else:
                stage = "response"
                error_type = "UnexpectedPhotoAnalysisStatus"
                failure_reason = _photo_failure_reason_for_status(
                    int(response.status_code)
                )
            await _notify_photo_job_failure(
                work_item,
                failure_reason=failure_reason,
                stage=stage,
                error_type=error_type,
            )
            _log_photo_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                stage=stage,
                event="job_failed",
                message="사진 분석 결과 일부를 저장하지 못해 작업을 실패로 처리했습니다.",
                details={"statusCode": int(response.status_code)},
            )
    finally:
        await _release_photo_analysis_session(work_item.measurement_session_id)
