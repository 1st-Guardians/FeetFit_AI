from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.routing import APIRoute
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme
from app.schemas.pressure_heatmap import (
    PressureBalanceReportRequest,
    PressureBalanceSummaryRequest,
    PressureDistributionReportRequest,
    PressureHeatmapRequest,
    PressureMeasurementIdentity,
)
from app.services.pressure_heatmap_analysis import (
    PressureHeatmapError,
    create_pressure_heatmaps,
)
from app.services.measurement_failure_notifier import notify_measurement_failed
from app.services.pressure_measurement_status import (
    PressureStatusOutcome,
    notify_pressure_analysis_started, notify_pressure_retry_required,
    notify_pressure_measurement_received,
)
from app.services.pressure_attempt_registry import claim_pressure_attempt
from app.services.photo_capture_status import valid_status_authorization
from app.services.pressure_signal_quality import classify_weak_foot_signal
from app.services.report_text_generation import (
    build_pressure_balance_facts,
    generate_pressure_balance_report,
)


PIPELINE_VERSION = "pressure-heatmap-v3"
PRESSURE_ANALYSIS_MAX_PENDING_JOBS = 3
# Optional reference input for explicit PRESSURE_SENSOR_INPUT_MODE=fixed tests.
# The default hardware mode uses the sensor arrays from the POST request.
# LEFT_CORRECTED is already in the shared toe-first C0-C11 order.
# RIGHT_CORRECTED remains in raw hardware channel order. The renderer maps those
# channels to mirrored pads; pressure facts convert them to toe-first order.
LEFT_CORRECTED: tuple[float, ...] = (
    51.8,
    20.8,
    58.8,
    56.0,
    50.2,
    11099.0,
    629.3,
    45.2,
    43.8,
    587.7,
    8840.7,
    12134.9,
)
RIGHT_CORRECTED: tuple[float, ...] = (
    9939.8,
    7652.4,
    814.9,
    27.5,
    0.0,
    1684.3,
    8571.8,
    10.2,
    3379.6,
    4.9,
    4.4,
    0.0,
)
logger = logging.getLogger(__name__)
progress_logger = logging.getLogger("uvicorn.error")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_SENSITIVE_KEY_PARTS = ("authorization", "token", "secret", "password", "apikey")
_PROGRESS_DETAIL_KEYS = frozenset(
    {
        "report",
        "status",
        "statusCode",
        "retryable",
        "errorType",
        "generationSource",
        "errorCode",
        "leftTotal",
        "rightTotal",
        "leftPressurePercent",
        "rightPressurePercent",
        "leftActiveSensorCount",
        "rightActiveSensorCount",
        "leftForefootPercent",
        "leftMidfootPercent",
        "leftHeelPercent",
        "rightForefootPercent",
        "rightMidfootPercent",
        "rightHeelPercent",
        "bilateralBalanceScore",
        "leftDistributionScore",
        "rightDistributionScore",
        "balanceScore",
        "qualityStatus",
        "loadBalanceLevel",
        "leftDistributionLevel",
        "rightDistributionLevel",
        "leftImageBytes",
        "rightImageBytes",
        "coreReportsSaved",
        "downstreamReportsSaved",
        "completedStageCount",
        "failedStageCount",
        "partialSave",
        "allSaved",
        "pressureMeasurementAttempt",
    }
)
_PROGRESS_ENUM_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class PressureValidationRoute(APIRoute):
    """Route sensor-only schema failures through the recoverable retry callback."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                return await _pressure_validation_response(request, exc)

        return handler


router = APIRouter(route_class=PressureValidationRoute)


@dataclass(frozen=True, slots=True, repr=False)
class PressureAnalysisWorkItem:
    """Immutable, request-independent inputs for one pressure job."""

    request_id: str
    measurement_session_id: int
    output_dir: Path
    left_pressure_values: tuple[float, ...]
    right_pressure_values: tuple[float, ...]
    authorization: str
    pressure_measurement_attempt: int


# Pressure jobs deliberately have their own queue and worker semaphore.  A long
# pressure analysis therefore never occupies the photo or environment worker.
_pressure_analysis_worker_semaphore = asyncio.Semaphore(1)
_pressure_analysis_reservation_lock = threading.Lock()
_active_pressure_analysis_sessions: set[int] = set()


def _safe_generation_failure_code(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value) else None


def _progress_details(details: dict[str, object] | None) -> dict[str, object]:
    """Keep progress metadata scalar, bounded, and free of request content."""

    safe: dict[str, object] = {}
    for key, value in (details or {}).items():
        if key not in _PROGRESS_DETAIL_KEYS or value is None:
            continue
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
        elif isinstance(value, str) and _PROGRESS_ENUM_PATTERN.fullmatch(value):
            safe[key] = value
    return safe


def _log_progress(
    *,
    request_id: str,
    measurement_session_id: int,
    event: str,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    payload = {
        "pipeline": PIPELINE_VERSION,
        "requestId": request_id,
        "measurementSessionId": measurement_session_id,
        "event": event,
        "stage": stage,
        "message": message,
        "details": _progress_details(details),
    }
    progress_logger.info(
        "[압력 분석][세션 %s][요청 %s] %s | pipeline_progress %s",
        measurement_session_id,
        request_id[:8],
        message,
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


_PRESSURE_REPORT_LABELS = {
    "pressureHeatmap": "압력 히트맵",
    "pressureDistribution": "좌우 압력 분포",
    "balance": "자세 균형",
    "summary": "PRESSURE_BALANCE 지표 리포트",
}


def _pressure_report_label(report: str) -> str:
    return _PRESSURE_REPORT_LABELS.get(report, "압력 분석")


def _fact_progress_details(pressure_facts: dict[str, Any]) -> dict[str, object]:
    left_zones = pressure_facts.get("leftZonePercent")
    right_zones = pressure_facts.get("rightZonePercent")
    left_zones = left_zones if isinstance(left_zones, dict) else {}
    right_zones = right_zones if isinstance(right_zones, dict) else {}
    return {
        "leftTotal": pressure_facts.get("leftTotal"),
        "rightTotal": pressure_facts.get("rightTotal"),
        "leftPressurePercent": pressure_facts.get("leftLoadPercent"),
        "rightPressurePercent": pressure_facts.get("rightLoadPercent"),
        "leftActiveSensorCount": pressure_facts.get("leftActiveSensorCount"),
        "rightActiveSensorCount": pressure_facts.get("rightActiveSensorCount"),
        "leftForefootPercent": left_zones.get("forefoot"),
        "leftMidfootPercent": left_zones.get("midfoot"),
        "leftHeelPercent": left_zones.get("heel"),
        "rightForefootPercent": right_zones.get("forefoot"),
        "rightMidfootPercent": right_zones.get("midfoot"),
        "rightHeelPercent": right_zones.get("heel"),
        "bilateralBalanceScore": pressure_facts.get("bilateralBalanceScore"),
        "leftDistributionScore": pressure_facts.get("leftDistributionScore"),
        "rightDistributionScore": pressure_facts.get("rightDistributionScore"),
        "balanceScore": pressure_facts.get("balanceScore"),
        "qualityStatus": pressure_facts.get("qualityStatus"),
        "loadBalanceLevel": pressure_facts.get("loadBalanceLevel"),
        "leftDistributionLevel": pressure_facts.get("leftDistributionLevel"),
        "rightDistributionLevel": pressure_facts.get("rightDistributionLevel"),
    }


def _backend_progress_details(
    report: str,
    result: httpx.Response | BaseException,
) -> dict[str, object]:
    details: dict[str, object] = {"report": report}
    if isinstance(result, httpx.Response):
        details.update(
            {
                "status": "saved" if _is_2xx(result) else "http-error",
                "statusCode": int(result.status_code),
                "retryable": (
                    False
                    if _is_2xx(result)
                    else _is_retryable_http_status(int(result.status_code))
                ),
            }
        )
    else:
        details.update(
            {
                "status": "transport-error",
                "errorType": type(result).__name__,
                "retryable": _is_retryable_transport(result),
            }
        )
    return details


def _select_pressure_analysis_values(
    request: PressureHeatmapRequest,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Select the configured source while preserving both sensor order contracts."""

    if settings.pressure_sensor_input_mode == "hardware":
        return (
            tuple(float(value) for value in request.left_pressure_values),
            tuple(float(value) for value in request.right_pressure_values),
        )
    return LEFT_CORRECTED, RIGHT_CORRECTED


def _request_dir() -> tuple[str, Path]:
    request_id = uuid4().hex
    return request_id, Path(settings.pressure_heatmap_output_dir) / request_id


def _trace_headers(request_id: str) -> dict[str, str]:
    return {
        "X-Feetfit-Pipeline": PIPELINE_VERSION,
        "X-Feetfit-Request-Id": request_id,
    }


async def _confirm_pressure_measurement_received(work_item: PressureAnalysisWorkItem) -> None:
    try:
        outcome = await notify_pressure_measurement_received(
            measurement_session_id=work_item.measurement_session_id, authorization=work_item.authorization,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt, request_id=work_item.request_id,
        )
    except Exception:
        outcome = PressureStatusOutcome()
    if not outcome.confirmed or outcome.status != "MEASURING_PRESSURE" or outcome.attempt_conflict:
        raise HTTPException(
            status.HTTP_409_CONFLICT if outcome.attempt_conflict else status.HTTP_502_BAD_GATEWAY,
            detail={"pipeline": PIPELINE_VERSION, "stage": "pressure-status-measuring",
                    "measurementSessionId": work_item.measurement_session_id,
                    "pressureMeasurementAttempt": work_item.pressure_measurement_attempt,
                    "requestId": work_item.request_id,
                    "errorType": "PressureMeasurementAttemptConflict" if outcome.attempt_conflict else "UnconfirmedPressureStatus",
                    "message": "Could not confirm the attempt-bound MEASURING_PRESSURE state.",
                    "retryable": False, "partialSave": False, "reports": {}},
            headers=_trace_headers(work_item.request_id),
        )
    _log_progress(
        request_id=work_item.request_id, measurement_session_id=work_item.measurement_session_id,
        event="completed", stage="pressure-status-measuring",
        message="받은 압력 회차로 MEASURING_PRESSURE 상태를 확인했습니다.",
        details={"pressureMeasurementAttempt": work_item.pressure_measurement_attempt},
    )


async def _stop_for_pressure_retry(
    *, measurement_session_id: int, authorization: str,
    pressure_measurement_attempt: int, request_id: str, error_type: str,
    quality_status: str | None = None,
    presence_facts: dict[str, float | int | str] | None = None,
) -> NoReturn:
    try:
        outcome = await notify_pressure_retry_required(
            measurement_session_id=measurement_session_id, authorization=authorization,
            pressure_measurement_attempt=pressure_measurement_attempt,
            request_id=request_id, error_type=error_type,
        )
    except Exception:
        outcome = PressureStatusOutcome()
    detail = {
        "pipeline": PIPELINE_VERSION, "stage": "pressure-balance-analysis",
        "measurementSessionId": measurement_session_id,
        "pressureMeasurementAttempt": pressure_measurement_attempt,
        "requestId": request_id, "errorType": error_type,
        "failureReason": "INVALID_PRESSURE_DATA", "partialSave": False,
        "reports": {}, "mayStartPressure": False,
        "statusCallbackConfirmed": outcome.confirmed,
        "pressureRetryRequired": outcome.confirmed and outcome.status == "WAITING_FOR_PRESSURE_RETRY",
        "retryable": False,
    }
    if quality_status is not None:
        detail["qualityStatus"] = quality_status
    if presence_facts is not None:
        detail.update(presence_facts)
    if outcome.confirmed:
        detail["measurementStatus"] = outcome.status
    if outcome.attempt_conflict:
        detail.update(message="The pressure measurement attempt is stale.",
                      errorType="PressureMeasurementAttemptConflict")
        response_status = status.HTTP_409_CONFLICT
    elif not outcome.confirmed:
        detail["message"] = "Could not confirm the pressure retry state with the backend."
        response_status = status.HTTP_502_BAD_GATEWAY
    else:
        detail["message"] = (
            "Pressure data is invalid; use the backend pressure remeasurement flow."
            if outcome.status == "WAITING_FOR_PRESSURE_RETRY"
            else "The backend exhausted the pressure remeasurement allowance."
        )
        response_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    raise HTTPException(response_status, detail=detail, headers=_trace_headers(request_id))


async def _pressure_validation_response(request: Request, exc: RequestValidationError):
    """Only invalid sensor arrays with trustworthy identity can request a retry."""
    body = exc.body
    errors = exc.errors()
    authorization = request.headers.get("Authorization", "")
    sensor_fields = {"leftPressureValues", "rightPressureValues", "left_pressure_values", "right_pressure_values"}
    if (not isinstance(body, dict) or not errors
            or any(len(error.get("loc", ())) < 2
                   or error["loc"][0] != "body" or error["loc"][1] not in sensor_fields
                   for error in errors)
            or not valid_status_authorization(authorization)
            or not re.fullmatch(r"(?i)Bearer\s+\S+", authorization)):
        return await request_validation_exception_handler(request, exc)
    try:
        identity = PressureMeasurementIdentity.model_validate({
            key: body[key] for key in ("measurementSessionId", "measurement_session_id",
                                      "pressureMeasurementAttempt", "pressure_measurement_attempt")
            if key in body
        })
    except ValidationError:
        return await request_validation_exception_handler(request, exc)
    request_id, _ = _request_dir()
    session_id = identity.measurement_session_id
    reserved = False
    try:
        await _reserve_pressure_analysis_session(
            measurement_session_id=session_id, request_id=request_id,
            pressure_measurement_attempt=identity.pressure_measurement_attempt,
        )
        reserved = True
        await _stop_for_pressure_retry(
            measurement_session_id=session_id, authorization=authorization,
            pressure_measurement_attempt=identity.pressure_measurement_attempt,
            request_id=request_id, error_type="InvalidPressureData",
        )
    except HTTPException as error:
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail}, headers=error.headers)
    finally:
        if reserved:
            await _release_pressure_analysis_session(session_id)


_PRESSURE_RESPONSE_HEADERS = {
    "X-Feetfit-Pipeline": {
        "description": "Pressure-analysis pipeline version.",
        "schema": {"type": "string"},
    },
    "X-Feetfit-Request-Id": {
        "description": "Safe per-request correlation identifier.",
        "schema": {"type": "string"},
    },
}


def _pressure_queue_error(
    *,
    status_code: int,
    request_id: str,
    measurement_session_id: int,
    pressure_measurement_attempt: int,
    error_type: str,
    retryable: bool,
    message: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "pipeline": PIPELINE_VERSION,
            "stage": "request-reservation",
            "message": message,
            "measurementSessionId": measurement_session_id,
            "pressureMeasurementAttempt": pressure_measurement_attempt,
            "requestId": request_id,
            "errorType": error_type,
            "retryable": retryable,
            "partialSave": False,
            "reports": {},
        },
        headers=_trace_headers(request_id),
    )


def _pressure_attempt_registry_path() -> Path:
    return Path(settings.pressure_heatmap_output_dir) / "_state" / "pressure_attempts.sqlite3"


async def _reserve_pressure_analysis_session(
    *,
    measurement_session_id: int,
    request_id: str,
    pressure_measurement_attempt: int,
) -> None:
    with _pressure_analysis_reservation_lock:
        if measurement_session_id in _active_pressure_analysis_sessions:
            raise _pressure_queue_error(
                status_code=status.HTTP_409_CONFLICT,
                request_id=request_id,
                measurement_session_id=measurement_session_id,
                pressure_measurement_attempt=pressure_measurement_attempt,
                error_type="DuplicatePressureAnalysisError",
                retryable=False,
                message=(
                    "Pressure analysis is already active for this measurement "
                    "session."
                ),
            )
        if (
            len(_active_pressure_analysis_sessions)
            >= PRESSURE_ANALYSIS_MAX_PENDING_JOBS
        ):
            error = _pressure_queue_error(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                request_id=request_id,
                measurement_session_id=measurement_session_id,
                pressure_measurement_attempt=pressure_measurement_attempt,
                error_type="PressureAnalysisQueueFullError",
                retryable=True,
                message="The pressure analysis queue is full. Retry later.",
            )
            assert error.headers is not None
            error.headers["Retry-After"] = "5"
            raise error
        _active_pressure_analysis_sessions.add(measurement_session_id)
    try:
        claimed = await run_in_threadpool(
            claim_pressure_attempt, registry_path=_pressure_attempt_registry_path(),
            measurement_session_id=measurement_session_id,
            pressure_measurement_attempt=pressure_measurement_attempt,
        )
        if not claimed:
            raise _pressure_queue_error(
                status_code=status.HTTP_409_CONFLICT, request_id=request_id,
                measurement_session_id=measurement_session_id,
                pressure_measurement_attempt=pressure_measurement_attempt,
                error_type="PressureMeasurementAttemptAlreadyConsumed", retryable=False,
                message="This pressure measurement attempt was already received.",
            )
    except HTTPException:
        await _release_pressure_analysis_session(measurement_session_id)
        raise
    except Exception as exc:
        await _release_pressure_analysis_session(measurement_session_id)
        logger.warning("Pressure attempt registry unavailable. measurement_session_id=%s error_type=%s",
                       measurement_session_id, type(exc).__name__)
        raise _pressure_queue_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, request_id=request_id,
            measurement_session_id=measurement_session_id,
            pressure_measurement_attempt=pressure_measurement_attempt,
            error_type="PressureAttemptRegistryUnavailable", retryable=False,
            message="Could not safely register the pressure measurement attempt.",
        ) from exc
    except BaseException:
        await _release_pressure_analysis_session(measurement_session_id)
        raise


async def _release_pressure_analysis_session(
    measurement_session_id: int,
) -> None:
    with _pressure_analysis_reservation_lock:
        _active_pressure_analysis_sessions.discard(measurement_session_id)


async def _notify_pressure_job_failure(
    work_item: PressureAnalysisWorkItem,
    *,
    failure_reason: str,
    stage: str,
    error_type: str,
) -> None:
    try:
        await notify_measurement_failed(
            measurement_session_id=work_item.measurement_session_id,
            authorization=work_item.authorization,
            failure_reason=failure_reason,
            pipeline=PIPELINE_VERSION,
            stage=stage,
            error_type=error_type,
            request_id=work_item.request_id,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt,
        )
    except Exception as callback_error:
        # A best-effort callback must never crash the background worker or
        # replace the original pressure failure.
        logger.warning(
            "Pressure failure callback could not be delivered. "
            "measurement_session_id=%s callback_error_type=%s",
            work_item.measurement_session_id,
            type(callback_error).__name__,
        )


def _is_2xx(response: httpx.Response) -> bool:
    return 200 <= int(response.status_code) < 300


def _response_summary(response: httpx.Response) -> dict[str, object]:
    try:
        body: object = response.json() if response.content else None
    except (json.JSONDecodeError, ValueError):
        body = response.text[:2000]
    return {
        "status": "saved" if _is_2xx(response) else "http-error",
        "statusCode": int(response.status_code),
        "contentType": response.headers.get("content-type") or None,
        "body": _redact_response_value(body),
        "retryable": _is_retryable_http_status(int(response.status_code)),
    }


def _redact_response_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            normalized = str(key).lower().replace("-", "").replace("_", "")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact_response_value(item, depth=depth + 1)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_response_value(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, str):
        return _BEARER_PATTERN.sub("Bearer [REDACTED]", value)[:2000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _BEARER_PATTERN.sub("Bearer [REDACTED]", str(value))[:500]


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code < 600


def _is_retryable_transport(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _result_summary(result: httpx.Response | BaseException) -> dict[str, object]:
    if isinstance(result, httpx.Response):
        return _response_summary(result)
    return {
        "status": "transport-error",
        "errorType": type(result).__name__,
        "retryable": _is_retryable_transport(result),
    }


def _skip_summary(reason: str) -> dict[str, object]:
    return {"status": "skipped", "reason": reason}


def _raw_load_percentages(
    left_values: list[float],
    right_values: list[float],
) -> tuple[float, float] | None:
    """Compute backend percentages directly from the validated raw totals."""

    left_total = math.fsum(left_values)
    right_total = math.fsum(right_values)
    pair_total = left_total + right_total
    if pair_total <= 0.0:
        return None
    left_percent = round(left_total / pair_total * 100.0, 1)
    # Derivation from the rounded left side makes the persisted pair sum to 100.
    right_percent = round(100.0 - left_percent, 1)
    return left_percent, right_percent


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="좌우 압력 히트맵·자세 균형 분석 및 저장",
    description=(
        "Validates exactly 12 measured pressure values for each foot and "
        "returns 202 Accepted after registering an independent background job. "
        "The job renders both heatmaps locally and computes the authoritative "
        "score deterministically as 60% "
        "bilateral load balance plus 20% for each foot's non-clinical, sensor-count-"
        "adjusted three-zone spread index. It performs four backend saves: raw "
        "left C0-C11 and right heel-first raw values with heatmaps, load "
        "percentages with pressure images, the "
        "balance score/comment, and the PRESSURE_BALANCE summary. GPT generates "
        "only non-diagnostic left/right regional observations, an overall balance "
        "observation, and a practical tip; it never changes a score. The "
        "regional analysis covers forefoot (C0-C4), midfoot (C5-C8), and heel "
        "(C9-C11). Outside-outline C0/C1 pads are clipped unless each reading is "
        "at least 2% of that foot's peak and 0.5% of its total; valid pads extend "
        "the transparent foot mask through a connected local bridge. Downstream "
        "balance and summary saves run only after both core "
        "image reports return 2xx and the paired-foot signal is conclusive."
    ),
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Measured pressure input accepted and queued for background analysis.",
            "headers": _PRESSURE_RESPONSE_HEADERS,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": [
                            "accepted",
                            "status",
                            "measurementSessionId",
                            "pressureMeasurementAttempt",
                            "pipeline",
                            "requestId",
                        ],
                        "properties": {
                            "accepted": {"type": "boolean"},
                            "status": {"type": "string", "enum": ["QUEUED"]},
                            "measurementSessionId": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "pressureMeasurementAttempt": {"type": "integer", "minimum": 0},
                            "pipeline": {"type": "string"},
                            "requestId": {"type": "string"},
                        },
                    }
                }
            },
        },
        status.HTTP_409_CONFLICT: {
            "description": "Pressure analysis is already active or the pressure attempt is stale.",
            "headers": _PRESSURE_RESPONSE_HEADERS,
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": (
                "The JSON body or one of the 12-value sensor arrays is invalid. "
                "Schema errors occur before a background job is registered."
            ),
            "headers": _PRESSURE_RESPONSE_HEADERS,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The validated pressure job could not be scheduled.",
            "headers": _PRESSURE_RESPONSE_HEADERS,
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "The attempt-bound MEASURING_PRESSURE status could not be confirmed.",
            "headers": _PRESSURE_RESPONSE_HEADERS,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The pressure analysis queue is full.",
            "headers": {
                **_PRESSURE_RESPONSE_HEADERS,
                "Retry-After": {
                    "description": "Minimum retry delay in seconds.",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
async def create_pressure_heatmap_report(
    background_tasks: BackgroundTasks,
    request: PressureHeatmapRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> JSONResponse:
    request_id, output_dir = _request_dir()
    authorization = f"{credentials.scheme} {credentials.credentials}"
    left_pressure_values, right_pressure_values = _select_pressure_analysis_values(
        request
    )
    work_item = PressureAnalysisWorkItem(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        output_dir=output_dir,
        left_pressure_values=left_pressure_values,
        right_pressure_values=right_pressure_values,
        authorization=authorization,
        pressure_measurement_attempt=request.pressure_measurement_attempt,
    )
    await _reserve_pressure_analysis_session(
        measurement_session_id=request.measurement_session_id,
        request_id=request_id,
        pressure_measurement_attempt=work_item.pressure_measurement_attempt,
    )
    try:
        await _confirm_pressure_measurement_received(work_item)
    except BaseException:
        await _release_pressure_analysis_session(work_item.measurement_session_id)
        raise
    try:
        background_tasks.add_task(_run_pressure_analysis_job, work_item)
    except Exception as exc:
        await _release_pressure_analysis_session(request.measurement_session_id)
        await _notify_pressure_job_failure(
            work_item,
            failure_reason="AI_SERVER_ERROR",
            stage="background-schedule",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": "background-schedule",
                "message": "Could not schedule the pressure analysis job.",
                "measurementSessionId": request.measurement_session_id,
                "requestId": request_id,
                "errorType": type(exc).__name__,
                "retryable": False,
                "partialSave": False,
                "reports": {},
            },
            headers=_trace_headers(request_id),
        ) from exc

    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="queued",
        stage="queue",
        message="압력 분석 요청을 접수해 백그라운드 대기열에 등록했습니다.",
        details={"pressureMeasurementAttempt": work_item.pressure_measurement_attempt},
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "accepted": True,
            "status": "QUEUED",
            "measurementSessionId": request.measurement_session_id,
            "pressureMeasurementAttempt": request.pressure_measurement_attempt,
            "pipeline": PIPELINE_VERSION,
            "requestId": request_id,
        },
        headers=_trace_headers(request_id),
    )


async def _execute_pressure_pipeline(
    work_item: PressureAnalysisWorkItem,
) -> JSONResponse:
    request = PressureHeatmapRequest(
        measurementSessionId=work_item.measurement_session_id,
        pressureMeasurementAttempt=work_item.pressure_measurement_attempt,
        leftPressureValues=list(work_item.left_pressure_values),
        rightPressureValues=list(work_item.right_pressure_values),
    )
    request_id = work_item.request_id
    output_dir = work_item.output_dir
    completed_stages: list[str] = []
    authorization = work_item.authorization
    failure_notified = False

    async def notify_terminal_failure(
        *,
        failure_reason: str,
        failure_stage: str,
        error_type: str,
    ) -> None:
        """Send at most one best-effort terminal callback for this request."""

        nonlocal failure_notified
        if failure_notified:
            return
        # Claim the callback before awaiting it so even an unexpected notifier
        # exception cannot cause a second terminal notification later.
        failure_notified = True
        try:
            await notify_measurement_failed(
                measurement_session_id=request.measurement_session_id,
                authorization=authorization,
                failure_reason=failure_reason,
                pipeline=PIPELINE_VERSION,
                stage=failure_stage,
                error_type=error_type,
                request_id=request_id,
                pressure_measurement_attempt=work_item.pressure_measurement_attempt,
            )
        except Exception as callback_error:
            # The notifier is itself best effort, but retain this boundary so a
            # test double or future implementation cannot mask the root result.
            logger.warning(
                "압력 분석 실패 상태를 백엔드에 전달하지 못했습니다. "
                "measurement_session_id=%s callback_error_type=%s",
                request.measurement_session_id,
                type(callback_error).__name__,
            )

    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="started",
        stage="pressure-request",
        message="좌우 압력 센서값을 정상적으로 받았습니다. 압력 분석을 시작합니다.",
        details={"pressureMeasurementAttempt": work_item.pressure_measurement_attempt},
    )

    stage = "pressure-balance-analysis"
    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="started",
        stage=stage,
        message="좌우 하중과 각 발의 앞꿈치·중간·뒤꿈치 압력 분포를 분석하는 중입니다.",
    )
    try:
        pressure_facts = build_pressure_balance_facts(
            request.left_pressure_values,
            request.right_pressure_values,
        )
    except ValueError as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            event="failed",
            stage=stage,
            message="압력 센서값 검증에 실패했습니다.",
            details={"errorType": type(exc).__name__, "retryable": False},
        )
        await _stop_for_pressure_retry(
            measurement_session_id=request.measurement_session_id,
            authorization=authorization,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt,
            request_id=request_id, error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": stage,
                "message": "The pressure sensor frame is invalid.",
                "requestId": request_id,
                "retryable": False,
                "partialSave": False,
                "reports": {},
            },
            headers=_trace_headers(request_id),
        ) from exc
    except Exception as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            event="failed",
            stage=stage,
            message="압력 분석 내부 오류로 센서값 계산을 완료하지 못했습니다.",
            details={"errorType": type(exc).__name__, "retryable": True},
        )
        await notify_terminal_failure(
            failure_reason="AI_SERVER_ERROR",
            failure_stage=stage,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": stage,
                "message": "Pressure fact generation failed.",
                "requestId": request_id,
                "errorType": type(exc).__name__,
                "retryable": True,
                "partialSave": False,
                "reports": {},
            },
            headers=_trace_headers(request_id),
        ) from exc
    if pressure_facts.get("qualityStatus") != "valid":
        quality_status = str(pressure_facts.get("qualityStatus") or "inconclusive")
        await _stop_for_pressure_retry(
            measurement_session_id=request.measurement_session_id, authorization=authorization,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt, request_id=request_id,
            error_type={"no-load": "NoLoadPressureSignal", "single-side-no-load": "SingleSidePressureSignal"}.get(
                quality_status, "InconclusivePressureSignal"),
            quality_status=quality_status,
        )
    weak_signal = classify_weak_foot_signal(
        request.left_pressure_values, request.right_pressure_values,
        minimum_foot_load_percent=settings.pressure_min_foot_load_percent,
        sensor_peak_percent=settings.pressure_presence_sensor_peak_percent,
    )
    if weak_signal is not None:
        await _stop_for_pressure_retry(
            measurement_session_id=request.measurement_session_id, authorization=authorization,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt, request_id=request_id,
            error_type="SingleSideWeakPressureSignal", presence_facts=weak_signal,
        )
    try:
        outcome = await notify_pressure_analysis_started(
            measurement_session_id=request.measurement_session_id, authorization=authorization,
            pressure_measurement_attempt=work_item.pressure_measurement_attempt, request_id=request_id,
        )
    except Exception:
        outcome = PressureStatusOutcome()
    if not outcome.confirmed or outcome.status != "ANALYZING" or outcome.attempt_conflict:
        raise HTTPException(
            status.HTTP_409_CONFLICT if outcome.attempt_conflict else status.HTTP_502_BAD_GATEWAY,
            detail={"pipeline": PIPELINE_VERSION, "stage": "pressure-status-transition",
                    "requestId": request_id, "pressureMeasurementAttempt": work_item.pressure_measurement_attempt,
                    "errorType": "PressureMeasurementAttemptConflict" if outcome.attempt_conflict else "UnconfirmedPressureStatus",
                    "message": "Could not confirm the attempt-bound ANALYZING state.",
                    "retryable": False, "partialSave": False, "reports": {}},
            headers=_trace_headers(request_id),
        )
    completed_stages.append(stage)
    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="completed",
        stage=stage,
        message="좌우 하중과 각 발의 압력 분포 분석을 완료했습니다.",
        details=_fact_progress_details(pressure_facts),
    )

    raw_percentages = _raw_load_percentages(
        request.left_pressure_values,
        request.right_pressure_values,
    )
    if raw_percentages is not None:
        left_pressure_percent, right_pressure_percent = raw_percentages
        try:
            fact_percentages = (
                float(pressure_facts["leftLoadPercent"]),
                float(pressure_facts["rightLoadPercent"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            _log_progress(
                request_id=request_id,
                measurement_session_id=request.measurement_session_id,
                event="failed",
                stage="pressure-fact-validation",
                message="계산된 압력 분석값의 형식을 검증하지 못했습니다.",
                details={"errorType": type(exc).__name__, "retryable": False},
            )
            await notify_terminal_failure(
                failure_reason="AI_SERVER_ERROR",
                failure_stage="pressure-fact-validation",
                error_type=type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "pipeline": PIPELINE_VERSION,
                    "stage": "pressure-fact-validation",
                    "message": "Server pressure facts are invalid.",
                    "requestId": request_id,
                    "errorType": type(exc).__name__,
                    "retryable": False,
                    "partialSave": False,
                    "reports": {},
                },
                headers=_trace_headers(request_id),
            ) from exc
        if fact_percentages != raw_percentages:
            _log_progress(
                request_id=request_id,
                measurement_session_id=request.measurement_session_id,
                event="failed",
                stage="pressure-fact-validation",
                message="계산된 좌우 압력 비율이 일치하지 않아 검증에 실패했습니다.",
                details={"retryable": False},
            )
            await notify_terminal_failure(
                failure_reason="AI_SERVER_ERROR",
                failure_stage="pressure-fact-validation",
                error_type="PressureFactConsistencyError",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "pipeline": PIPELINE_VERSION,
                    "stage": "pressure-fact-validation",
                    "message": "Server pressure percentages are inconsistent.",
                    "requestId": request_id,
                    "retryable": False,
                    "partialSave": False,
                    "reports": {},
                },
                headers=_trace_headers(request_id),
            )
    else:
        left_pressure_percent = None
        right_pressure_percent = None

    balance_conclusive = pressure_facts.get("qualityStatus") == "valid"
    balance_score = pressure_facts.get("balanceScore") if balance_conclusive else None
    balance_skip_reason = (
        None
        if balance_conclusive
        else "pressure balance is inconclusive because paired-foot load is insufficient"
    )

    stage = "pressure-heatmap-render"
    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="started",
        stage=stage,
        message="왼발과 오른발 압력 히트맵을 생성하는 중입니다.",
    )
    try:
        generated = await run_in_threadpool(
            create_pressure_heatmaps,
            request.left_pressure_values,
            request.right_pressure_values,
            output_dir,
        )
    except PressureHeatmapError as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            event="failed",
            stage=stage,
            message="압력 히트맵 생성에 실패했습니다.",
            details={"errorType": type(exc).__name__, "retryable": True},
        )
        failure = {
            "pipeline": PIPELINE_VERSION,
            "stage": stage,
            "message": "Validated pressure values could not be rendered.",
            "requestId": request_id,
            "errorType": type(exc).__name__,
            "retryable": True,
            "partialSave": False,
            "reports": {},
        }
        try:
            # This artifact is diagnostic-only. A full or read-only filesystem
            # must not replace the structured render error or suppress its
            # best-effort FAILED callback.
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as diagnostics_error:
            logger.error(
                "압력 히트맵 실패 진단 파일을 저장하지 못했습니다. "
                "request_id=%s error_type=%s",
                request_id,
                type(diagnostics_error).__name__,
            )
        logger.error(
            "압력 분석 중 오류가 발생했습니다. request_id=%s stage=%s error_type=%s",
            request_id,
            stage,
            type(exc).__name__,
        )
        await notify_terminal_failure(
            failure_reason="AI_SERVER_ERROR",
            failure_stage=stage,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=failure,
            headers=_trace_headers(request_id),
        ) from exc
    completed_stages.append(stage)
    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="completed",
        stage=stage,
        message="왼발과 오른발 압력 히트맵 생성을 완료했습니다.",
        details={
            "leftImageBytes": len(generated.left_png),
            "rightImageBytes": len(generated.right_png),
        },
    )

    headers = {
        "accept": "*/*",
        "Authorization": authorization,
    }
    raw_heatmap_request = json.dumps(
        request.model_dump(by_alias=True, mode="json", exclude={"pressure_measurement_attempt"}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    distribution_request_json: str | None = None
    if raw_percentages is not None:
        distribution_request = PressureDistributionReportRequest(
            measurement_session_id=request.measurement_session_id,
            left_pressure_percent=left_pressure_percent,
            right_pressure_percent=right_pressure_percent,
        )
        distribution_request_json = json.dumps(
            distribution_request.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    core_stage_by_name = {
        "pressureHeatmap": "pressure-heatmap-backend-forward",
        "pressureDistribution": "pressure-distribution-backend-forward",
    }
    downstream_stage_by_name = {
        "balance": "pressure-balance-backend-forward",
        "summary": "pressure-summary-backend-forward",
    }
    core_results: dict[str, httpx.Response | BaseException] = {}
    downstream_results: dict[str, httpx.Response | BaseException] = {}
    generated_report = None
    stage = "pressure-core-backend-forward"

    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            core_operations = {
                "pressureHeatmap": client.post(
                    settings.pressure_heatmap_report_endpoint,
                    headers=headers,
                    files={
                        "request": (None, raw_heatmap_request, "application/json"),
                        "leftPressureHeatmapImage": (
                            generated.left_path.name,
                            generated.left_png,
                            "image/png",
                        ),
                        "rightPressureHeatmapImage": (
                            generated.right_path.name,
                            generated.right_png,
                            "image/png",
                        ),
                    },
                )
            }
            if distribution_request_json is not None:
                core_operations["pressureDistribution"] = client.post(
                    settings.pressure_distribution_report_endpoint,
                    headers=headers,
                    files={
                        "request": (
                            None,
                            distribution_request_json,
                            "application/json",
                        ),
                        "leftPressureImage": (
                            generated.left_path.name,
                            generated.left_png,
                            "image/png",
                        ),
                        "rightPressureImage": (
                            generated.right_path.name,
                            generated.right_png,
                            "image/png",
                        ),
                    },
                )
            else:
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="skipped",
                    stage=core_stage_by_name["pressureDistribution"],
                    message="양발의 유효 압력이 없어 좌우 압력 분포 저장을 건너뜁니다.",
                    details={
                        "report": "pressureDistribution",
                        "status": "skipped",
                    },
                )

            for name in core_operations:
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="started",
                    stage=core_stage_by_name[name],
                    message=(
                        f"{_pressure_report_label(name)} 결과를 "
                        "백엔드에 저장하는 중입니다."
                    ),
                    details={"report": name},
                )

            core_values = await asyncio.gather(
                *core_operations.values(),
                return_exceptions=True,
            )
            core_results = dict(zip(core_operations, core_values, strict=True))
            for name, result in core_results.items():
                result_ok = isinstance(result, httpx.Response) and _is_2xx(result)
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="completed" if result_ok else "failed",
                    stage=core_stage_by_name[name],
                    message=(
                        f"{_pressure_report_label(name)} 결과를 "
                        "백엔드에 저장했습니다."
                        if result_ok
                        else (
                            f"{_pressure_report_label(name)} 결과의 "
                            "백엔드 저장에 실패했습니다."
                        )
                    ),
                    details=_backend_progress_details(name, result),
                )
                if isinstance(result, httpx.Response) and _is_2xx(result):
                    completed_stages.append(core_stage_by_name[name])

            core_transport_failures = [
                (name, result)
                for name, result in core_results.items()
                if isinstance(result, BaseException)
            ]
            if core_transport_failures:
                name, transport_error = core_transport_failures[0]
                reports = {
                    "pressureHeatmap": _result_summary(
                        core_results.get(
                            "pressureHeatmap",
                            RuntimeError("not attempted"),
                        )
                    ),
                    "pressureDistribution": (
                        _result_summary(core_results["pressureDistribution"])
                        if "pressureDistribution" in core_results
                        else _skip_summary("combined load is zero")
                    ),
                    "balance": _skip_summary("required pressure core transport failed"),
                    "summary": _skip_summary("required pressure core transport failed"),
                }
                partial_save = any(
                    isinstance(result, httpx.Response) and _is_2xx(result)
                    for result in core_results.values()
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "pipeline": PIPELINE_VERSION,
                        "stage": core_stage_by_name[name],
                        "message": "A pressure core report transport request failed.",
                        "requestId": request_id,
                        "errorType": type(transport_error).__name__,
                        "retryable": _is_retryable_transport(transport_error),
                        "partialSave": partial_save,
                        "completedStages": completed_stages,
                        "reports": reports,
                    },
                    headers=_trace_headers(request_id),
                )

            both_core_saved = (
                set(core_results) == {"pressureHeatmap", "pressureDistribution"}
                and all(
                    isinstance(result, httpx.Response) and _is_2xx(result)
                    for result in core_results.values()
                )
            )

            if both_core_saved and balance_conclusive:
                stage = "pressure-comment-generation"
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="started",
                    stage=stage,
                    message="분석 수치를 바탕으로 좌우 및 각 발의 압력 분포 코멘트를 생성하는 중입니다.",
                )
                try:
                    generated_report = await generate_pressure_balance_report(
                        left_pressure_values=request.left_pressure_values,
                        right_pressure_values=request.right_pressure_values,
                        pressure_facts=pressure_facts,
                        left_heatmap_png=generated.left_png,
                        right_heatmap_png=generated.right_png,
                    )
                    if generated_report.generation_source != "OPENAI" and not (
                        settings.openai_report_fallback_enabled
                        and generated_report.generation_source == "DETERMINISTIC_FALLBACK"
                    ):
                        raise RuntimeError("Required pressure report text source was not accepted.")
                except Exception as exc:
                    _log_progress(
                        request_id=request_id,
                        measurement_session_id=request.measurement_session_id,
                        event="failed",
                        stage=stage,
                        message="GPT 압력 분석 코멘트 생성에 실패했습니다.",
                        details={
                            "errorType": type(exc).__name__,
                            "retryable": True,
                        },
                    )
                    reports = {
                        "pressureHeatmap": _result_summary(
                            core_results["pressureHeatmap"]
                        ),
                        "pressureDistribution": _result_summary(
                            core_results["pressureDistribution"]
                        ),
                        "balance": _skip_summary("pressure comment generation failed"),
                        "summary": _skip_summary("pressure comment generation failed"),
                    }
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={
                            "pipeline": PIPELINE_VERSION,
                            "stage": stage,
                            "message": "Pressure comment generation failed.",
                            "requestId": request_id,
                            "errorType": type(exc).__name__,
                            "retryable": True,
                            "partialSave": True,
                            "completedStages": completed_stages,
                            "reports": reports,
                        },
                        headers=_trace_headers(request_id),
                    ) from exc
                completed_stages.append(stage)
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="completed",
                    stage=stage,
                    message=(
                        "GPT 압력 분석 코멘트 생성을 완료했습니다."
                        if generated_report.generation_source == "OPENAI"
                        else "서버 압력 분석 사실을 바탕으로 기본 코멘트 생성을 완료했습니다."
                    ),
                    details={
                        "generationSource": generated_report.generation_source,
                        "errorCode": _safe_generation_failure_code(
                            getattr(generated_report, "generation_failure_code", None)
                        ),
                    },
                )

                balance_request = PressureBalanceReportRequest(
                    measurement_session_id=request.measurement_session_id,
                    balance_score=balance_score,
                    balance_comment=generated_report.balance_comment,
                )
                summary_request = PressureBalanceSummaryRequest(
                    measurement_session_id=request.measurement_session_id,
                    metric_type="PRESSURE_BALANCE",
                    score=balance_score,
                    advice=tuple(generated_report.advice),
                )
                downstream_operations = {
                    "balance": client.post(
                        settings.pressure_balance_report_endpoint,
                        headers=headers,
                        json=balance_request.model_dump(by_alias=True, mode="json"),
                    ),
                    "summary": client.post(
                        settings.summary_report_endpoint,
                        headers=headers,
                        json=summary_request.model_dump(by_alias=True, mode="json"),
                    ),
                }
                for name in downstream_operations:
                    _log_progress(
                        request_id=request_id,
                        measurement_session_id=request.measurement_session_id,
                        event="started",
                        stage=downstream_stage_by_name[name],
                        message=(
                            f"{_pressure_report_label(name)} 결과를 "
                            "백엔드에 저장하는 중입니다."
                        ),
                        details={"report": name},
                    )
                downstream_values = await asyncio.gather(
                    *downstream_operations.values(),
                    return_exceptions=True,
                )
                downstream_results = dict(
                    zip(downstream_operations, downstream_values, strict=True)
                )
                for name, result in downstream_results.items():
                    result_ok = isinstance(result, httpx.Response) and _is_2xx(result)
                    _log_progress(
                        request_id=request_id,
                        measurement_session_id=request.measurement_session_id,
                        event="completed" if result_ok else "failed",
                        stage=downstream_stage_by_name[name],
                        message=(
                            f"{_pressure_report_label(name)} 결과를 "
                            "백엔드에 저장했습니다."
                            if result_ok
                            else (
                                f"{_pressure_report_label(name)} 결과의 "
                                "백엔드 저장에 실패했습니다."
                            )
                        ),
                        details=_backend_progress_details(name, result),
                    )
                    if isinstance(result, httpx.Response) and _is_2xx(result):
                        completed_stages.append(downstream_stage_by_name[name])

                downstream_transport_failures = [
                    (name, result)
                    for name, result in downstream_results.items()
                    if isinstance(result, BaseException)
                ]
                if downstream_transport_failures:
                    name, transport_error = downstream_transport_failures[0]
                    reports = {
                        "pressureHeatmap": _result_summary(
                            core_results["pressureHeatmap"]
                        ),
                        "pressureDistribution": _result_summary(
                            core_results["pressureDistribution"]
                        ),
                        "balance": _result_summary(
                            downstream_results.get(
                                "balance",
                                RuntimeError("not attempted"),
                            )
                        ),
                        "summary": _result_summary(
                            downstream_results.get(
                                "summary",
                                RuntimeError("not attempted"),
                            )
                        ),
                    }
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail={
                            "pipeline": PIPELINE_VERSION,
                            "stage": downstream_stage_by_name[name],
                            "message": "A pressure downstream report transport request failed.",
                            "requestId": request_id,
                            "errorType": type(transport_error).__name__,
                            "retryable": _is_retryable_transport(transport_error),
                            "partialSave": True,
                            "completedStages": completed_stages,
                            "reports": reports,
                        },
                        headers=_trace_headers(request_id),
                    )
            else:
                skip_message = (
                    "필수 압력 결과가 저장되지 않아 GPT 코멘트 생성을 건너뜁니다."
                    if not both_core_saved
                    else "양발의 유효 압력이 부족해 GPT 코멘트 생성을 건너뜁니다."
                )
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=request.measurement_session_id,
                    event="skipped",
                    stage="pressure-comment-generation",
                    message=skip_message,
                    details={"status": "skipped"},
                )
                for name in ("balance", "summary"):
                    _log_progress(
                        request_id=request_id,
                        measurement_session_id=request.measurement_session_id,
                        event="skipped",
                        stage=downstream_stage_by_name[name],
                        message=(
                            f"{_pressure_report_label(name)} 결과의 "
                            "백엔드 저장을 건너뜁니다."
                        ),
                        details={"report": name, "status": "skipped"},
                    )
    except HTTPException as exc:
        if isinstance(exc.detail, dict) and exc.detail.get("errorType") == "PressureMeasurementAttemptConflict":
            raise
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        terminal_stage = str(detail.get("stage") or stage)
        unauthorized_failure = next(
            (
                (
                    core_stage_by_name[name]
                    if name in core_stage_by_name
                    else downstream_stage_by_name[name]
                )
                for name, result in (
                    *core_results.items(),
                    *downstream_results.items(),
                )
                if isinstance(result, httpx.Response)
                and int(result.status_code) == status.HTTP_401_UNAUTHORIZED
            ),
            None,
        )
        _log_progress(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            event="failed",
            stage=terminal_stage,
            message="오류가 발생해 압력 분석을 완료하지 못했습니다.",
            details={
                "statusCode": int(exc.status_code),
                "retryable": detail.get("retryable"),
                "partialSave": detail.get("partialSave"),
                "completedStageCount": len(completed_stages),
            },
        )
        if unauthorized_failure is not None:
            # Concurrent writes can return a concrete 401 alongside a
            # transport exception. Preserve the transport-facing HTTP result,
            # but prioritize the actionable authentication failure callback.
            failure_reason = "TOKEN_EXPIRED"
            callback_stage = unauthorized_failure
            callback_error_type = "UpstreamUnauthorized"
        elif exc.status_code == status.HTTP_502_BAD_GATEWAY:
            failure_reason = "NETWORK_ERROR"
            callback_stage = terminal_stage
            callback_error_type = str(
                detail.get("errorType") or "PressurePipelineError"
            )
        elif exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
            failure_reason = "INVALID_CAPTURE_DATA"
            callback_stage = terminal_stage
            callback_error_type = str(
                detail.get("errorType") or "PressurePipelineError"
            )
        else:
            failure_reason = "AI_SERVER_ERROR"
            callback_stage = terminal_stage
            callback_error_type = str(
                detail.get("errorType") or "PressurePipelineError"
            )
        await notify_terminal_failure(
            failure_reason=failure_reason,
            failure_stage=callback_stage,
            error_type=callback_error_type,
        )
        raise
    except Exception as exc:
        reports = {
            "pressureHeatmap": (
                _result_summary(core_results["pressureHeatmap"])
                if "pressureHeatmap" in core_results
                else _skip_summary("not attempted")
            ),
            "pressureDistribution": (
                _result_summary(core_results["pressureDistribution"])
                if "pressureDistribution" in core_results
                else _skip_summary("not attempted")
            ),
            "balance": (
                _result_summary(downstream_results["balance"])
                if "balance" in downstream_results
                else _skip_summary("not attempted")
            ),
            "summary": (
                _result_summary(downstream_results["summary"])
                if "summary" in downstream_results
                else _skip_summary("not attempted")
            ),
        }
        partial_save = any(
            isinstance(result, httpx.Response) and _is_2xx(result)
            for result in (*core_results.values(), *downstream_results.values())
        )
        _log_progress(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            event="failed",
            stage=stage,
            message="백엔드 저장 중 오류가 발생해 압력 분석을 완료하지 못했습니다.",
            details={
                "statusCode": status.HTTP_502_BAD_GATEWAY,
                "errorType": type(exc).__name__,
                "retryable": _is_retryable_transport(exc),
                "partialSave": partial_save,
                "completedStageCount": len(completed_stages),
            },
        )
        await notify_terminal_failure(
            failure_reason=(
                "NETWORK_ERROR" if _is_retryable_transport(exc) else "AI_SERVER_ERROR"
            ),
            failure_stage=stage,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": stage,
                "message": "The pressure report backend connection could not be completed.",
                "requestId": request_id,
                "errorType": type(exc).__name__,
                "retryable": _is_retryable_transport(exc),
                "partialSave": partial_save,
                "completedStages": completed_stages,
                "reports": reports,
            },
            headers=_trace_headers(request_id),
        ) from exc

    core_all_saved = (
        set(core_results) == {"pressureHeatmap", "pressureDistribution"}
        and all(
            isinstance(result, httpx.Response) and _is_2xx(result)
            for result in core_results.values()
        )
    )
    downstream_all_saved = (
        set(downstream_results) == {"balance", "summary"}
        and all(
            isinstance(result, httpx.Response) and _is_2xx(result)
            for result in downstream_results.values()
        )
    )

    if "pressureHeatmap" in core_results:
        heatmap_report = _result_summary(core_results["pressureHeatmap"])
    else:
        heatmap_report = _skip_summary("not attempted")
    if "pressureDistribution" in core_results:
        distribution_report = _result_summary(core_results["pressureDistribution"])
    else:
        distribution_report = _skip_summary("combined load is zero")

    if downstream_results:
        balance_report = _result_summary(downstream_results["balance"])
        summary_report = _result_summary(downstream_results["summary"])
    else:
        if not core_all_saved:
            downstream_skip_reason = "required pressure core reports did not return 2xx"
        else:
            downstream_skip_reason = balance_skip_reason or "pressure balance is inconclusive"
        balance_report = _skip_summary(downstream_skip_reason)
        summary_report = _skip_summary(downstream_skip_reason)

    reports = {
        "pressureHeatmap": heatmap_report,
        "pressureDistribution": distribution_report,
        "balance": balance_report,
        "summary": summary_report,
    }
    failed_stages: list[str] = []
    errors: list[dict[str, object]] = []
    for name, result in (*core_results.items(), *downstream_results.items()):
        if not isinstance(result, httpx.Response) or _is_2xx(result):
            continue
        error_stage = (
            core_stage_by_name[name]
            if name in core_stage_by_name
            else downstream_stage_by_name[name]
        )
        failed_stages.append(error_stage)
        errors.append(
            {
                "stage": error_stage,
                "errorType": "UpstreamHTTPError",
                "statusCode": int(result.status_code),
                "retryable": _is_retryable_http_status(int(result.status_code)),
            }
        )

    all_saved = core_all_saved and downstream_all_saved
    partial_save = (not all_saved) and any(
        isinstance(result, httpx.Response) and _is_2xx(result)
        for result in (*core_results.values(), *downstream_results.values())
    )
    if not all_saved:
        failed_http_results = [
            (name, result)
            for name, result in (*core_results.items(), *downstream_results.items())
            if isinstance(result, httpx.Response) and not _is_2xx(result)
        ]
        unauthorized = next(
            (
                (name, result)
                for name, result in failed_http_results
                if int(result.status_code) == status.HTTP_401_UNAUTHORIZED
            ),
            None,
        )
        if unauthorized is not None:
            failure_name, _ = unauthorized
            failure_reason = "TOKEN_EXPIRED"
            failure_stage = (
                core_stage_by_name[failure_name]
                if failure_name in core_stage_by_name
                else downstream_stage_by_name[failure_name]
            )
            error_type = "UpstreamUnauthorized"
        elif failed_http_results:
            failure_name, _ = failed_http_results[0]
            failure_reason = "REPORT_SAVE_ERROR"
            failure_stage = (
                core_stage_by_name[failure_name]
                if failure_name in core_stage_by_name
                else downstream_stage_by_name[failure_name]
            )
            error_type = "UpstreamHTTPError"
        elif not balance_conclusive:
            failure_reason = "PRESSURE_SENSOR_ERROR"
            failure_stage = "pressure-balance-analysis"
            quality_status = str(pressure_facts.get("qualityStatus") or "inconclusive")
            error_type = {
                "no-load": "NoLoadPressureSignal",
                "single-side-no-load": "SingleSidePressureSignal",
            }.get(quality_status, "InconclusivePressureSignal")
        elif partial_save:
            failure_reason = "REPORT_SAVE_ERROR"
            failure_stage = "pressure-pipeline"
            error_type = "PartialReportSave"
        else:
            failure_reason = "UNKNOWN"
            failure_stage = "pressure-pipeline"
            error_type = "IncompletePressurePipeline"
        await notify_terminal_failure(
            failure_reason=failure_reason,
            failure_stage=failure_stage,
            error_type=error_type,
        )
    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        event="completed" if all_saved else "partial",
        stage="pressure-pipeline",
        message=(
            "압력 분석과 모든 결과 저장을 완료했습니다."
            if all_saved
            else "압력 분석은 끝났지만 일부 결과가 저장되지 않았습니다."
        ),
        details={
            "statusCode": (
                status.HTTP_200_OK if all_saved else status.HTTP_207_MULTI_STATUS
            ),
            "coreReportsSaved": sum(
                isinstance(result, httpx.Response) and _is_2xx(result)
                for result in core_results.values()
            ),
            "downstreamReportsSaved": sum(
                isinstance(result, httpx.Response) and _is_2xx(result)
                for result in downstream_results.values()
            ),
            "completedStageCount": len(completed_stages),
            "failedStageCount": len(failed_stages),
            "partialSave": partial_save,
            "allSaved": all_saved,
        },
    )
    return JSONResponse(
        content=jsonable_encoder(
            {
                "measurementSessionId": request.measurement_session_id,
                "pipeline": PIPELINE_VERSION,
                "requestId": request_id,
                "completedStages": completed_stages,
                "partialSave": partial_save,
                "failedStages": failed_stages,
                "errors": errors,
                "analysis": {
                    "pressureFacts": pressure_facts,
                    "score": balance_score,
                    "reportTextGeneration": (
                        {
                            "source": generated_report.generation_source,
                            "model": settings.openai_report_model if generated_report.generation_source == "OPENAI" else None,
                            **({"failureCode": code} if (code := _safe_generation_failure_code(
                                getattr(generated_report, "generation_failure_code", None)
                            )) is not None else {}),
                        }
                        if generated_report is not None else None
                    ),
                    "leftRegionalObservation": (
                        generated_report.left_regional_observation
                        if generated_report is not None
                        else None
                    ),
                    "rightRegionalObservation": (
                        generated_report.right_regional_observation
                        if generated_report is not None
                        else None
                    ),
                    "overallBalanceObservation": (
                        generated_report.overall_balance_observation
                        if generated_report is not None
                        else None
                    ),
                    "practicalTip": (
                        generated_report.practical_tip
                        if generated_report is not None
                        else None
                    ),
                    "balanceComment": (
                        generated_report.balance_comment
                        if generated_report is not None
                        else None
                    ),
                    "advice": (
                        list(generated_report.advice)
                        if generated_report is not None
                        else []
                    ),
                },
                "reports": reports,
            }
        ),
        status_code=(
            status.HTTP_200_OK if all_saved else status.HTTP_207_MULTI_STATUS
        ),
        headers=_trace_headers(request_id),
    )


async def _run_pressure_analysis_job(
    work_item: PressureAnalysisWorkItem,
) -> None:
    """Run one accepted pressure job independently of the request lifecycle."""

    try:
        async with _pressure_analysis_worker_semaphore:
            _log_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                event="started",
                stage="queue",
                message="대기열의 압력 분석 작업을 백그라운드에서 시작합니다.",
                details={"pressureMeasurementAttempt": work_item.pressure_measurement_attempt},
            )
            try:
                response = await _execute_pressure_pipeline(work_item)
            except HTTPException as exc:
                # The pipeline maps and reports every structured terminal
                # failure before raising.  Do not emit a duplicate callback.
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                _log_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    event="failed",
                    stage=str(detail.get("stage") or "background-worker"),
                    message="백그라운드 압력 분석 작업이 실패했습니다.",
                    details={"statusCode": int(exc.status_code)},
                )
                return
            except Exception as exc:
                error_type = type(exc).__name__
                logger.error(
                    "Background pressure analysis failed. pipeline=%s "
                    "request_id=%s stage=%s error_type=%s",
                    PIPELINE_VERSION,
                    work_item.request_id,
                    "background-worker",
                    error_type,
                )
                await _notify_pressure_job_failure(
                    work_item,
                    failure_reason="AI_SERVER_ERROR",
                    stage="background-worker",
                    error_type=error_type,
                )
                _log_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    event="failed",
                    stage="background-worker",
                    message="백그라운드 압력 분석 중 예상하지 못한 오류가 발생했습니다.",
                    details={"statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR},
                )
                return

            succeeded = response.status_code == status.HTTP_200_OK
            _log_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                event="completed" if succeeded else "failed",
                stage="queue",
                message=(
                    "백그라운드 압력 분석과 결과 저장을 모두 완료했습니다."
                    if succeeded
                    else "압력 분석 결과 일부를 저장하지 못해 실패 상태를 전달했습니다."
                ),
                details={"statusCode": int(response.status_code)},
            )
    finally:
        await _release_pressure_analysis_session(work_item.measurement_session_id)
