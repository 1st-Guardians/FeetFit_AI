from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Security, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.core.config import settings
from app.core.security import bearer_scheme
from app.schemas.environment import (
    FootEnvironmentAnalysisRequest,
    FootEnvironmentBackendRequest,
    FootEnvironmentSummaryRequest,
)
from app.services.measurement_failure_notifier import notify_measurement_failed
from app.services.report_text_generation import generate_foot_environment_report


router = APIRouter()
PIPELINE_VERSION = "foot-environment-v1"
ENVIRONMENT_ANALYSIS_MAX_PENDING_JOBS = 3
progress_logger = logging.getLogger("uvicorn.error")

_ENVIRONMENT_RESPONSE_HEADERS = {
    "X-Feetfit-Pipeline": {
        "description": "Environment-analysis pipeline version.",
        "schema": {"type": "string"},
    },
    "X-Feetfit-Request-Id": {
        "description": "Safe per-request correlation identifier.",
        "schema": {"type": "string"},
    },
}


@dataclass(frozen=True, slots=True, repr=False)
class EnvironmentAnalysisWorkItem:
    """Immutable, request-independent input owned by one background job."""

    request_id: str
    measurement_session_id: int
    before_temperature_celsius: float
    before_humidity_percent: float
    after_temperature_celsius: float
    after_humidity_percent: float
    authorization: str


class EnvironmentPipelineFailure(Exception):
    """Safe terminal metadata propagated from the executor to its worker."""

    def __init__(
        self,
        *,
        status_code: int,
        failure_reason: str,
        stage: str,
        error_type: str,
    ) -> None:
        super().__init__(error_type)
        self.status_code = status_code
        self.failure_reason = failure_reason
        self.stage = stage
        self.error_type = error_type


# Environment work is serialized only against other environment jobs.  The
# photo and pressure routes own separate semaphores, so all three pipelines can
# progress in parallel without sharing model/queue capacity.
_environment_analysis_worker_semaphore = asyncio.Semaphore(1)
_environment_analysis_reservation_lock = threading.Lock()
_active_environment_analysis_sessions: set[int] = set()

_NUMERIC_LOG_DETAIL_KEYS = {
    "beforeTemperatureCelsius",
    "beforeHumidityPercent",
    "afterTemperatureCelsius",
    "afterHumidityPercent",
    "avgTemperatureCelsius",
    "avgHumidityPercent",
    "temperatureDeltaCelsius",
    "humidityDeltaPercent",
    "score",
    "adviceCount",
    "statusCode",
}
_BOOLEAN_LOG_DETAIL_KEYS = {"adviceGenerated", "saved", "retryable"}
_ENUM_LOG_DETAIL_KEYS = {
    "generationSource",
    "report",
    "status",
    "reason",
    "outcome",
    "errorType",
}
_SHORT_ENUM_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _new_request_id() -> str:
    return str(uuid4())


def _allowlisted_log_details(details: dict[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (details or {}).items():
        if key in _NUMERIC_LOG_DETAIL_KEYS:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"Log detail {key} must be numeric.")
        elif key in _BOOLEAN_LOG_DETAIL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"Log detail {key} must be boolean.")
        elif key in _ENUM_LOG_DETAIL_KEYS:
            if not isinstance(value, str) or not _SHORT_ENUM_PATTERN.fullmatch(value):
                raise ValueError(f"Log detail {key} must be a short enum.")
        else:
            raise ValueError(f"Log detail {key} is not allowlisted.")
        safe[key] = value
    return safe


def _log_progress(
    *,
    request_id: str,
    measurement_session_id: int,
    stage: str,
    event: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "pipeline": PIPELINE_VERSION,
        "requestId": request_id,
        "measurementSessionId": measurement_session_id,
        "event": event,
        "stage": stage,
        "message": message,
        "details": _allowlisted_log_details(details),
    }
    progress_logger.info(
        "[온습도 분석][세션 %s][요청 %s] %s | pipeline_progress %s",
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


_ENVIRONMENT_REPORT_LABELS = {
    "ENVIRONMENT": "평균 온도·습도 결과",
    "FOOT_ENVIRONMENT_SUMMARY": "FOOT_ENVIRONMENT 지표 리포트",
}


def _environment_report_label(report: str) -> str:
    return _ENVIRONMENT_REPORT_LABELS.get(report, "온습도 분석 결과")


def _generation_source() -> str:
    if settings.openai_report_text_enabled and (
        settings.openai_api_key or ""
    ).strip():
        return "GPT_WITH_DETERMINISTIC_FALLBACK"
    return "DETERMINISTIC_FALLBACK"


def _error_type_enum(exc: BaseException) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).upper()
    sanitized = re.sub(r"[^A-Z0-9_]", "_", words).strip("_")
    return (sanitized or "UNKNOWN_ERROR")[:64]


def _is_2xx(response: httpx.Response) -> bool:
    return 200 <= int(response.status_code) < 300


def _response_summary(response: httpx.Response) -> dict[str, object]:
    try:
        body: object = response.json() if response.content else None
    except (json.JSONDecodeError, ValueError):
        body = response.text[:2000]
    return {
        "statusCode": int(response.status_code),
        "contentType": response.headers.get("content-type") or None,
        "body": body,
    }


async def _notify_environment_failure(
    *,
    measurement_session_id: int,
    authorization: str,
    failure_reason: str,
    stage: str,
    error_type: str,
    request_id: str,
) -> None:
    """Send one best-effort FAILED callback without masking the root failure."""

    try:
        await notify_measurement_failed(
            measurement_session_id=measurement_session_id,
            authorization=authorization,
            failure_reason=failure_reason,
            pipeline=PIPELINE_VERSION,
            stage=stage,
            error_type=error_type,
            request_id=request_id,
        )
    except Exception as exc:
        # The shared notifier already absorbs its own failures.  Keep this
        # defensive boundary so even a regression in that integration cannot
        # replace the environment pipeline's original response or exception.
        progress_logger.warning(
            "[온습도 분석][세션 %s][요청 %s] 백엔드에 실패 상태를 전달하지 "
            "못했지만 원래 분석 오류 응답은 그대로 반환합니다. | "
            "pipeline=%s stage=%s callback_error_type=%s",
            measurement_session_id,
            request_id[:8],
            PIPELINE_VERSION,
            stage,
            _error_type_enum(exc),
        )


async def _reserve_environment_analysis_session(
    *, measurement_session_id: int, request_id: str
) -> None:
    with _environment_analysis_reservation_lock:
        if measurement_session_id in _active_environment_analysis_sessions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "pipeline": PIPELINE_VERSION,
                    "requestId": request_id,
                    "message": (
                        "Environment analysis is already active for this "
                        "measurement session."
                    ),
                },
                headers={
                    "X-Feetfit-Pipeline": PIPELINE_VERSION,
                    "X-Feetfit-Request-Id": request_id,
                },
            )
        if (
            len(_active_environment_analysis_sessions)
            >= ENVIRONMENT_ANALYSIS_MAX_PENDING_JOBS
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "pipeline": PIPELINE_VERSION,
                    "requestId": request_id,
                    "message": "The environment analysis queue is full. Retry later.",
                },
                headers={
                    "X-Feetfit-Pipeline": PIPELINE_VERSION,
                    "X-Feetfit-Request-Id": request_id,
                    "Retry-After": "5",
                },
            )
        _active_environment_analysis_sessions.add(measurement_session_id)


async def _release_environment_analysis_session(
    measurement_session_id: int,
) -> None:
    with _environment_analysis_reservation_lock:
        _active_environment_analysis_sessions.discard(measurement_session_id)


async def _execute_foot_environment_pipeline(
    work_item: EnvironmentAnalysisWorkItem,
) -> JSONResponse:
    request_id = work_item.request_id
    measurement_session_id = work_item.measurement_session_id
    authorization = work_item.authorization
    before_temperature = work_item.before_temperature_celsius
    before_humidity = work_item.before_humidity_percent
    after_temperature = work_item.after_temperature_celsius
    after_humidity = work_item.after_humidity_percent
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-measurements",
        event="INPUT_ACCEPTED",
        message="온도·습도 전후 측정값을 정상적으로 받았습니다.",
        details={
            "beforeTemperatureCelsius": before_temperature,
            "beforeHumidityPercent": before_humidity,
            "afterTemperatureCelsius": after_temperature,
            "afterHumidityPercent": after_humidity,
        },
    )
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-measurements",
        event="ANALYSIS_STARTED",
        message="온도·습도 전후 변화와 평균값을 분석하는 중입니다.",
    )
    average_temperature = round(
        (before_temperature + after_temperature) / 2.0,
        4,
    )
    average_humidity = round((before_humidity + after_humidity) / 2.0, 4)
    temperature_delta = round(after_temperature - before_temperature, 4)
    humidity_delta = round(after_humidity - before_humidity, 4)
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-measurements",
        event="INPUT_SUMMARIZED",
        message="온도·습도 전후 변화와 평균값 분석을 완료했습니다.",
        details={
            "beforeTemperatureCelsius": before_temperature,
            "beforeHumidityPercent": before_humidity,
            "afterTemperatureCelsius": after_temperature,
            "afterHumidityPercent": after_humidity,
            "avgTemperatureCelsius": average_temperature,
            "avgHumidityPercent": average_humidity,
            "temperatureDeltaCelsius": temperature_delta,
            "humidityDeltaPercent": humidity_delta,
        },
    )

    # The generator computes the same deltas from these four raw finite values.
    # It always completes (with a deterministic fallback when needed) before
    # either backend write starts.
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-report-generation",
        event="GENERATION_STARTED",
        message="GPT로 발 환경 점수와 관리 멘트를 생성하는 중입니다.",
        details={"generationSource": _generation_source()},
    )
    try:
        generated = await generate_foot_environment_report(
            before_temperature_celsius=before_temperature,
            before_humidity_percent=before_humidity,
            after_temperature_celsius=after_temperature,
            after_humidity_percent=after_humidity,
        )
    except Exception as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="environment-report-generation",
            event="GENERATION_FAILED",
            message="발 환경 점수와 관리 멘트 생성에 실패했습니다.",
            details={
                "errorType": _error_type_enum(exc),
                "retryable": False,
            },
        )
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="environment-pipeline",
            event="PIPELINE_COMPLETED",
            message="내부 오류가 발생해 온습도 분석을 완료하지 못했습니다.",
            details={"statusCode": 500, "outcome": "GENERATION_ERROR"},
        )
        raise EnvironmentPipelineFailure(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            failure_reason="AI_SERVER_ERROR",
            stage="environment-report-generation",
            error_type=_error_type_enum(exc),
        ) from exc
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-report-generation",
        event="REPORT_GENERATED",
        message="발 환경 점수와 관리 멘트 생성을 완료했습니다.",
        details={
            "generationSource": _generation_source(),
            "score": generated.score,
            "adviceCount": len(generated.advice),
            "adviceGenerated": bool(generated.advice),
        },
    )

    environment_request = FootEnvironmentBackendRequest(
        measurement_session_id=measurement_session_id,
        before_temperature_celsius=before_temperature,
        before_humidity_percent=before_humidity,
        after_temperature_celsius=after_temperature,
        after_humidity_percent=after_humidity,
    )
    summary_request = FootEnvironmentSummaryRequest(
        measurement_session_id=measurement_session_id,
        metric_type="FOOT_ENVIRONMENT",
        score=generated.score,
        advice=tuple(generated.advice),
    )
    headers = {
        "accept": "*/*",
        "Authorization": authorization,
    }

    backend_stage = "environment-backend-save"
    backend_report = "ENVIRONMENT"
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage=backend_stage,
        event="BACKEND_SAVE_STARTED",
        message="평균 온도·습도 결과를 백엔드에 저장하는 중입니다.",
        details={"report": backend_report},
    )
    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            environment_response = await client.post(
                settings.environment_report_endpoint,
                headers=headers,
                json=environment_request.model_dump(by_alias=True, mode="json"),
            )
            if _is_2xx(environment_response):
                backend_stage = "foot-environment-summary-save"
                backend_report = "FOOT_ENVIRONMENT_SUMMARY"
                _log_progress(
                    request_id=request_id,
                    measurement_session_id=measurement_session_id,
                    stage=backend_stage,
                    event="BACKEND_SAVE_STARTED",
                    message=(
                        "FOOT_ENVIRONMENT 지표 리포트를 "
                        "백엔드에 저장하는 중입니다."
                    ),
                    details={"report": backend_report},
                )
                summary_response = await client.post(
                    settings.summary_report_endpoint,
                    headers=headers,
                    json=summary_request.model_dump(by_alias=True, mode="json"),
                )
            else:
                summary_response = None
    except httpx.HTTPError as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage=backend_stage,
            event="BACKEND_TRANSPORT_FAILED",
            message=(
                f"{_environment_report_label(backend_report)}를 "
                "백엔드에 전송하지 못했습니다."
            ),
            details={
                "report": backend_report,
                "errorType": _error_type_enum(exc),
                "retryable": True,
            },
        )
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="environment-pipeline",
            event="PIPELINE_COMPLETED",
            message="백엔드 연결 오류로 온습도 분석 결과를 저장하지 못했습니다.",
            details={"statusCode": 502, "outcome": "TRANSPORT_ERROR"},
        )
        raise EnvironmentPipelineFailure(
            status_code=status.HTTP_502_BAD_GATEWAY,
            failure_reason="NETWORK_ERROR",
            stage=backend_stage,
            error_type=_error_type_enum(exc),
        ) from exc
    except Exception as exc:
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage=backend_stage,
            event="BACKEND_INTERNAL_FAILED",
            message=(
                f"{_environment_report_label(backend_report)}의 "
                "백엔드 저장 처리 중 오류가 발생했습니다."
            ),
            details={
                "report": backend_report,
                "errorType": _error_type_enum(exc),
                "retryable": False,
            },
        )
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="environment-pipeline",
            event="PIPELINE_COMPLETED",
            message="내부 오류가 발생해 온습도 분석을 완료하지 못했습니다.",
            details={"statusCode": 500, "outcome": "INTERNAL_ERROR"},
        )
        raise EnvironmentPipelineFailure(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            failure_reason="AI_SERVER_ERROR",
            stage=backend_stage,
            error_type=_error_type_enum(exc),
        ) from exc

    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-backend-save",
        event="BACKEND_RESULT",
        message=(
            "평균 온도·습도 결과를 백엔드에 저장했습니다."
            if _is_2xx(environment_response)
            else "평균 온도·습도 결과의 백엔드 저장에 실패했습니다."
        ),
        details={
            "report": "ENVIRONMENT",
            "statusCode": int(environment_response.status_code),
            "saved": _is_2xx(environment_response),
        },
    )
    if summary_response is None:
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="foot-environment-summary-save",
            event="BACKEND_SKIPPED",
            message="온습도 원본 결과 저장에 실패해 지표 리포트 저장을 건너뜁니다.",
            details={
                "report": "FOOT_ENVIRONMENT_SUMMARY",
                "status": "SKIPPED",
                "reason": "CORE_NOT_2XX",
            },
        )
    else:
        _log_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            stage="foot-environment-summary-save",
            event="BACKEND_RESULT",
            message=(
                "FOOT_ENVIRONMENT 지표 리포트를 백엔드에 저장했습니다."
                if _is_2xx(summary_response)
                else "FOOT_ENVIRONMENT 지표 리포트의 백엔드 저장에 실패했습니다."
            ),
            details={
                "report": "FOOT_ENVIRONMENT_SUMMARY",
                "statusCode": int(summary_response.status_code),
                "saved": _is_2xx(summary_response),
            },
        )

    reports: dict[str, object] = {
        "environment": _response_summary(environment_response),
        "summary": (
            _response_summary(summary_response)
            if summary_response is not None
            else {
                "status": "skipped",
                "reason": "environment core report did not return 2xx",
            }
        ),
    }
    all_saved = _is_2xx(environment_response) and (
        summary_response is not None and _is_2xx(summary_response)
    )
    response_body = {
        "measurementSessionId": measurement_session_id,
        "pipeline": PIPELINE_VERSION,
        "analysis": {
            "beforeTemperatureCelsius": before_temperature,
            "beforeHumidityPercent": before_humidity,
            "afterTemperatureCelsius": after_temperature,
            "afterHumidityPercent": after_humidity,
            "avgTemperatureCelsius": average_temperature,
            "avgHumidityPercent": average_humidity,
            "temperatureDeltaCelsius": temperature_delta,
            "humidityDeltaPercent": humidity_delta,
            "score": generated.score,
            "advice": list(generated.advice),
        },
        "reports": reports,
    }
    response_status = status.HTTP_200_OK if all_saved else status.HTTP_207_MULTI_STATUS
    _log_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        stage="environment-pipeline",
        event="PIPELINE_COMPLETED",
        message=(
            "온습도 분석과 모든 결과 저장을 완료했습니다."
            if all_saved
            else "온습도 분석은 끝났지만 일부 결과가 저장되지 않았습니다."
        ),
        details={
            "statusCode": response_status,
            "outcome": "SUCCESS" if all_saved else "PARTIAL_SUCCESS",
        },
    )
    return JSONResponse(
        content=jsonable_encoder(response_body),
        status_code=response_status,
    )


def _partial_environment_failure_metadata(
    response: JSONResponse,
) -> tuple[str, str, str]:
    """Map a 207 executor result to one deterministic FAILED callback."""

    try:
        payload = json.loads(bytes(response.body))
    except (TypeError, ValueError, json.JSONDecodeError):
        return (
            "environment-response",
            "UnexpectedEnvironmentResponse",
            "REPORT_SAVE_ERROR",
        )

    reports = payload.get("reports", {}) if isinstance(payload, dict) else {}
    candidates = (
        ("environment", "environment-backend-save"),
        ("summary", "foot-environment-summary-save"),
    )
    first_failure: tuple[str, int] | None = None
    for report_name, stage_name in candidates:
        report = reports.get(report_name, {}) if isinstance(reports, dict) else {}
        if not isinstance(report, dict):
            continue
        raw_status = report.get("statusCode")
        try:
            status_code = int(raw_status)
        except (TypeError, ValueError):
            continue
        if 200 <= status_code < 300:
            continue
        if status_code == status.HTTP_401_UNAUTHORIZED:
            return stage_name, "UPSTREAM_HTTP_401", "TOKEN_EXPIRED"
        if first_failure is None:
            first_failure = (stage_name, status_code)

    if first_failure is not None:
        stage_name, status_code = first_failure
        return stage_name, f"UPSTREAM_HTTP_{status_code}", "REPORT_SAVE_ERROR"
    return (
        "environment-response",
        "UnexpectedEnvironmentResponse",
        "REPORT_SAVE_ERROR",
    )


async def _run_environment_analysis_job(
    work_item: EnvironmentAnalysisWorkItem,
) -> None:
    """Run one accepted environment job and own every terminal callback."""

    try:
        async with _environment_analysis_worker_semaphore:
            _log_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                stage="queue",
                event="JOB_STARTED",
                message=(
                    "대기열에서 온습도 분석 작업을 가져와 백그라운드 분석을 "
                    "시작합니다."
                ),
            )
            try:
                response = await _execute_foot_environment_pipeline(work_item)
            except EnvironmentPipelineFailure as exc:
                await _notify_environment_failure(
                    measurement_session_id=work_item.measurement_session_id,
                    authorization=work_item.authorization,
                    failure_reason=exc.failure_reason,
                    stage=exc.stage,
                    error_type=exc.error_type,
                    request_id=work_item.request_id,
                )
                _log_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage=exc.stage,
                    event="JOB_FAILED",
                    message=(
                        "온습도 분석 작업에 실패해 백엔드에 측정 실패 상태를 "
                        "전달했습니다."
                    ),
                    details={"statusCode": exc.status_code},
                )
                return
            except Exception as exc:
                error_type = _error_type_enum(exc)
                await _notify_environment_failure(
                    measurement_session_id=work_item.measurement_session_id,
                    authorization=work_item.authorization,
                    failure_reason="AI_SERVER_ERROR",
                    stage="environment-background-worker",
                    error_type=error_type,
                    request_id=work_item.request_id,
                )
                _log_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage="environment-background-worker",
                    event="JOB_FAILED",
                    message=(
                        "온습도 백그라운드 작업 중 예상하지 못한 오류가 발생해 "
                        "실패 상태를 전달했습니다."
                    ),
                    details={"statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR},
                )
                return

            if response.status_code == status.HTTP_200_OK:
                _log_progress(
                    request_id=work_item.request_id,
                    measurement_session_id=work_item.measurement_session_id,
                    stage="queue",
                    event="JOB_SUCCEEDED",
                    message="온습도 분석과 결과 저장을 모두 완료했습니다.",
                    details={"statusCode": status.HTTP_200_OK},
                )
                return

            stage_name, error_type, failure_reason = (
                _partial_environment_failure_metadata(response)
            )
            await _notify_environment_failure(
                measurement_session_id=work_item.measurement_session_id,
                authorization=work_item.authorization,
                failure_reason=failure_reason,
                stage=stage_name,
                error_type=error_type,
                request_id=work_item.request_id,
            )
            _log_progress(
                request_id=work_item.request_id,
                measurement_session_id=work_item.measurement_session_id,
                stage=stage_name,
                event="JOB_FAILED",
                message=(
                    "온습도 결과 일부를 저장하지 못해 작업을 실패로 "
                    "처리했습니다."
                ),
                details={"statusCode": int(response.status_code)},
            )
    finally:
        await _release_environment_analysis_session(
            work_item.measurement_session_id
        )


# Request-body validation happens in FastAPI before this handler runs.  A 422
# caused by missing/invalid values therefore cannot be tied to a trusted work
# item and intentionally does not emit a FAILED callback here.
@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="측정 전후 발 환경 온·습도 비동기 분석 및 저장",
    responses={
        202: {
            "description": "입력 검증 완료 및 온습도 분석 대기열 등록",
            "headers": _ENVIRONMENT_RESPONSE_HEADERS,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": [
                            "accepted",
                            "status",
                            "measurementSessionId",
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
                            "pipeline": {"type": "string"},
                            "requestId": {"type": "string"},
                        },
                    }
                }
            },
        },
        409: {
            "description": "같은 측정 세션의 온습도 분석이 이미 진행 중",
            "headers": _ENVIRONMENT_RESPONSE_HEADERS,
        },
        422: {"description": "측정값 형식 또는 허용 범위가 올바르지 않음"},
        500: {
            "description": "백그라운드 작업 등록 실패",
            "headers": _ENVIRONMENT_RESPONSE_HEADERS,
        },
        503: {
            "description": "온습도 분석 대기열이 가득 참",
            "headers": {
                **_ENVIRONMENT_RESPONSE_HEADERS,
                "Retry-After": {
                    "description": "재시도 전 최소 대기 시간(초)",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
async def create_foot_environment_report(
    background_tasks: BackgroundTasks,
    request: FootEnvironmentAnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> JSONResponse:
    request_id = _new_request_id()
    authorization = f"{credentials.scheme} {credentials.credentials}"
    work_item = EnvironmentAnalysisWorkItem(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        before_temperature_celsius=request.before_temperature_celsius,
        before_humidity_percent=request.before_humidity_percent,
        after_temperature_celsius=request.after_temperature_celsius,
        after_humidity_percent=request.after_humidity_percent,
        authorization=authorization,
    )
    await _reserve_environment_analysis_session(
        measurement_session_id=request.measurement_session_id,
        request_id=request_id,
    )
    try:
        background_tasks.add_task(_run_environment_analysis_job, work_item)
    except Exception as exc:
        await _release_environment_analysis_session(request.measurement_session_id)
        error_type = _error_type_enum(exc)
        await _notify_environment_failure(
            measurement_session_id=request.measurement_session_id,
            authorization=authorization,
            failure_reason="AI_SERVER_ERROR",
            stage="environment-background-schedule",
            error_type=error_type,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "pipeline": PIPELINE_VERSION,
                "requestId": request_id,
                "message": "Could not schedule the environment analysis job.",
            },
            headers={
                "X-Feetfit-Pipeline": PIPELINE_VERSION,
                "X-Feetfit-Request-Id": request_id,
            },
        ) from exc

    _log_progress(
        request_id=request_id,
        measurement_session_id=request.measurement_session_id,
        stage="queue",
        event="JOB_QUEUED",
        message=(
            "온습도 분석 요청을 접수했습니다. 백그라운드 분석 대기열에 "
            "등록했습니다."
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "accepted": True,
            "status": "QUEUED",
            "measurementSessionId": request.measurement_session_id,
            "pipeline": PIPELINE_VERSION,
            "requestId": request_id,
        },
        headers={
            "X-Feetfit-Pipeline": PIPELINE_VERSION,
            "X-Feetfit-Request-Id": request_id,
        },
    )
