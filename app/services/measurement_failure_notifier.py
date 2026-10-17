"""Best-effort notification of terminal AI-pipeline failures.

The backend failure detail is one short, allowlisted Korean sentence. Raw
exception messages, uploaded filenames, and credentials must never become
part of the backend query string or operator logs.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import httpx

from app.core.config import settings
from app.services.photo_capture_status import (
    is_photo_capture_attempt_conflict,
    valid_measurement_session_id,
    valid_photo_capture_attempt,
)
from app.services.photo_failure_guidance import photo_failure_guidance


logger = logging.getLogger(__name__)

FailureReason = Literal[
    "CAMERA_ERROR",
    "PRESSURE_SENSOR_ERROR",
    "AI_SERVER_ERROR",
    "HARDWARE_TIMEOUT",
    "NETWORK_ERROR",
    "USER_CANCELLED",
    "REPORT_SAVE_ERROR",
    "INVALID_CAPTURE_DATA",
    "INVALID_PRESSURE_DATA",
    "DEVICE_DISCONNECTED",
    "TOKEN_EXPIRED",
    "UNKNOWN",
]

_FAILURE_REASONS = frozenset(
    {
        "CAMERA_ERROR",
        "PRESSURE_SENSOR_ERROR",
        "AI_SERVER_ERROR",
        "HARDWARE_TIMEOUT",
        "NETWORK_ERROR",
        "USER_CANCELLED",
        "REPORT_SAVE_ERROR",
        "INVALID_CAPTURE_DATA",
        "INVALID_PRESSURE_DATA",
        "DEVICE_DISCONNECTED",
        "TOKEN_EXPIRED",
        "UNKNOWN",
    }
)
_SAFE_DETAIL_CHARACTER = re.compile(r"[^A-Za-z0-9_.:-]+")
_MAX_DETAIL_VALUE_LENGTH = 80
_FAILURE_DETAILS = {
    "CAMERA_ERROR": "사진 촬영에 실패했습니다.",
    "PRESSURE_SENSOR_ERROR": "압력 측정에 실패했습니다.",
    "AI_SERVER_ERROR": "AI 분석에 실패했습니다.",
    "HARDWARE_TIMEOUT": "장치 응답 시간이 초과되었습니다.",
    "NETWORK_ERROR": "서버 통신에 실패했습니다.",
    "USER_CANCELLED": "측정이 취소되었습니다.",
    "REPORT_SAVE_ERROR": "분석 결과 저장에 실패했습니다.",
    "INVALID_CAPTURE_DATA": "사진 촬영에 실패했습니다.",
    "INVALID_PRESSURE_DATA": "압력 측정에 실패했습니다.",
    "DEVICE_DISCONNECTED": "장치 연결이 끊어졌습니다.",
    "TOKEN_EXPIRED": "인증이 만료되었습니다.",
    "UNKNOWN": "측정 처리에 실패했습니다.",
}
_CAPTURE_FAILURE_DETAILS = {
    "dorsalFootImage": "발등 촬영에 실패했습니다.",
    "leftPlantarFootImage": "왼발 발바닥 촬영에 실패했습니다.",
    "rightPlantarFootImage": "오른발 발바닥 촬영에 실패했습니다.",
}
_DORSAL_MARKER_DIAGNOSTIC_CODES = frozenset({
    "DORSAL_MARKERS_INSUFFICIENT",
    "DORSAL_ARUCO_GEOMETRY_INVALID",
    "DORSAL_MARKER_OCCLUDED_BY_FOOT",
})


def _safe_detail_value(value: object) -> str:
    """Return a compact ASCII value safe to include in a query parameter."""

    sanitized = _SAFE_DETAIL_CHARACTER.sub("_", str(value)).strip("_.:-")
    if not sanitized:
        return "unknown"
    return sanitized[:_MAX_DETAIL_VALUE_LENGTH]


def _build_failure_detail(
    *,
    failure_reason: str,
    diagnostic_code: str | None = None,
    image_field: str | None = None,
) -> str:
    """Return app-ready copy without actions or diagnostic metadata."""

    guidance = photo_failure_guidance(diagnostic_code, image_field)
    if guidance is not None:
        code = guidance.diagnostic_code
        if code.startswith("GPT_REPORT_"):
            return "사진 분석 결과 생성에 실패했습니다."
        if code.startswith("DORSAL_"):
            return _CAPTURE_FAILURE_DETAILS["dorsalFootImage"]
        if code.startswith("LEFT_PLANTAR_"):
            return _CAPTURE_FAILURE_DETAILS["leftPlantarFootImage"]
        if code.startswith("RIGHT_PLANTAR_"):
            return _CAPTURE_FAILURE_DETAILS["rightPlantarFootImage"]
        return _CAPTURE_FAILURE_DETAILS.get(str(image_field), _FAILURE_DETAILS["CAMERA_ERROR"])
    return _FAILURE_DETAILS.get(failure_reason, _FAILURE_DETAILS["UNKNOWN"])


def _confirmed_marker_failure_state(
    response: httpx.Response,
    *,
    measurement_session_id: int,
    photo_capture_attempt: int | None,
    failure_reason: str,
) -> bool:
    """Only a valid persisted FAILED state can confirm a marker callback."""
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    state = body.get("result", body)
    if not isinstance(state, dict) or state.get("status") != "FAILED":
        return False
    for fields in (body, state):
        if "isSuccess" in fields and fields["isSuccess"] is not True:
            return False
        if "status" in fields and fields["status"] != "FAILED":
            return False
        if "measurementSessionId" in fields:
            session_id = fields["measurementSessionId"]
            if not valid_measurement_session_id(session_id) or session_id != measurement_session_id:
                return False
        if "photoCaptureAttempt" in fields:
            attempt = fields["photoCaptureAttempt"]
            if not valid_photo_capture_attempt(attempt) or attempt != photo_capture_attempt:
                return False
        # INVALID_CAPTURE_DATA and AI_SERVER_ERROR can both be truthful marker
        # outcomes. Validate an explicit echo against what this caller sent.
        if "failureReason" in fields and fields["failureReason"] != failure_reason:
            return False
    return True


async def notify_measurement_failed(
    *,
    measurement_session_id: int,
    authorization: str,
    failure_reason: FailureReason | str,
    pipeline: str,
    stage: str,
    error_type: str,
    request_id: str,
    diagnostic_code: str | None = None,
    image_field: str | None = None,
    validation_field: str | None = None,
    validation_rule: str | None = None,
    photo_capture_attempt: int | None = None,
    pressure_measurement_attempt: int | None = None,
    marker_occlusion_reason: str | None = None,
) -> bool:
    """PATCH the measurement session to FAILED without propagating failures.

    ``True`` means the report server returned a 2xx response without a capture
    attempt conflict. Marker-specific callbacks also require an explicitly
    confirmed FAILED state with no contradictory identity/attempt echoes.
    Invalid input, transport errors, and rejected responses
    are logged using safe metadata and return ``False`` so this secondary
    callback cannot mask the pipeline's original failure. A supplied photo
    attempt must be an integer from zero through three and is sent unchanged.
    """

    if pressure_measurement_attempt is not None and (
        type(pressure_measurement_attempt) is not int or not 0 <= pressure_measurement_attempt <= 2_147_483_647
    ):
        logger.warning("Pressure failure callback skipped. code=INVALID_PRESSURE_MEASUREMENT_ATTEMPT")
        return False
    if measurement_session_id <= 0:
        logger.warning(
            "측정 실패 상태 전달을 건너뜁니다: 측정 세션 ID가 올바르지 않습니다."
        )
        return False
    if failure_reason not in _FAILURE_REASONS:
        logger.warning(
            "측정 실패 상태 전달을 건너뜁니다: 실패 사유가 올바르지 않습니다. "
            "measurement_session_id=%s",
            measurement_session_id,
        )
        return False
    if not authorization:
        logger.warning(
            "측정 실패 상태 전달을 건너뜁니다: Authorization 헤더가 없습니다. "
            "measurement_session_id=%s",
            measurement_session_id,
        )
        return False
    if photo_capture_attempt is not None and not valid_photo_capture_attempt(
        photo_capture_attempt
    ):
        logger.warning(
            "측정 실패 상태 전달을 건너뜁니다: 사진 촬영 시도 번호가 올바르지 않습니다. "
            "measurement_session_id=%s",
            measurement_session_id,
        )
        return False

    try:
        endpoint = settings.measurement_status_endpoint_template.format(
            measurement_session_id=measurement_session_id
        )
        failure_detail = _build_failure_detail(
            failure_reason=failure_reason,
            diagnostic_code=diagnostic_code,
            image_field=image_field,
        )
        params: dict[str, str | int] = {
            "status": "FAILED",
            "failureReason": failure_reason,
            "failureDetail": failure_detail,
        }
        if photo_capture_attempt is not None:
            params["photoCaptureAttempt"] = photo_capture_attempt
        if pressure_measurement_attempt is not None:
            params["pressureMeasurementAttempt"] = pressure_measurement_attempt
        guidance = photo_failure_guidance(diagnostic_code, image_field)
        if guidance is not None and guidance.diagnostic_code in _DORSAL_MARKER_DIAGNOSTIC_CODES:
            # Supply the current failure context explicitly so the backend
            # need not reuse a diagnostic saved during an earlier recapture.
            params["diagnosticCode"] = guidance.diagnostic_code
        async with httpx.AsyncClient(
            timeout=settings.measurement_failure_notification_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.patch(
                endpoint,
                headers={"Authorization": authorization},
                params=params,
            )
    except Exception as exc:  # best-effort callback must never mask the root failure
        logger.warning(
            "백엔드에 측정 실패 상태를 전달하지 못했습니다. "
            "measurement_session_id=%s callback_error_type=%s",
            measurement_session_id,
            type(exc).__name__,
        )
        return False

    if is_photo_capture_attempt_conflict(response):
        logger.warning(
            "사진 촬영 시도 상태가 달라 실패 상태 전달을 중단했습니다. "
            "measurement_session_id=%s status_code=%s conflict=photo_capture_attempt",
            measurement_session_id,
            response.status_code,
        )
        return False

    if not 200 <= response.status_code < 300:
        logger.warning(
            "백엔드가 측정 실패 상태 저장 요청을 거절했습니다. "
            "measurement_session_id=%s status_code=%s",
            measurement_session_id,
            response.status_code,
        )
        return False

    if "diagnosticCode" in params and not _confirmed_marker_failure_state(
        response,
        measurement_session_id=measurement_session_id,
        photo_capture_attempt=photo_capture_attempt,
        failure_reason=failure_reason,
    ):
        logger.warning(
            "백엔드의 마커 촬영 실패 상태를 확인하지 못했습니다. "
            "measurement_session_id=%s status_code=%s code=UNCONFIRMED_MARKER_FAILURE",
            measurement_session_id,
            response.status_code,
        )
        return False

    logger.info(
        "백엔드에 측정 실패 상태 전달을 완료했습니다. measurement_session_id=%s",
        measurement_session_id,
    )
    return True
