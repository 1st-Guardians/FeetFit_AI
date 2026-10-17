"""Attempt-bound pressure status callbacks; the backend owns retry limits."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Literal

import httpx

from app.core.config import settings
from app.services.photo_capture_status import valid_measurement_session_id, valid_status_authorization

logger = logging.getLogger(__name__)
PressureMeasurementStatus = Literal["MEASURING_PRESSURE", "WAITING_FOR_PRESSURE_RETRY", "ANALYZING", "FAILED"]


@dataclass(frozen=True, slots=True)
class PressureStatusOutcome:
    confirmed: bool = False
    status: PressureMeasurementStatus | None = None
    attempt_conflict: bool = False
    backend_status_code: int | None = None


def valid_pressure_measurement_attempt(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2_147_483_647


def is_pressure_attempt_conflict(response: httpx.Response) -> bool:
    return response.status_code == 409


def parse_pressure_status_response(
    response: httpx.Response, *, measurement_session_id: int,
    pressure_measurement_attempt: int, expected_status: str,
) -> PressureStatusOutcome:
    unknown = PressureStatusOutcome(backend_status_code=response.status_code)
    if is_pressure_attempt_conflict(response):
        return PressureStatusOutcome(attempt_conflict=True, backend_status_code=response.status_code)
    if not 200 <= response.status_code < 300:
        return unknown
    try:
        body = response.json()
    except Exception:
        return unknown
    if not isinstance(body, dict):
        return unknown
    state = body.get("result", body)
    if not isinstance(state, dict):
        return unknown
    status = state.get("status")
    allowed = {expected_status}
    if expected_status == "WAITING_FOR_PRESSURE_RETRY":
        allowed.add("FAILED")
    if not isinstance(status, str) or status not in allowed:
        return unknown
    if status == "FAILED" and state.get("failureReason") != "INVALID_PRESSURE_DATA":
        return unknown
    for fields in (body, state):
        if "isSuccess" in fields and fields["isSuccess"] is not True:
            return unknown
        if "status" in fields and fields["status"] != status:
            return unknown
        if "measurementSessionId" in fields:
            value = fields["measurementSessionId"]
            if not valid_measurement_session_id(value) or value != measurement_session_id:
                return unknown
        if "pressureMeasurementAttempt" in fields:
            value = fields["pressureMeasurementAttempt"]
            if not valid_pressure_measurement_attempt(value) or value != pressure_measurement_attempt:
                return unknown
        if (status in {"WAITING_FOR_PRESSURE_RETRY", "FAILED"}
                and "failureReason" in fields and fields["failureReason"] != "INVALID_PRESSURE_DATA"):
            return unknown
    return PressureStatusOutcome(confirmed=True, status=status, backend_status_code=response.status_code)


async def _notify_pressure_status(
    *, measurement_session_id: int, authorization: str,
    pressure_measurement_attempt: int, request_id: str,
    target_status: Literal["MEASURING_PRESSURE", "WAITING_FOR_PRESSURE_RETRY", "ANALYZING"],
    error_type: str | None = None,
) -> PressureStatusOutcome:
    if (not valid_measurement_session_id(measurement_session_id)
            or not valid_pressure_measurement_attempt(pressure_measurement_attempt)
            or not valid_status_authorization(authorization)
            or not isinstance(request_id, str) or not request_id.strip()):
        logger.warning("Pressure status callback skipped. code=INVALID_CALLBACK_INPUT")
        return PressureStatusOutcome()
    safe_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", request_id).strip("_.:-")[:80] or "unknown"
    params: dict[str, str | int] = {
        "status": target_status,
        "pressureMeasurementAttempt": pressure_measurement_attempt,
        "requestId": safe_id,
    }
    if target_status == "WAITING_FOR_PRESSURE_RETRY":
        params["failureReason"] = "INVALID_PRESSURE_DATA"
        safe_error = re.sub(r"[^A-Za-z0-9_.:-]+", "_", error_type or "InvalidPressureSignal")[:80]
        params["failureDetail"] = (
            "압력 데이터가 부족하거나 유효하지 않습니다. 양발을 센서 위에 올리고 다시 측정해 주세요. "
            f"(오류 유형: {safe_error}, 요청 ID: {safe_id[:8]})"
        )
    try:
        endpoint = settings.measurement_status_endpoint_template.format(
            measurement_session_id=measurement_session_id,
        )
        async with httpx.AsyncClient(
            timeout=settings.measurement_failure_notification_timeout_seconds, trust_env=False,
        ) as client:
            response = await client.patch(endpoint, headers={"Authorization": authorization}, params=params)
        outcome = parse_pressure_status_response(
            response, measurement_session_id=measurement_session_id,
            pressure_measurement_attempt=pressure_measurement_attempt, expected_status=target_status,
        )
    except Exception:
        logger.warning("Pressure status callback could not be confirmed. measurement_session_id=%s code=CALLBACK_ERROR",
                       measurement_session_id)
        return PressureStatusOutcome()
    if not outcome.confirmed:
        logger.warning("Pressure status callback unconfirmed. measurement_session_id=%s status_code=%s",
                       measurement_session_id, outcome.backend_status_code)
    return outcome


async def notify_pressure_measurement_received(
    *, measurement_session_id: int, authorization: str,
    pressure_measurement_attempt: int, request_id: str,
) -> PressureStatusOutcome:
    return await _notify_pressure_status(
        measurement_session_id=measurement_session_id, authorization=authorization,
        pressure_measurement_attempt=pressure_measurement_attempt, request_id=request_id,
        target_status="MEASURING_PRESSURE",
    )


async def notify_pressure_retry_required(
    *, measurement_session_id: int, authorization: str,
    pressure_measurement_attempt: int, request_id: str, error_type: str,
) -> PressureStatusOutcome:
    return await _notify_pressure_status(
        measurement_session_id=measurement_session_id, authorization=authorization,
        pressure_measurement_attempt=pressure_measurement_attempt, request_id=request_id,
        target_status="WAITING_FOR_PRESSURE_RETRY", error_type=error_type,
    )


async def notify_pressure_analysis_started(
    *, measurement_session_id: int, authorization: str,
    pressure_measurement_attempt: int, request_id: str,
) -> PressureStatusOutcome:
    return await _notify_pressure_status(
        measurement_session_id=measurement_session_id, authorization=authorization,
        pressure_measurement_attempt=pressure_measurement_attempt, request_id=request_id,
        target_status="ANALYZING",
    )
