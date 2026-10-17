"""Attempt-bound photo status callbacks and safe backend-state validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import re
from typing import Literal

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)
PhotoCaptureStatus = Literal["WAITING_FOR_RECAPTURE", "WAITING_FOR_ENVIRONMENT", "FAILED"]


@dataclass(frozen=True, slots=True)
class PhotoCaptureStatusOutcome:
    # Keep the original recapture outcome's first four positional fields.
    confirmed: bool = False
    status: PhotoCaptureStatus | None = None
    recapture_count: int | None = None
    max_recaptures: int | None = None
    attempt_conflict: bool = False
    backend_status_code: int | None = None
    photo_capture_attempt: int | None = None


def valid_photo_capture_attempt(value: object) -> bool:
    return type(value) is int and 0 <= value <= 3


def valid_measurement_session_id(value: object) -> bool:
    return type(value) is int and 0 < value <= 9_223_372_036_854_775_807


def valid_status_authorization(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.isascii()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _safe_request_id(value: str) -> str:
    # Match the existing failure notifier's metadata sanitizer without importing
    # that notifier: it also consumes this module's conflict helper.
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_.:-")[:80] or "unknown"


def is_photo_capture_attempt_conflict(value: object, status_code: int | None = None) -> bool:
    """Recognize only HTTP 409 or the backend's explicit stale-attempt code.

    Accept a response, parsed JSON, or an already-sanitized status summary.
    Never infer a conflict by searching free-form messages or exception text.
    """
    if type(status_code) is int and status_code == 409:
        return True
    if isinstance(value, httpx.Response):
        if value.status_code == 409:
            return True
        try:
            value = value.json()
        except Exception:
            return False
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            if id(current) in seen:
                continue
            seen.add(id(current))
            for key, item in current.items():
                if key in ("code", "errorCode") and isinstance(item, str) and item == "MEASUREMENT4009":
                    return True
                if key == "statusCode" and type(item) is int and item == 409:
                    return True
                if isinstance(item, (Mapping, list, tuple)):
                    pending.append(item)
        elif isinstance(current, (list, tuple)):
            if id(current) in seen:
                continue
            seen.add(id(current))
            pending.extend(current)
    return False


def parse_photo_capture_status_response(
    response: httpx.Response,
    *,
    measurement_session_id: int,
    photo_capture_attempt: int,
    expected_statuses: frozenset[str],
) -> PhotoCaptureStatusOutcome:
    status_code = response.status_code
    unknown = PhotoCaptureStatusOutcome(backend_status_code=status_code)
    try:
        body = response.json()
    except Exception:
        body = None
    if is_photo_capture_attempt_conflict(body, status_code=status_code):
        return PhotoCaptureStatusOutcome(attempt_conflict=True, backend_status_code=status_code)
    if not 200 <= status_code < 300 or not isinstance(body, dict):
        return unknown
    state = body.get("result", body)
    if not isinstance(state, dict):
        return unknown
    status = state.get("status")
    if not isinstance(status, str) or status not in expected_statuses:
        return unknown
    if status == "FAILED" and state.get("failureReason") != "INVALID_CAPTURE_DATA":
        return unknown

    echoed_attempt = None
    # Check both wrapper and result: a contradictory explicit echo must never
    # be hidden by an otherwise successful-looking result status.
    for fields in (body, state):
        if "isSuccess" in fields and fields["isSuccess"] is not True:
            return unknown
        if "measurementSessionId" in fields:
            session_id = fields["measurementSessionId"]
            if not valid_measurement_session_id(session_id) or session_id != measurement_session_id:
                return unknown
        if "photoCaptureAttempt" in fields:
            attempt = fields["photoCaptureAttempt"]
            if not valid_photo_capture_attempt(attempt) or attempt != photo_capture_attempt:
                return unknown
            echoed_attempt = attempt
    count = state.get("photoRecaptureCount")
    maximum = state.get("maxPhotoRecaptures")
    for value in (count, maximum):
        if value is not None and (type(value) is not int or not 0 <= value <= 2_147_483_647):
            return unknown
    if count is not None and maximum is not None and count > maximum:
        return unknown
    return PhotoCaptureStatusOutcome(
        confirmed=True, status=status, recapture_count=count, max_recaptures=maximum,
        backend_status_code=status_code, photo_capture_attempt=echoed_attempt,
    )


async def notify_photo_capture_passed(
    *,
    measurement_session_id: int,
    authorization: str,
    photo_capture_attempt: int,
    request_id: str,
) -> PhotoCaptureStatusOutcome:
    """Confirm one capture attempt passed without retrying or replacing it."""
    if (
        not valid_measurement_session_id(measurement_session_id)
        or not valid_photo_capture_attempt(photo_capture_attempt)
        or not valid_status_authorization(authorization)
        or not isinstance(request_id, str)
        or not request_id.strip()
    ):
        logger.warning("Photo capture passed callback skipped. code=INVALID_CALLBACK_INPUT")
        return PhotoCaptureStatusOutcome()
    try:
        endpoint = settings.measurement_status_endpoint_template.format(
            measurement_session_id=measurement_session_id,
        )
        async with httpx.AsyncClient(
            timeout=settings.measurement_failure_notification_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.patch(
                endpoint,
                headers={"Authorization": authorization},
                params={
                    "status": "WAITING_FOR_ENVIRONMENT",
                    "photoCaptureAttempt": photo_capture_attempt,
                    "requestId": _safe_request_id(request_id),
                },
            )
        outcome = parse_photo_capture_status_response(
            response,
            measurement_session_id=measurement_session_id,
            photo_capture_attempt=photo_capture_attempt,
            expected_statuses=frozenset({"WAITING_FOR_ENVIRONMENT"}),
        )
    except Exception:
        logger.warning(
            "Photo capture passed callback could not be confirmed. measurement_session_id=%s code=CALLBACK_ERROR",
            measurement_session_id,
        )
        return PhotoCaptureStatusOutcome()
    if not outcome.confirmed:
        logger.warning(
            "Photo capture passed callback was not confirmed. measurement_session_id=%s status_code=%s code=%s",
            measurement_session_id, outcome.backend_status_code,
            "MEASUREMENT4009" if outcome.attempt_conflict else "UNCONFIRMED_STATE",
        )
        return outcome
    logger.info(
        "Photo capture passed callback confirmed. measurement_session_id=%s status=WAITING_FOR_ENVIRONMENT",
        measurement_session_id,
    )
    return outcome
