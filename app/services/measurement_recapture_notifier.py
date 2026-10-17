"""Request a recoverable capture state and trust only the backend's persisted reply.

The backend owns the retake counter and terminal transition. This callback
neither increments a local counter nor falls back to sending FAILED.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings
from app.services.measurement_failure_notifier import _safe_detail_value
from app.services.photo_failure_guidance import photo_failure_guidance
from app.services.photo_capture_status import (
    PhotoCaptureStatusOutcome,
    parse_photo_capture_status_response,
    valid_measurement_session_id,
    valid_photo_capture_attempt,
    valid_status_authorization,
)


logger = logging.getLogger(__name__)

ARUCO_RECAPTURE_DIAGNOSTIC_CODES = frozenset({
    "DORSAL_MARKER_OCCLUDED_BY_FOOT",
})
MeasurementRecaptureOutcome = PhotoCaptureStatusOutcome


async def notify_measurement_recapture_required(
    *,
    measurement_session_id: int,
    authorization: str,
    diagnostic_code: str,
    photo_capture_attempt: int,
    request_id: str,
) -> MeasurementRecaptureOutcome:
    """Send one WAITING_FOR_RECAPTURE request without inferring persisted state.

    A successful HTTP code alone is insufficient: a recognized backend status
    is required. Transport failures and malformed replies return an unconfirmed
    outcome so callers cannot claim that the waiting state was saved.
    """
    if not valid_measurement_session_id(measurement_session_id):
        logger.warning("Measurement recapture callback skipped. code=INVALID_SESSION_ID")
        return MeasurementRecaptureOutcome()
    if not valid_photo_capture_attempt(photo_capture_attempt):
        logger.warning("Measurement recapture callback skipped. code=INVALID_PHOTO_CAPTURE_ATTEMPT")
        return MeasurementRecaptureOutcome()
    if not isinstance(diagnostic_code, str) or diagnostic_code not in ARUCO_RECAPTURE_DIAGNOSTIC_CODES:
        logger.warning("Measurement recapture callback skipped. code=INVALID_DIAGNOSTIC_CODE")
        return MeasurementRecaptureOutcome()
    if not valid_status_authorization(authorization):
        logger.warning("Measurement recapture callback skipped. code=INVALID_AUTHORIZATION")
        return MeasurementRecaptureOutcome()
    if not isinstance(request_id, str) or not request_id.strip():
        logger.warning("Measurement recapture callback skipped. code=INVALID_REQUEST_ID")
        return MeasurementRecaptureOutcome()

    try:
        guidance = photo_failure_guidance(diagnostic_code, "dorsalFootImage")
        if guidance is None:
            logger.warning("Measurement recapture callback skipped. code=GUIDANCE_UNAVAILABLE")
            return MeasurementRecaptureOutcome()
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
                    "status": "WAITING_FOR_RECAPTURE",
                    "diagnosticCode": diagnostic_code,
                    "photoCaptureAttempt": photo_capture_attempt,
                    "failureDetail": guidance.failure_detail,
                    "requestId": _safe_detail_value(request_id),
                },
            )
        outcome = parse_photo_capture_status_response(
            response,
            measurement_session_id=measurement_session_id,
            photo_capture_attempt=photo_capture_attempt,
            expected_statuses=frozenset({"WAITING_FOR_RECAPTURE", "FAILED"}),
        )
    except Exception:
        # Never log exception text, response bodies, request headers, or URLs.
        logger.warning(
            "Measurement recapture callback could not be confirmed. measurement_session_id=%s code=CALLBACK_ERROR",
            measurement_session_id,
        )
        return MeasurementRecaptureOutcome()

    if not outcome.confirmed:
        logger.warning(
            "Measurement recapture callback returned an invalid state. measurement_session_id=%s status_code=%s code=%s",
            measurement_session_id, outcome.backend_status_code,
            "MEASUREMENT4009" if outcome.attempt_conflict else "INVALID_RESPONSE",
        )
        return outcome
    logger.info(
        "Measurement recapture callback confirmed. measurement_session_id=%s status=%s",
        measurement_session_id, outcome.status,
    )
    return outcome
