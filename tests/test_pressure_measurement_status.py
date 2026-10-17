from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from app.services import pressure_measurement_status as service
from app.services import measurement_failure_notifier as failed


class Backend:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def patch(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


class PressureStatusTests(unittest.IsolatedAsyncioTestCase):
    def values(self, attempt=0):
        return dict(measurement_session_id=34, authorization="Bearer private-test-token",
                    pressure_measurement_attempt=attempt, request_id="pressure-request")

    async def test_measuring_retry_and_analyzing_forward_the_same_attempt_once(self):
        for attempt in (0, 1, 3, 7):
            for target in ("MEASURING_PRESSURE", "WAITING_FOR_PRESSURE_RETRY", "ANALYZING"):
                with self.subTest(attempt=attempt, target=target):
                    backend = Backend(httpx.Response(200, json={"result": {
                        "status": target, "measurementSessionId": 34,
                        "pressureMeasurementAttempt": attempt,
                    }}))
                    with patch.object(service.httpx, "AsyncClient", return_value=backend) as client:
                        if target == "MEASURING_PRESSURE":
                            outcome = await service.notify_pressure_measurement_received(**self.values(attempt))
                        elif target == "ANALYZING":
                            outcome = await service.notify_pressure_analysis_started(**self.values(attempt))
                        else:
                            outcome = await service.notify_pressure_retry_required(**self.values(attempt), error_type="NoLoadPressureSignal")
                    self.assertTrue(outcome.confirmed)
                    self.assertEqual(outcome.status, target)
                    self.assertEqual(len(backend.calls), 1)
                    params = backend.calls[0][1]["params"]
                    self.assertEqual(params["pressureMeasurementAttempt"], attempt)
                    self.assertEqual(params["status"], target)
                    self.assertNotIn("photoCaptureAttempt", params)
                    if target == "WAITING_FOR_PRESSURE_RETRY":
                        self.assertEqual(params["failureReason"], "INVALID_PRESSURE_DATA")
                    else:
                        self.assertNotIn("failureReason", params)
                    self.assertFalse(client.call_args.kwargs["trust_env"])

    async def test_backend_retry_exhaustion_is_confirmed_without_another_failed_patch(self):
        backend = Backend(httpx.Response(200, json={"result": {
            "status": "FAILED", "failureReason": "INVALID_PRESSURE_DATA",
            "pressureMeasurementAttempt": 3,
        }}))
        with patch.object(service.httpx, "AsyncClient", return_value=backend):
            outcome = await service.notify_pressure_retry_required(**self.values(3), error_type="InvalidPressureData")
        self.assertTrue(outcome.confirmed)
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1]["params"]["status"], "WAITING_FOR_PRESSURE_RETRY")

    async def test_stale_identity_malformed_success_and_conflict_are_never_confirmed(self):
        bodies = [None, {}, {"status": "WAITING_FOR_ENVIRONMENT"},
                  {"status": "FAILED", "failureReason": "PRESSURE_SENSOR_ERROR"},
                  {"status": "ANALYZING", "pressureMeasurementAttempt": 0},
                  {"status": "ANALYZING", "pressureMeasurementAttempt": True},
                  {"status": "ANALYZING", "measurementSessionId": 35},
                  {"isSuccess": False, "result": {"status": "ANALYZING"}},
                  {"pressureMeasurementAttempt": 0, "result": {"status": "ANALYZING", "pressureMeasurementAttempt": 1}},
                  {"status": "FAILED", "result": {"status": "ANALYZING"}}]
        for body in bodies:
            with self.subTest(body=body):
                backend = Backend(httpx.Response(200, json=body))
                with patch.object(service.httpx, "AsyncClient", return_value=backend):
                    outcome = await service.notify_pressure_analysis_started(**self.values(1))
                self.assertFalse(outcome.confirmed)
                self.assertEqual(len(backend.calls), 1)
        backend = Backend(httpx.Response(409, content=b"private-conflict-body"))
        with patch.object(service.httpx, "AsyncClient", return_value=backend):
            outcome = await service.notify_pressure_analysis_started(**self.values(1))
        self.assertTrue(outcome.attempt_conflict)
        self.assertFalse(outcome.confirmed)

    async def test_failed_callback_and_invalid_attempt_do_not_retry_or_expose_secrets(self):
        for error in (httpx.ReadTimeout("private-provider-error"), RuntimeError("private-provider-error")):
            backend = Backend(error=error)
            with patch.object(service.httpx, "AsyncClient", return_value=backend), self.assertLogs(service.logger, level="WARNING") as logs:
                outcome = await service.notify_pressure_retry_required(**self.values(), error_type="InvalidPressureData")
            self.assertFalse(outcome.confirmed)
            self.assertEqual(len(backend.calls), 1)
            self.assertNotIn("private", "\n".join(logs.output))
        with patch.object(service.httpx, "AsyncClient") as client:
            for attempt in (-1, True, "1", 1.0, None, 2**31):
                outcome = await service.notify_pressure_analysis_started(**self.values(attempt))
                self.assertFalse(outcome.confirmed)
            client.assert_not_called()

    async def test_terminal_failure_keeps_pressure_attempt_and_uppercase_reason(self):
        backend = Backend(httpx.Response(204))
        with patch.object(failed.httpx, "AsyncClient", return_value=backend):
            confirmed = await failed.notify_measurement_failed(
                measurement_session_id=34, authorization="Bearer test", failure_reason="INVALID_PRESSURE_DATA",
                pipeline="pressure-heatmap-v3", stage="pressure-pipeline", error_type="TestFailure",
                request_id="request-34", pressure_measurement_attempt=2,
            )
        self.assertTrue(confirmed)
        params = backend.calls[0][1]["params"]
        self.assertEqual(params["status"], "FAILED")
        self.assertEqual(params["failureReason"], "INVALID_PRESSURE_DATA")
        self.assertEqual(params["pressureMeasurementAttempt"], 2)
        self.assertNotIn("photoCaptureAttempt", params)


if __name__ == "__main__":
    unittest.main()
