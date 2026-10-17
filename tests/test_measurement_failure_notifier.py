from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from app.services import measurement_failure_notifier as notifier


class _Backend:
    def __init__(
        self,
        *,
        status_code: int = 204,
        exception: Exception | None = None,
        json_body: object | None = None,
    ) -> None:
        self.status_code = status_code
        self.exception = exception
        self.json_body = json_body
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def patch(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if self.exception is not None:
            raise self.exception
        if self.json_body is not None:
            return httpx.Response(self.status_code, json=self.json_body)
        return httpx.Response(self.status_code)


class MeasurementFailureNotifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_photo_capture_attempt_is_forwarded_unchanged(self) -> None:
        for attempt in range(4):
            with self.subTest(attempt=attempt):
                backend = _Backend()
                with patch.object(notifier.httpx, "AsyncClient", return_value=backend):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer safe-test-token",
                        failure_reason="CAMERA_ERROR",
                        pipeline="full-photo-analysis-v1",
                        stage="capture-preflight",
                        error_type="PhotoCapturePreflightError",
                        request_id="request-123",
                        photo_capture_attempt=attempt,
                    )

                self.assertTrue(result)
                self.assertEqual(len(backend.calls), 1)
                params = backend.calls[0][1]["params"]
                self.assertEqual(params["status"], "FAILED")
                self.assertEqual(params["photoCaptureAttempt"], attempt)
                self.assertIs(type(params["photoCaptureAttempt"]), int)

    async def test_no_photo_attempt_preserves_sensor_callback_contract(self) -> None:
        backend = _Backend()
        with patch.object(notifier.httpx, "AsyncClient", return_value=backend):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=34,
                authorization="Bearer safe-test-token",
                failure_reason="PRESSURE_SENSOR_ERROR",
                pipeline="pressure-heatmap-v1",
                stage="decode",
                error_type="ValueError",
                request_id="request-123",
                photo_capture_attempt=None,
            )

        self.assertTrue(result)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(
            set(backend.calls[0][1]["params"]),
            {"status", "failureReason", "failureDetail"},
        )

    async def test_invalid_photo_attempt_is_rejected_before_http_call(self) -> None:
        for attempt in (-1, 4, 100, True, False, 0.0, 1.0, "0", "3", [], {}):
            with self.subTest(attempt=attempt):
                with (
                    patch.object(notifier.httpx, "AsyncClient") as client_factory,
                    self.assertLogs(notifier.logger, level="WARNING"),
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer safe-test-token",
                        failure_reason="CAMERA_ERROR",
                        pipeline="full-photo-analysis-v1",
                        stage="capture-preflight",
                        error_type="PhotoCapturePreflightError",
                        request_id="request-123",
                        photo_capture_attempt=attempt,
                    )

                self.assertFalse(result)
                client_factory.assert_not_called()

    async def test_capture_conflicts_stop_after_one_patch_without_exposing_body(self) -> None:
        cases = (
            (409, None),
            (409, {"message": "private backend state"}),
            (200, {"code": "MEASUREMENT4009", "message": "private backend state"}),
            (200, {"result": {"errorCode": "MEASUREMENT4009"}}),
        )
        for status_code, body in cases:
            with self.subTest(status_code=status_code, body=body):
                backend = _Backend(status_code=status_code, json_body=body)
                token = "Bearer do-not-log-test-token"
                with (
                    patch.object(notifier.httpx, "AsyncClient", return_value=backend),
                    self.assertLogs(notifier.logger, level="WARNING") as captured,
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization=token,
                        failure_reason="CAMERA_ERROR",
                        pipeline="full-photo-analysis-v1",
                        stage="capture-preflight",
                        error_type="PhotoCapturePreflightError",
                        request_id="request-123",
                        photo_capture_attempt=2,
                    )

                self.assertFalse(result)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(backend.calls[0][1]["params"]["photoCaptureAttempt"], 2)
                logs = "\n".join(captured.output)
                self.assertIn(f"status_code={status_code}", logs)
                self.assertNotIn(token, logs)
                self.assertNotIn("private backend state", logs)

    async def test_every_documented_failure_reason_is_forwarded(self) -> None:
        documented_reasons = (
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
        )

        for failure_reason in documented_reasons:
            with self.subTest(failure_reason=failure_reason):
                backend = _Backend()
                with patch.object(
                    notifier.httpx,
                    "AsyncClient",
                    return_value=backend,
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer safe-test-token",
                        failure_reason=failure_reason,
                        pipeline="full-photo-analysis-v1",
                        stage="test-stage",
                        error_type="TestError",
                        request_id="request-123",
                    )

                self.assertTrue(result)
                self.assertEqual(
                    backend.calls[0][1]["params"]["failureReason"],
                    failure_reason,
                )

    async def test_success_forwards_authorization_and_exact_backend_contract(self) -> None:
        backend = _Backend()
        token = "Bearer secret-access-token"

        with (
            patch.object(
                notifier.httpx,
                "AsyncClient",
                return_value=backend,
            ) as client_factory,
            patch.object(
                notifier.settings,
                "measurement_status_endpoint_template",
                "http://127.0.0.1:8080/api/measurement-sessions/{measurement_session_id}/status",
            ),
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=34,
                authorization=token,
                failure_reason="AI_SERVER_ERROR",
                pipeline="full-photo-analysis-v1",
                stage="gpt-report-text",
                error_type="FullPhotoOutputValidationError",
                request_id="request-123",
            )

        self.assertTrue(result)
        client_factory.assert_called_once_with(timeout=5.0, trust_env=False)
        self.assertEqual(len(backend.calls), 1)
        url, kwargs = backend.calls[0]
        self.assertEqual(
            url,
            "http://127.0.0.1:8080/api/measurement-sessions/34/status",
        )
        self.assertEqual(kwargs["headers"], {"Authorization": token})
        self.assertEqual(
            kwargs["params"],
            {
                "status": "FAILED",
                "failureReason": "AI_SERVER_ERROR",
                "failureDetail": "AI 분석에 실패했습니다.",
            },
        )

    async def test_gpt_failure_codes_forward_short_frontend_detail(self) -> None:
        cases = {
            "GPT_REPORT_RESPONSE_INVALID": "사진 분석 결과 생성에 실패했습니다.",
            "GPT_REPORT_GENERATION_FAILED": "사진 분석 결과 생성에 실패했습니다.",
        }

        for diagnostic_code, expected_detail in cases.items():
            with self.subTest(diagnostic_code=diagnostic_code):
                backend = _Backend()
                with patch.object(
                    notifier.httpx,
                    "AsyncClient",
                    return_value=backend,
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer safe-test-token",
                        failure_reason="AI_SERVER_ERROR",
                        pipeline="full-photo-analysis-v1",
                        stage="gpt-report-text",
                        error_type="RuntimeError",
                        request_id="request-gpt-123",
                        diagnostic_code=diagnostic_code,
                        validation_field=(
                            "metric_advice.HALLUX_VALGUS.observation"
                            if diagnostic_code == "GPT_REPORT_RESPONSE_INVALID"
                            else None
                        ),
                        validation_rule=(
                            "SENTENCE_COUNT_OUT_OF_RANGE"
                            if diagnostic_code == "GPT_REPORT_RESPONSE_INVALID"
                            else None
                        ),
                    )

                self.assertTrue(result)
                self.assertEqual(
                    backend.calls[0][1]["params"],
                    {
                        "status": "FAILED",
                        "failureReason": "AI_SERVER_ERROR",
                        "failureDetail": expected_detail,
                    },
                )

    async def test_detail_is_short_and_excludes_diagnostic_metadata(self) -> None:
        backend = _Backend()
        with patch.object(
            notifier.httpx,
            "AsyncClient",
            return_value=backend,
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=1,
                authorization="Bearer token",
                failure_reason="CAMERA_ERROR",
                pipeline="photo\nAuthorization: stolen",
                stage="decode?token=secret",
                error_type="ValueError(raw database message)",
                request_id="r" * 1000,
            )

        self.assertTrue(result)
        detail = backend.calls[0][1]["params"]["failureDetail"]
        self.assertLessEqual(len(detail), 400)
        self.assertNotIn("\n", detail)
        self.assertNotIn("?", detail)
        self.assertNotIn("(", detail)
        self.assertEqual(detail, "사진 촬영에 실패했습니다.")

    async def test_photo_failure_detail_identifies_capture_in_one_sentence(self) -> None:
        cases = (
            ("RIGHT_PLANTAR_FOOT_NOT_DETECTED", "rightPlantarFootImage", "오른발 발바닥 촬영에 실패했습니다."),
            ("LEFT_PLANTAR_FOOT_NOT_DETECTED", "leftPlantarFootImage", "왼발 발바닥 촬영에 실패했습니다."),
            ("RIGHT_PLANTAR_FOOT_NOT_DETECTED", None, "오른발 발바닥 촬영에 실패했습니다."),
            ("LEFT_PLANTAR_FOOT_NOT_DETECTED", None, "왼발 발바닥 촬영에 실패했습니다."),
            ("PLANTAR_DECODE_FAILED", "rightPlantarFootImage", "오른발 발바닥 촬영에 실패했습니다."),
            ("PLANTAR_CAPTURE_BLANK", "leftPlantarFootImage", "왼발 발바닥 촬영에 실패했습니다."),
            ("PLANTAR_CAPTURE_NEAR_WHITE", "rightPlantarFootImage", "오른발 발바닥 촬영에 실패했습니다."),
            ("PHOTO_UPLOAD_EMPTY", "rightPlantarFootImage", "오른발 발바닥 촬영에 실패했습니다."),
            ("PHOTO_UPLOAD_TOO_LARGE", "leftPlantarFootImage", "왼발 발바닥 촬영에 실패했습니다."),
            ("DORSAL_DECODE_FAILED", "dorsalFootImage", "발등 촬영에 실패했습니다."),
            ("DORSAL_FEET_NOT_DETECTED", "dorsalFootImage", "발등 촬영에 실패했습니다."),
            ("PHOTO_UPLOAD_EMPTY", "dorsalFootImage", "발등 촬영에 실패했습니다."),
        )
        for diagnostic_code, image_field, expected_detail in cases:
            with self.subTest(code=diagnostic_code, image_field=image_field):
                backend = _Backend()
                with patch.object(notifier.httpx, "AsyncClient", return_value=backend):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer test-token",
                        failure_reason="CAMERA_ERROR",
                        pipeline="full-photo-analysis-v1",
                        stage="capture-preflight",
                        error_type="PhotoCapturePreflightError",
                        request_id="request-123",
                        diagnostic_code=diagnostic_code,
                        image_field=image_field,
                        photo_capture_attempt=0,
                    )
                self.assertTrue(result)
                self.assertEqual(backend.calls[0][1]["params"], {
                    "status": "FAILED",
                    "failureReason": "CAMERA_ERROR",
                    "photoCaptureAttempt": 0,
                    "failureDetail": expected_detail,
                })

    async def test_known_photo_diagnostic_forwards_allowlisted_korean_detail(self) -> None:
        backend = _Backend(status_code=200, json_body={"result": {"status": "FAILED"}})
        with patch.object(
            notifier.httpx,
            "AsyncClient",
            return_value=backend,
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=22,
                authorization="Bearer secret-access-token",
                failure_reason="CAMERA_ERROR",
                pipeline="full-photo-analysis-v1",
                stage="capture-preflight",
                error_type="PhotoCapturePreflightError",
                request_id="request-aruco-22",
                diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
                image_field="dorsalFootImage",
            )

        self.assertTrue(result)
        detail = backend.calls[0][1]["params"]["failureDetail"]
        self.assertEqual(detail, "발등 촬영에 실패했습니다.")
        self.assertEqual(backend.calls[0][1]["params"]["diagnosticCode"], "DORSAL_MARKERS_INSUFFICIENT")
        self.assertNotIn("secret-access-token", detail)

    async def test_marker_diagnostic_and_attempt_are_preserved_with_short_detail(self) -> None:
        cases = (
            ("DORSAL_MARKERS_INSUFFICIENT", "REQUIRED_MARKERS_NOT_DETECTED"),
            ("DORSAL_MARKERS_INSUFFICIENT", "MARKER_PROJECTION_UNCERTAIN"),
            ("DORSAL_MARKERS_INSUFFICIENT", "NO_FOOT_OVERLAP"),
            ("DORSAL_MARKERS_INSUFFICIENT", "FOOT_MODEL_INFERENCE_FAILED"),
            ("DORSAL_ARUCO_GEOMETRY_INVALID", "MARKER_PROJECTION_UNCERTAIN"),
            ("DORSAL_MARKER_OCCLUDED_BY_FOOT", "FOOT_OVERLAP_CONFIRMED"),
        )
        for diagnostic_code, reason in cases:
            with self.subTest(code=diagnostic_code, reason=reason):
                backend = _Backend(status_code=200, json_body={"result": {"status": "FAILED"}})
                with (
                    patch.object(notifier.httpx, "AsyncClient", return_value=backend),
                    self.assertLogs(notifier.logger, level="INFO") as captured,
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34,
                        authorization="Bearer private-test-auth",
                        failure_reason="INVALID_CAPTURE_DATA",
                        pipeline="full-photo-analysis-v1",
                        stage="capture-preflight",
                        error_type="PrivateError(private-image.jpg)",
                        request_id="current-request-34",
                        diagnostic_code=diagnostic_code,
                        image_field="dorsalFootImage",
                        marker_occlusion_reason=reason,
                        photo_capture_attempt=2,
                    )
                self.assertTrue(result)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(backend.calls[0][1]["headers"], {"Authorization": "Bearer private-test-auth"})
                params = backend.calls[0][1]["params"]
                self.assertEqual(set(params), {"status", "failureReason", "failureDetail", "diagnosticCode", "photoCaptureAttempt"})
                self.assertEqual(params["status"], "FAILED")
                self.assertEqual(params["failureReason"], "INVALID_CAPTURE_DATA")
                self.assertEqual(params["diagnosticCode"], diagnostic_code)
                self.assertEqual(params["photoCaptureAttempt"], 2)
                self.assertEqual(params["failureDetail"], "발등 촬영에 실패했습니다.")
                self.assertLessEqual(len(params["failureDetail"]), 400)
                self.assertNotIn("private-image", params["failureDetail"])
                self.assertNotIn("private-", "\n".join(captured.output))

    async def test_unknown_marker_reasons_cannot_enter_backend_detail_or_logs(self) -> None:
        for reason in (None, "private-token ../private-image.jpg", "UNKNOWN_REASON",
                       "FOOT_MODEL_INFERENCE_FAILED\nprivate-stack-trace", [], {"private": "value"}):
            with self.subTest(reason_type=type(reason).__name__):
                backend = _Backend(status_code=200, json_body={"result": {"status": "FAILED"}})
                with (
                    patch.object(notifier.httpx, "AsyncClient", return_value=backend),
                    self.assertLogs(notifier.logger, level="INFO") as captured,
                ):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34, authorization="Bearer private-auth",
                        failure_reason="INVALID_CAPTURE_DATA", pipeline="full-photo-analysis-v1",
                        stage="capture-preflight", error_type="PhotoCapturePreflightError",
                        request_id="request-34", diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
                        image_field="dorsalFootImage", marker_occlusion_reason=reason,
                    )
                self.assertTrue(result)
                self.assertEqual(len(backend.calls), 1)
                params = backend.calls[0][1]["params"]
                self.assertEqual(params["diagnosticCode"], "DORSAL_MARKERS_INSUFFICIENT")
                self.assertNotIn("가림 판별 코드:", params["failureDetail"])
                self.assertNotIn("private", params["failureDetail"])
                self.assertNotIn("UNKNOWN_REASON", params["failureDetail"])
                self.assertNotIn("private", "\n".join(captured.output))

    async def test_marker_context_is_not_added_to_unrelated_or_unknown_failure_codes(self) -> None:
        for code in (None, "DORSAL_FEET_NOT_DETECTED", "GPT_REPORT_GENERATION_FAILED",
                     "DORSAL_MARKERS_PRIVATE_UNKNOWN", "private-code", []):
            with self.subTest(code=code):
                backend = _Backend()
                with patch.object(notifier.httpx, "AsyncClient", return_value=backend):
                    result = await notifier.notify_measurement_failed(
                        measurement_session_id=34, authorization="Bearer test",
                        failure_reason="CAMERA_ERROR", pipeline="full-photo-analysis-v1",
                        stage="capture-preflight", error_type="PhotoCapturePreflightError",
                        request_id="request-34", diagnostic_code=code,
                        image_field="dorsalFootImage", marker_occlusion_reason="FOOT_MODEL_INFERENCE_FAILED",
                    )
                self.assertTrue(result)
                params = backend.calls[0][1]["params"]
                self.assertNotIn("diagnosticCode", params)
                self.assertNotIn("FOOT_MODEL_INFERENCE_FAILED", params["failureDetail"])
                self.assertNotIn("가림 판별 코드:", params["failureDetail"])

    async def test_marker_failure_conflict_stops_with_current_diagnostic_and_no_second_patch(self) -> None:
        for status_code, body in (
            (409, None), (200, {"errorCode": "MEASUREMENT4009", "message": "private-server-detail"}),
        ):
            backend = _Backend(status_code=status_code, json_body=body)
            with (
                patch.object(notifier.httpx, "AsyncClient", return_value=backend),
                self.assertLogs(notifier.logger, level="WARNING") as captured,
            ):
                result = await notifier.notify_measurement_failed(
                    measurement_session_id=34, authorization="Bearer private-auth",
                    failure_reason="INVALID_CAPTURE_DATA", pipeline="full-photo-analysis-v1",
                    stage="capture-preflight", error_type="PhotoCapturePreflightError",
                    request_id="request-34", diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
                    image_field="dorsalFootImage", marker_occlusion_reason="REQUIRED_MARKERS_NOT_DETECTED",
                    photo_capture_attempt=1,
                )
            self.assertFalse(result)
            self.assertEqual(len(backend.calls), 1)
            params = backend.calls[0][1]["params"]
            self.assertEqual(params["photoCaptureAttempt"], 1)
            self.assertEqual(params["diagnosticCode"], "DORSAL_MARKERS_INSUFFICIENT")
            self.assertNotIn("private", "\n".join(captured.output))

    async def _notify_marker(self, backend, **overrides):
        values = dict(
            measurement_session_id=34, authorization="Bearer private-auth",
            failure_reason="INVALID_CAPTURE_DATA", pipeline="full-photo-analysis-v1",
            stage="capture-preflight", error_type="PhotoCapturePreflightError",
            request_id="request-34", diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
            image_field="dorsalFootImage", marker_occlusion_reason="REQUIRED_MARKERS_NOT_DETECTED",
            photo_capture_attempt=1,
        )
        values.update(overrides)
        with patch.object(notifier.httpx, "AsyncClient", return_value=backend):
            return await notifier.notify_measurement_failed(**values)

    async def test_marker_failure_confirmation_accepts_actual_failed_state_and_matching_echoes(self):
        for reason in ("INVALID_CAPTURE_DATA", "AI_SERVER_ERROR"):
            for enveloped in (False, True):
                with self.subTest(reason=reason, enveloped=enveloped):
                    state = {"status": "FAILED", "failureReason": reason,
                             "measurementSessionId": 34, "photoCaptureAttempt": 1}
                    body = {"isSuccess": True, "result": state} if enveloped else state
                    backend = _Backend(status_code=200, json_body=body)
                    self.assertTrue(await self._notify_marker(backend, failure_reason=reason))
                    self.assertEqual(len(backend.calls), 1)
                    self.assertEqual(backend.calls[0][1]["params"]["photoCaptureAttempt"], 1)

    async def test_marker_failure_rejects_missing_state_and_contradictory_backend_metadata(self):
        bodies = [
            [], {}, {"result": None}, {"result": []},
            {"result": {}, "status": "FAILED"},
            {"status": "WAITING_FOR_RECAPTURE"},
            {"status": "WAITING_FOR_ENVIRONMENT"},
            {"status": "COMPLETED"},
            {"status": "FAILED", "result": {"status": "WAITING_FOR_RECAPTURE"}},
            {"status": "WAITING_FOR_RECAPTURE", "result": {"status": "FAILED"}},
        ]
        for key, values in (
            ("isSuccess", (False, None, "false", 0)),
            ("measurementSessionId", (35, "34", 34.0, None, True)),
            ("photoCaptureAttempt", (0, 2, "1", 1.0, None, True, 4)),
            ("failureReason", ("CAMERA_ERROR", "AI_SERVER_ERROR", None, True, "private-reason")),
        ):
            for value in values:
                bodies.append({"result": {"status": "FAILED", key: value}})
                bodies.append({key: value, "result": {"status": "FAILED"}})
        for body in bodies:
            with self.subTest(body=body):
                backend = _Backend(status_code=200, json_body=body)
                with self.assertLogs(notifier.logger, level="WARNING") as captured:
                    self.assertFalse(await self._notify_marker(backend))
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(backend.calls[0][1]["params"]["status"], "FAILED")
                self.assertNotIn("private", "\n".join(captured.output))

    async def test_empty_2xx_cannot_confirm_marker_failure_but_legacy_sensor_contract_remains(self):
        for status_code in (200, 204):
            with self.subTest(status_code=status_code):
                backend = _Backend(status_code=status_code)
                with self.assertLogs(notifier.logger, level="WARNING"):
                    self.assertFalse(await self._notify_marker(backend))
                self.assertEqual(len(backend.calls), 1)
                sensor_backend = _Backend(status_code=status_code)
                self.assertTrue(await self._notify_marker(
                    sensor_backend, failure_reason="PRESSURE_SENSOR_ERROR", pipeline="pressure-heatmap-v1",
                    diagnostic_code=None, image_field=None, marker_occlusion_reason=None,
                    photo_capture_attempt=None,
                ))
                self.assertEqual(set(sensor_backend.calls[0][1]["params"]), {"status", "failureReason", "failureDetail"})

    async def test_unknown_photo_diagnostic_uses_short_fallback_detail(self) -> None:
        backend = _Backend()
        with patch.object(
            notifier.httpx,
            "AsyncClient",
            return_value=backend,
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=22,
                authorization="Bearer secret-access-token",
                failure_reason="CAMERA_ERROR",
                pipeline="full-photo-analysis-v1",
                stage="capture-preflight",
                error_type="ValueError",
                request_id="request-22",
                diagnostic_code="Bearer raw-token ../../private.jpg",
                image_field="dorsalFootImage",
            )

        self.assertTrue(result)
        detail = backend.calls[0][1]["params"]["failureDetail"]
        self.assertEqual(detail, "사진 촬영에 실패했습니다.")
        self.assertNotIn("raw-token", detail)

    async def test_non_2xx_is_swallowed_without_logging_token_or_body(self) -> None:
        backend = _Backend(status_code=500)
        token = "Bearer never-log-this-token"
        with (
            patch.object(notifier.httpx, "AsyncClient", return_value=backend),
            self.assertLogs(notifier.logger, level="WARNING") as captured,
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=7,
                authorization=token,
                failure_reason="NETWORK_ERROR",
                pipeline="full-photo-analysis-v1",
                stage="core-save-metrics",
                error_type="UpstreamHTTPError",
                request_id="request-7",
            )

        self.assertFalse(result)
        logs = "\n".join(captured.output)
        self.assertIn("status_code=500", logs)
        self.assertNotIn(token, logs)

    async def test_transport_exception_is_swallowed_and_raw_message_is_not_logged(self) -> None:
        secret_message = "raw transport failure with secret-token"
        backend = _Backend(exception=httpx.ConnectError(secret_message))
        with (
            patch.object(notifier.httpx, "AsyncClient", return_value=backend),
            self.assertLogs(notifier.logger, level="WARNING") as captured,
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=8,
                authorization="Bearer also-secret",
                failure_reason="NETWORK_ERROR",
                pipeline="foot-environment-v1",
                stage="backend-save",
                error_type="ConnectError",
                request_id="request-8",
            )

        self.assertFalse(result)
        logs = "\n".join(captured.output)
        self.assertIn("callback_error_type=ConnectError", logs)
        self.assertNotIn(secret_message, logs)
        self.assertNotIn("also-secret", logs)

    async def test_invalid_failure_reason_is_rejected_before_http_call(self) -> None:
        with (
            patch.object(notifier.httpx, "AsyncClient") as client_factory,
            self.assertLogs(notifier.logger, level="WARNING"),
        ):
            result = await notifier.notify_measurement_failed(
                measurement_session_id=9,
                authorization="Bearer token",
                failure_reason="NOT_A_BACKEND_ENUM",
                pipeline="full-photo-analysis-v1",
                stage="unknown",
                error_type="RuntimeError",
                request_id="request-9",
            )

        self.assertFalse(result)
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
