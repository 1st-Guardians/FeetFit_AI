from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from app.api.routes import pressure_heatmaps as route
from app.main import app
from app.services.pressure_measurement_status import PressureStatusOutcome
from app.services.pressure_heatmap_analysis import (
    PressureHeatmapError,
    SavedPressureHeatmaps,
)
from app.services.report_text_generation import PressureBalanceReportText


ENDPOINT = "/api/reports/daily-foot-analysis/pressure-heatmap"


def _payload(
    *,
    measurement_session_id: int = 81,
    left: list[float] | None = None,
    right: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "measurementSessionId": measurement_session_id,
        "pressureMeasurementAttempt": 0,
        "leftPressureValues": left if left is not None else list(range(1, 13)),
        "rightPressureValues": right if right is not None else list(range(13, 25)),
    }


def _png_bytes() -> bytes:
    image = np.zeros((5, 5, 4), dtype=np.uint8)
    image[1:-1, 1:-1, :3] = (30, 120, 220)
    image[1:-1, 1:-1, 3] = 255
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not encode pressure callback test PNG.")
    return encoded.tobytes()


def _saved_heatmaps(output_dir: Path) -> SavedPressureHeatmaps:
    content = _png_bytes()
    return SavedPressureHeatmaps(
        left_path=output_dir / "left.png",
        right_path=output_dir / "right.png",
        left_png=content,
        right_png=content,
    )


REPORT_TEXT = PressureBalanceReportText(
    score=88.0,
    left_regional_observation="왼발 압력은 앞꿈치와 뒤꿈치에 비교적 고르게 나타납니다.",
    right_regional_observation="오른발 압력은 앞꿈치 쪽에 조금 더 나타납니다.",
    overall_balance_observation="좌우 하중은 전반적으로 안정적인 범위입니다.",
    practical_tip="양발에 체중을 고르게 싣는 자세를 유지해주세요.",
)


class _Backend:
    def __init__(self, outcomes: dict[str, int | BaseException] | None = None) -> None:
        self.outcomes = outcomes or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    @staticmethod
    def _key(url: str) -> str:
        if url == route.settings.pressure_heatmap_report_endpoint:
            return "heatmap"
        if url == route.settings.pressure_distribution_report_endpoint:
            return "distribution"
        if url == route.settings.pressure_balance_report_endpoint:
            return "balance"
        if url == route.settings.summary_report_endpoint:
            return "summary"
        raise AssertionError(f"Unexpected report URL: {url}")

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        key = self._key(url)
        outcome = self.outcomes.get(key, 200)
        if isinstance(outcome, BaseException):
            raise outcome
        return httpx.Response(int(outcome), json={"report": key})


class PressureFailureCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.output_dir = Path("route-test-output") / "pressure-failure-callback"
        route._active_pressure_analysis_sessions.clear()
        registry_path = Path(".build") / f"pressure-attempt-failure-test-{uuid4().hex}.sqlite3"
        self.addCleanup(registry_path.unlink, missing_ok=True)
        registry_patcher = patch.object(route, "_pressure_attempt_registry_path", return_value=registry_path)
        registry_patcher.start()
        self.addCleanup(registry_patcher.stop)
        for name, target in (("notify_pressure_measurement_received", "MEASURING_PRESSURE"),
                             ("notify_pressure_analysis_started", "ANALYZING"),
                             ("notify_pressure_retry_required", "WAITING_FOR_PRESSURE_RETRY")):
            patcher = patch.object(route, name, new=AsyncMock(return_value=PressureStatusOutcome(confirmed=True, status=target)))
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        route._active_pressure_analysis_sessions.clear()
        self.client.close()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        request = route.PressureHeatmapRequest.model_validate(payload)
        request_id, output_dir = route._request_dir()
        work_item = route.PressureAnalysisWorkItem(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            output_dir=output_dir,
            left_pressure_values=tuple(request.left_pressure_values),
            right_pressure_values=tuple(request.right_pressure_values),
            authorization="Bearer pressure-callback-token",
            pressure_measurement_attempt=request.pressure_measurement_attempt,
        )
        try:
            response = asyncio.run(route._execute_pressure_pipeline(work_item))
        except route.HTTPException as exc:
            return httpx.Response(
                exc.status_code,
                json={"detail": exc.detail},
                headers=exc.headers,
            )
        return httpx.Response(
            response.status_code,
            content=response.body,
            headers=dict(response.headers),
        )

    def _intake_post(self, payload: dict[str, Any]) -> httpx.Response:
        return self.client.post(
            ENDPOINT,
            json=payload,
            headers={"Authorization": "Bearer pressure-callback-token"},
        )

    def _run(
        self,
        *,
        payload: dict[str, Any] | None = None,
        outcomes: dict[str, int | BaseException] | None = None,
        notifier: AsyncMock | None = None,
        render_side_effect: BaseException | None = None,
        diagnostics_mkdir_side_effect: BaseException | None = None,
        gpt_side_effect: BaseException | None = None,
        fact_side_effect: BaseException | None = None,
    ) -> tuple[httpx.Response, AsyncMock]:
        callback = notifier or AsyncMock(return_value=True)
        render_kwargs: dict[str, Any]
        if render_side_effect is None:
            render_kwargs = {"return_value": _saved_heatmaps(self.output_dir)}
        else:
            render_kwargs = {"side_effect": render_side_effect}
        generate = AsyncMock(
            side_effect=gpt_side_effect,
            return_value=None if gpt_side_effect is not None else REPORT_TEXT,
        )

        common_contexts = (
            patch.object(
                route,
                "_request_dir",
                return_value=("pressure-callback-request", self.output_dir),
            ),
            patch.object(route, "create_pressure_heatmaps", **render_kwargs),
            patch.object(route, "generate_pressure_balance_report", new=generate),
            patch.object(route.httpx, "AsyncClient", return_value=_Backend(outcomes)),
            patch.object(route, "notify_measurement_failed", new=callback),
        )
        with ExitStack() as stack:
            for context in common_contexts:
                stack.enter_context(context)
            if fact_side_effect is not None:
                stack.enter_context(
                    patch.object(
                        route,
                        "build_pressure_balance_facts",
                        side_effect=fact_side_effect,
                    )
                )
            if render_side_effect is not None:
                stack.enter_context(
                    patch.object(
                        Path,
                        "mkdir",
                        autospec=True,
                        side_effect=diagnostics_mkdir_side_effect,
                    )
                )
                stack.enter_context(patch.object(Path, "write_text", autospec=True))
            response = self._post(payload or _payload())
        return response, callback

    def assert_callback(
        self,
        callback: AsyncMock,
        *,
        reason: str,
        stage: str,
        error_type: str,
    ) -> None:
        callback.assert_awaited_once()
        kwargs = callback.await_args.kwargs
        self.assertEqual(kwargs["measurement_session_id"], 81)
        self.assertEqual(kwargs["failure_reason"], reason)
        self.assertEqual(kwargs["pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(kwargs["stage"], stage)
        self.assertEqual(kwargs["error_type"], error_type)
        self.assertEqual(kwargs["request_id"], "pressure-callback-request")
        self.assertEqual(kwargs["authorization"], "Bearer pressure-callback-token")

    def test_success_does_not_send_failed_callback(self) -> None:
        response, callback = self._run()

        self.assertEqual(response.status_code, 200, response.text)
        callback.assert_not_awaited()

    def test_sensor_schema_failure_requests_remeasurement_without_failed_callback(self) -> None:
        callback = AsyncMock(return_value=True)
        with patch.object(route, "notify_measurement_failed", new=callback):
            response = self._intake_post(_payload(left=[1.0] * 11))

        self.assertEqual(response.status_code, 422, response.text)
        callback.assert_not_awaited()
        self.notify_pressure_retry_required.assert_awaited_once()

    def test_handler_sensor_fact_error_requests_remeasurement_without_failed_callback(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback failed"))
        response, callback = self._run(
            notifier=callback,
            fact_side_effect=ValueError("raw sensor secret must not be forwarded"),
        )

        self.assertEqual(response.status_code, 422, response.text)
        callback.assert_not_awaited()
        self.notify_pressure_retry_required.assert_awaited_once()
        self.assertNotIn("raw sensor secret", str(self.notify_pressure_retry_required.await_args.kwargs))

    def test_no_load_and_single_side_signal_request_remeasurement(self) -> None:
        cases = (
            ("NoLoadPressureSignal", [0.0] * 12, [0.0] * 12),
            ("SingleSidePressureSignal", [1.0] * 12, [0.0] * 12),
        )
        for error_type, left, right in cases:
            with self.subTest(error_type=error_type):
                response, callback = self._run(payload=_payload(left=left, right=right))

                self.assertEqual(response.status_code, 422, response.text)
                callback.assert_not_awaited()
                self.assertEqual(self.notify_pressure_retry_required.await_args.kwargs["error_type"], error_type)

    def test_local_render_gpt_and_internal_errors_are_ai_server_errors(self) -> None:
        cases = (
            (
                "pressure-balance-analysis",
                "RuntimeError",
                {"fact_side_effect": RuntimeError("private fact detail")},
            ),
            (
                "pressure-heatmap-render",
                "PressureHeatmapError",
                {"render_side_effect": PressureHeatmapError("private render detail")},
            ),
            (
                "pressure-comment-generation",
                "RuntimeError",
                {"gpt_side_effect": RuntimeError("private GPT detail")},
            ),
        )
        for stage, error_type, kwargs in cases:
            with self.subTest(stage=stage):
                callback = AsyncMock(side_effect=RuntimeError("callback unavailable"))
                response, callback = self._run(notifier=callback, **kwargs)

                self.assertEqual(response.status_code, 500, response.text)
                self.assert_callback(
                    callback,
                    reason="AI_SERVER_ERROR",
                    stage=stage,
                    error_type=error_type,
                )
                self.assertNotIn("private", str(callback.await_args.kwargs))

    def test_backend_transport_error_is_network_error_and_keeps_502(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback unavailable"))
        response, callback = self._run(
            notifier=callback,
            outcomes={"heatmap": httpx.ConnectError("private network detail")},
        )

        self.assertEqual(response.status_code, 502, response.text)
        self.assert_callback(
            callback,
            reason="NETWORK_ERROR",
            stage="pressure-heatmap-backend-forward",
            error_type="ConnectError",
        )

    def test_diagnostic_directory_failure_keeps_structured_render_error_and_callback(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback unavailable"))
        response, callback = self._run(
            notifier=callback,
            render_side_effect=PressureHeatmapError("original render failure"),
            diagnostics_mkdir_side_effect=OSError("diagnostic filesystem unavailable"),
        )

        self.assertEqual(response.status_code, 500, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(detail["stage"], "pressure-heatmap-render")
        self.assertEqual(detail["errorType"], "PressureHeatmapError")
        self.assertEqual(
            detail["message"],
            "Validated pressure values could not be rendered.",
        )
        self.assert_callback(
            callback,
            reason="AI_SERVER_ERROR",
            stage="pressure-heatmap-render",
            error_type="PressureHeatmapError",
        )

    def test_concurrent_401_takes_callback_priority_over_transport_failure(self) -> None:
        response, callback = self._run(
            outcomes={
                "heatmap": httpx.ConnectError("private network detail"),
                "distribution": 401,
            },
        )

        # Keep the original transport-facing response while reporting the
        # actionable authentication failure through the terminal callback.
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(
            response.json()["detail"]["stage"],
            "pressure-heatmap-backend-forward",
        )
        self.assert_callback(
            callback,
            reason="TOKEN_EXPIRED",
            stage="pressure-distribution-backend-forward",
            error_type="UpstreamUnauthorized",
        )

    def test_backend_401_is_token_expired(self) -> None:
        response, callback = self._run(outcomes={"heatmap": 401})

        self.assertEqual(response.status_code, 207, response.text)
        self.assert_callback(
            callback,
            reason="TOKEN_EXPIRED",
            stage="pressure-heatmap-backend-forward",
            error_type="UpstreamUnauthorized",
        )

    def test_other_backend_http_failure_is_report_save_error(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback unavailable"))
        response, callback = self._run(
            notifier=callback,
            outcomes={"summary": 503},
        )

        self.assertEqual(response.status_code, 207, response.text)
        self.assertTrue(response.json()["partialSave"])
        self.assert_callback(
            callback,
            reason="REPORT_SAVE_ERROR",
            stage="pressure-summary-backend-forward",
            error_type="UpstreamHTTPError",
        )


if __name__ == "__main__":
    unittest.main()
