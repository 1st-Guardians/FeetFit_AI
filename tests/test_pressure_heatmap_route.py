from __future__ import annotations

import asyncio
import json
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
from app.services.report_text_generation import (
    PressureBalanceReportText,
    build_pressure_balance_facts as real_build_pressure_balance_facts,
)


ENDPOINT = "/api/reports/daily-foot-analysis/pressure-heatmap"
LEFT_REGIONAL_OBSERVATION = "왼발은 세 구역 중 뒤꿈치 하중 비중이 상대적으로 크게 관찰됩니다."
RIGHT_REGIONAL_OBSERVATION = "오른발은 세 구역 중 앞꿈치 하중 비중이 상대적으로 크게 관찰됩니다."
OVERALL_BALANCE_OBSERVATION = "좌우 총하중은 오른발 쪽으로 이동한 경향이 관찰됩니다."
PRACTICAL_TIP = "양발에 힘을 고르게 싣는 자세를 천천히 연습해 보세요."
REPORT_ADVICE = (
    "\n".join(
        (
            LEFT_REGIONAL_OBSERVATION,
            RIGHT_REGIONAL_OBSERVATION,
            OVERALL_BALANCE_OBSERVATION,
        )
    ),
    PRACTICAL_TIP,
)
BALANCE_COMMENT = "\n".join(
    (
        LEFT_REGIONAL_OBSERVATION,
        RIGHT_REGIONAL_OBSERVATION,
        OVERALL_BALANCE_OBSERVATION,
        PRACTICAL_TIP,
    )
)


def _png_bytes(bgr: tuple[int, int, int]) -> bytes:
    image = np.zeros((9, 7, 4), dtype=np.uint8)
    image[..., :3] = bgr
    image[1:-1, 1:-1, 3] = 255
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not create the PNG route fixture.")
    return encoded.tobytes()


def _saved_heatmaps(output_dir: Path) -> SavedPressureHeatmaps:
    return SavedPressureHeatmaps(
        left_path=output_dir / "left_pressure_heatmap.png",
        right_path=output_dir / "right_pressure_heatmap.png",
        left_png=_png_bytes((20, 80, 220)),
        right_png=_png_bytes((180, 90, 30)),
    )


def _valid_payload(measurement_session_id: int = 1) -> dict[str, Any]:
    return {
        "measurementSessionId": measurement_session_id,
        "pressureMeasurementAttempt": 0,
        "leftPressureValues": list(range(12)),
        "rightPressureValues": list(range(12, 24)),
    }


class _Backend:
    def __init__(
        self,
        *,
        outcomes: dict[str, int | BaseException] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.events = events if events is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    @staticmethod
    def key_for(url: str) -> str:
        if url == route.settings.pressure_heatmap_report_endpoint:
            return "heatmap"
        if url == route.settings.pressure_distribution_report_endpoint:
            return "distribution"
        if url == route.settings.pressure_balance_report_endpoint:
            return "balance"
        if url == route.settings.summary_report_endpoint:
            return "summary"
        raise AssertionError(f"Unexpected backend URL: {url}")

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        key = self.key_for(url)
        if key in {"balance", "summary"} and "gpt" not in self.events:
            raise AssertionError("downstream write started before GPT generation")
        self.calls.append((url, kwargs))
        self.events.append(key)
        outcome = self.outcomes.get(key, 200)
        if isinstance(outcome, BaseException):
            raise outcome
        if 200 <= outcome < 300:
            body: dict[str, Any] = {"saved": key}
        else:
            body = {
                "failed": key,
                "authorization": "Bearer route-test-token",
                "api_token": "route-test-token",
            }
        return httpx.Response(
            outcome,
            json=body,
            headers={"content-type": "application/json"},
        )


class PressureHeatmapRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        route._active_pressure_analysis_sessions.clear()
        self.pressure_attempt_registry_path = Path(".build") / f"pressure-attempt-test-{uuid4().hex}.sqlite3"
        self.addCleanup(self.pressure_attempt_registry_path.unlink, missing_ok=True)
        registry_patcher = patch.object(route, "_pressure_attempt_registry_path", return_value=self.pressure_attempt_registry_path)
        registry_patcher.start()
        self.addCleanup(registry_patcher.stop)
        # Failure callback behavior has its own focused contract suite.  Keep
        # these route tests isolated from the notifier's separate HTTP client.
        self.failure_notifier_patcher = patch.object(
            route,
            "notify_measurement_failed",
            new=AsyncMock(return_value=True),
        )
        self.failure_notifier_patcher.start()
        for name, target in (("notify_pressure_measurement_received", "MEASURING_PRESSURE"),
                             ("notify_pressure_analysis_started", "ANALYZING"),
                             ("notify_pressure_retry_required", "WAITING_FOR_PRESSURE_RETRY")):
            patcher = patch.object(route, name, new=AsyncMock(return_value=PressureStatusOutcome(confirmed=True, status=target)))
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.failure_notifier_patcher.stop()
        route._active_pressure_analysis_sessions.clear()
        self.client.close()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Execute the background pipeline directly for focused result tests."""

        request = route.PressureHeatmapRequest.model_validate(payload)
        request_id, output_dir = route._request_dir()
        work_item = route.PressureAnalysisWorkItem(
            request_id=request_id,
            measurement_session_id=request.measurement_session_id,
            output_dir=output_dir,
            left_pressure_values=tuple(request.left_pressure_values),
            right_pressure_values=tuple(request.right_pressure_values),
            authorization="Bearer route-test-token",
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
            headers={"Authorization": "Bearer route-test-token"},
        )

    def _success_patches(
        self,
        backend: _Backend,
        *,
        output_name: str = "pressure-success",
        generation_source: str = "OPENAI",
        generation_failure_code: str | None = None,
    ):
        output_dir = Path("route-test-output") / output_name
        saved = _saved_heatmaps(output_dir)
        left_values = [float(value) for value in range(12)]
        right_values = [float(value) for value in range(12, 24)]
        facts = real_build_pressure_balance_facts(left_values, right_values)

        async def generate(**kwargs):
            self.assertEqual(backend.events, ["heatmap", "distribution"])
            backend.events.append("gpt")
            return PressureBalanceReportText(
                # The route must ignore this spoofed value for backend scores.
                score=0.0,
                left_regional_observation=LEFT_REGIONAL_OBSERVATION,
                right_regional_observation=RIGHT_REGIONAL_OBSERVATION,
                overall_balance_observation=OVERALL_BALANCE_OBSERVATION,
                practical_tip=PRACTICAL_TIP,
                generation_source=generation_source,
                generation_failure_code=generation_failure_code,
            )

        return (
            patch.object(
                route,
                "_request_dir",
                return_value=("pressure-request", output_dir),
            ),
            patch.object(
                route,
                "build_pressure_balance_facts",
                return_value=facts,
            ),
            patch.object(
                route,
                "create_pressure_heatmaps",
                return_value=saved,
            ),
            patch.object(
                route,
                "generate_pressure_balance_report",
                new=AsyncMock(side_effect=generate),
            ),
            patch.object(route.httpx, "AsyncClient", return_value=backend),
            facts,
            saved,
        )

    @staticmethod
    def _calls_by_key(backend: _Backend) -> dict[str, dict[str, Any]]:
        return {backend.key_for(url): kwargs for url, kwargs in backend.calls}

    @staticmethod
    def _render_progress_logs(log_info) -> list[str]:
        rendered: list[str] = []
        for log_call in log_info.call_args_list:
            if not log_call.args:
                continue
            message = str(log_call.args[0])
            if len(log_call.args) > 1:
                message %= tuple(log_call.args[1:])
            rendered.append(message)
        return rendered

    @classmethod
    def _progress_payloads(cls, log_info) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        marker = "pipeline_progress "
        for message in cls._render_progress_logs(log_info):
            if marker not in message:
                continue
            payloads.append(json.loads(message.split(marker, maxsplit=1)[1]))
        return payloads

    def test_hardware_measurement_start_is_not_an_ai_endpoint(self) -> None:
        schema = self.client.get("/openapi.json").json()
        self.assertNotIn("/measurement/pressure/start", schema["paths"])
        with (patch.object(route.BackgroundTasks, "add_task") as jobs,
              patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
            response = self.client.post("/measurement/pressure/start", json={"measurementSessionId": 4},
                                        headers={"Authorization": "Bearer route-test-token"})
        self.assertEqual(response.status_code, 404, response.text)
        jobs.assert_not_called()
        terminal.assert_not_awaited()
        self.notify_pressure_analysis_started.assert_not_awaited()
        self.notify_pressure_retry_required.assert_not_awaited()

    def test_route_and_openapi_publish_strict_v3_contract(self) -> None:
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"][ENDPOINT]["post"]
        self.assertEqual(
            set(operation["requestBody"]["content"]),
            {"application/json"},
        )
        body_schema = operation["requestBody"]["content"]["application/json"]["schema"]
        schema_name = body_schema["$ref"].split("/")[-1]
        request_schema = schema["components"]["schemas"][schema_name]
        properties = request_schema["properties"]
        self.assertEqual(
            set(properties),
            {"measurementSessionId", "pressureMeasurementAttempt", "leftPressureValues", "rightPressureValues"},
        )
        self.assertEqual(set(request_schema["required"]), set(properties))
        self.assertNotIn("default", properties["pressureMeasurementAttempt"])
        self.assertFalse(request_schema["additionalProperties"])
        self.assertEqual(
            request_schema["examples"],
            [
                {
                    "measurementSessionId": 34,
                    "pressureMeasurementAttempt": 0,
                    "leftPressureValues": [
                        92.2,
                        1.0,
                        605.4,
                        8.6,
                        29.3,
                        8430.6,
                        7289.3,
                        34.5,
                        4151.6,
                        14356.8,
                        272.2,
                        15257.0,
                    ],
                    "rightPressureValues": [
                        546.9,
                        752.2,
                        683.4,
                        101.4,
                        33.0,
                        1982.6,
                        3446.1,
                        64.3,
                        6026.4,
                        16111.5,
                        11520.5,
                        13116.1,
                    ],
                }
            ],
        )
        for side in ("leftPressureValues", "rightPressureValues"):
            self.assertEqual(properties[side]["minItems"], 12)
            self.assertEqual(properties[side]["maxItems"], 12)
        self.assertTrue(
            {"202", "409", "422", "500", "503"}.issubset(
                operation["responses"]
            )
        )
        accepted_schema = operation["responses"]["202"]["content"][
            "application/json"
        ]["schema"]
        self.assertEqual(accepted_schema["properties"]["status"]["enum"], ["QUEUED"])
        self.assertEqual(
            set(operation["responses"]["202"]["headers"]),
            {"X-Feetfit-Pipeline", "X-Feetfit-Request-Id"},
        )
        self.assertEqual(route.PIPELINE_VERSION, "pressure-heatmap-v3")
        self.assertIn("forefoot (C0-C4)", operation["description"])
        self.assertIn("midfoot (C5-C8)", operation["description"])
        self.assertIn("heel (C9-C11)", operation["description"])
        self.assertIn("Outside-outline C0/C1", operation["description"])
        self.assertIn("2%", operation["description"])
        self.assertIn("0.5%", operation["description"])
        self.assertIn("60%", operation["description"])
        self.assertIn("four backend", operation["description"])

    def test_valid_intake_returns_202_with_immutable_owned_work_item(self) -> None:
        output_dir = Path("route-test-output") / "pressure-intake"
        with (
            patch.object(
                route,
                "_request_dir",
                return_value=("pressure-intake-request", output_dir),
            ),
            patch.object(route.settings, "pressure_sensor_input_mode", "fixed"),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
        ):
            response = self._intake_post(
                _valid_payload(measurement_session_id=36)
            )

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json(),
            {
                "accepted": True,
                "status": "QUEUED",
                "measurementSessionId": 36,
                "pressureMeasurementAttempt": 0,
                "pipeline": route.PIPELINE_VERSION,
                "requestId": "pressure-intake-request",
            },
        )
        self.assertEqual(response.headers["X-Feetfit-Pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(
            response.headers["X-Feetfit-Request-Id"],
            "pressure-intake-request",
        )
        add_task.assert_called_once()
        worker, work_item = add_task.call_args.args
        self.assertIs(worker, route._run_pressure_analysis_job)
        self.assertIsInstance(work_item, route.PressureAnalysisWorkItem)
        self.assertEqual(work_item.measurement_session_id, 36)
        self.assertEqual(work_item.output_dir, output_dir)
        self.assertIsInstance(work_item.left_pressure_values, tuple)
        self.assertIsInstance(work_item.right_pressure_values, tuple)
        self.assertEqual(work_item.left_pressure_values, route.LEFT_CORRECTED)
        self.assertEqual(work_item.right_pressure_values, route.RIGHT_CORRECTED)
        self.assertEqual(work_item.authorization, "Bearer route-test-token")
        self.assertEqual(route._active_pressure_analysis_sessions, {36})

    def test_default_hardware_mode_queues_the_validated_request_values(self) -> None:
        payload = _valid_payload(measurement_session_id=37)
        output_dir = Path("route-test-output") / "pressure-hardware-intake"
        default_mode = type(route.settings).model_fields["pressure_sensor_input_mode"].default
        self.assertEqual(default_mode, "hardware")
        with (
            patch.object(
                route,
                "_request_dir",
                return_value=("pressure-hardware-request", output_dir),
            ),
            patch.object(route.settings, "pressure_sensor_input_mode", default_mode),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
        ):
            response = self._intake_post(payload)

        self.assertEqual(response.status_code, 202, response.text)
        add_task.assert_called_once()
        _, work_item = add_task.call_args.args
        self.assertEqual(
            work_item.left_pressure_values,
            tuple(float(value) for value in payload["leftPressureValues"]),
        )
        self.assertEqual(
            work_item.right_pressure_values,
            tuple(float(value) for value in payload["rightPressureValues"]),
        )
        self.assertEqual(route._active_pressure_analysis_sessions, {37})

    def test_duplicate_and_full_pressure_queue_are_rejected(self) -> None:
        with patch.object(route.BackgroundTasks, "add_task"):
            first = self._intake_post(_valid_payload(measurement_session_id=41))
            duplicate = self._intake_post(_valid_payload(measurement_session_id=41))

        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        self.assertEqual(
            duplicate.json()["detail"]["errorType"],
            "DuplicatePressureAnalysisError",
        )

        route._active_pressure_analysis_sessions.clear()
        route._active_pressure_analysis_sessions.update({51, 52, 53})
        full = self._intake_post(_valid_payload(measurement_session_id=54))
        self.assertEqual(full.status_code, 503, full.text)
        self.assertEqual(full.headers["Retry-After"], "5")
        self.assertEqual(
            full.json()["detail"]["errorType"],
            "PressureAnalysisQueueFullError",
        )

    def test_background_schedule_failure_releases_session_and_notifies(self) -> None:
        callback = AsyncMock(return_value=True)
        with (
            patch.object(
                route.BackgroundTasks,
                "add_task",
                side_effect=RuntimeError("scheduler unavailable"),
            ),
            patch.object(route, "notify_measurement_failed", new=callback),
        ):
            response = self._intake_post(
                _valid_payload(measurement_session_id=61)
            )

        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["detail"]["stage"], "background-schedule")
        self.assertNotIn(61, route._active_pressure_analysis_sessions)
        callback.assert_awaited_once()
        self.assertEqual(
            callback.await_args.kwargs["failure_reason"],
            "AI_SERVER_ERROR",
        )

    def test_background_runner_serializes_pressure_jobs_and_releases_sessions(self) -> None:
        active_workers = 0
        maximum_workers = 0
        execution_order: list[tuple[str, int]] = []

        async def execute(work_item: route.PressureAnalysisWorkItem):
            nonlocal active_workers, maximum_workers
            active_workers += 1
            maximum_workers = max(maximum_workers, active_workers)
            execution_order.append(("start", work_item.measurement_session_id))
            await asyncio.sleep(0.01)
            execution_order.append(("end", work_item.measurement_session_id))
            active_workers -= 1
            return route.JSONResponse(status_code=200, content={"saved": True})

        items = tuple(
            route.PressureAnalysisWorkItem(
                request_id=f"pressure-worker-{session_id}",
                measurement_session_id=session_id,
                output_dir=Path("route-test-output") / str(session_id),
                left_pressure_values=tuple(float(value) for value in range(12)),
                right_pressure_values=tuple(float(value) for value in range(12, 24)),
                authorization="Bearer worker-token",
                pressure_measurement_attempt=0,
            )
            for session_id in (71, 72)
        )
        route._active_pressure_analysis_sessions.update({71, 72})

        async def scenario() -> None:
            await asyncio.gather(
                *(route._run_pressure_analysis_job(item) for item in items)
            )

        with (
            patch.object(route, "_execute_pressure_pipeline", side_effect=execute),
            patch.object(
                route,
                "_pressure_analysis_worker_semaphore",
                asyncio.Semaphore(1),
            ),
        ):
            asyncio.run(scenario())

        self.assertEqual(maximum_workers, 1)
        self.assertEqual(
            execution_order,
            [("start", 71), ("end", 71), ("start", 72), ("end", 72)],
        )
        self.assertEqual(route._active_pressure_analysis_sessions, set())

    def test_background_runner_reports_unexpected_failure_and_releases_session(self) -> None:
        work_item = route.PressureAnalysisWorkItem(
            request_id="pressure-worker-failure",
            measurement_session_id=73,
            output_dir=Path("route-test-output") / "73",
            left_pressure_values=tuple(float(value) for value in range(12)),
            right_pressure_values=tuple(float(value) for value in range(12, 24)),
            authorization="Bearer worker-token",
            pressure_measurement_attempt=0,
        )
        callback = AsyncMock(return_value=True)
        route._active_pressure_analysis_sessions.add(73)

        with (
            patch.object(
                route,
                "_execute_pressure_pipeline",
                new=AsyncMock(side_effect=RuntimeError("unexpected secret")),
            ),
            patch.object(route, "notify_measurement_failed", new=callback),
            patch.object(
                route,
                "_pressure_analysis_worker_semaphore",
                asyncio.Semaphore(1),
            ),
        ):
            asyncio.run(route._run_pressure_analysis_job(work_item))

        callback.assert_awaited_once()
        callback_kwargs = callback.await_args.kwargs
        self.assertEqual(callback_kwargs["failure_reason"], "AI_SERVER_ERROR")
        self.assertEqual(callback_kwargs["stage"], "background-worker")
        self.assertEqual(callback_kwargs["error_type"], "RuntimeError")
        self.assertNotIn("unexpected secret", str(callback_kwargs))
        self.assertNotIn(73, route._active_pressure_analysis_sessions)

    def test_fallback_saves_each_report_once_without_changing_server_score(self) -> None:
        backend = _Backend()
        patches = self._success_patches(
            backend, output_name="fallback-success",
            generation_source="DETERMINISTIC_FALLBACK", generation_failure_code="API_TIMEOUT",
        )
        request_dir, builder, renderer, generator, backend_patch, facts, _ = patches
        with (
            request_dir, builder, renderer, generator as generate, backend_patch,
            patch.object(route.settings, "openai_report_fallback_enabled", True),
            patch.object(route.progress_logger, "info") as log_info,
        ):
            response = self._post(_valid_payload())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(backend.events, ["heatmap", "distribution", "gpt", "balance", "summary"])
        self.assertEqual(len(backend.calls), 4)
        generate.assert_awaited_once()
        self.assertEqual(response.json()["analysis"]["score"], facts["balanceScore"])
        self.assertEqual(response.json()["analysis"]["reportTextGeneration"], {
            "source": "DETERMINISTIC_FALLBACK", "model": None, "failureCode": "API_TIMEOUT",
        })
        for url, payload in backend.calls:
            if _Backend.key_for(url) == "balance":
                self.assertEqual(payload["json"]["balanceScore"], facts["balanceScore"])
            elif _Backend.key_for(url) == "summary":
                self.assertEqual(payload["json"]["score"], facts["balanceScore"])
        completed = next(row for row in self._progress_payloads(log_info)
                         if row["stage"] == "pressure-comment-generation" and row["event"] == "completed")
        self.assertEqual(completed["details"]["generationSource"], "DETERMINISTIC_FALLBACK")
        self.assertEqual(completed["details"]["errorCode"], "API_TIMEOUT")
        self.assertNotIn("GPT", completed["message"])

    def test_disabled_fallback_preserves_core_saves_and_skips_downstream(self) -> None:
        backend = _Backend()
        patches = self._success_patches(
            backend, output_name="fallback-disabled", generation_source="DETERMINISTIC_FALLBACK",
        )
        request_dir, builder, renderer, generator, backend_patch, _, _ = patches
        with (
            request_dir, builder, renderer, generator, backend_patch,
            patch.object(route.settings, "openai_report_fallback_enabled", False),
        ):
            response = self._post(_valid_payload())
        self.assertEqual(response.status_code, 500, response.text)
        self.assertTrue(response.json()["detail"]["partialSave"])
        self.assertEqual([_Backend.key_for(url) for url, _ in backend.calls], ["heatmap", "distribution"])

    def test_success_saves_exact_four_backend_contracts(self) -> None:
        backend = _Backend()
        patches = self._success_patches(backend)
        request_dir, builder, renderer, generator, backend_patch, facts, saved = patches
        with (
            request_dir,
            builder as build,
            renderer as create,
            generator as generate,
            backend_patch,
        ):
            response = self._post(_valid_payload(measurement_session_id=37))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            backend.events,
            ["heatmap", "distribution", "gpt", "balance", "summary"],
        )
        left_values = [float(value) for value in range(12)]
        right_values = [float(value) for value in range(12, 24)]
        build.assert_called_once_with(left_values, right_values)
        create.assert_called_once_with(
            left_values,
            right_values,
            Path("route-test-output") / "pressure-success",
        )
        generate.assert_awaited_once_with(
            left_pressure_values=left_values,
            right_pressure_values=right_values,
            pressure_facts=facts,
            left_heatmap_png=saved.left_png,
            right_heatmap_png=saved.right_png,
        )

        self.assertEqual(len(backend.calls), 4)
        calls = self._calls_by_key(backend)
        self.assertEqual(set(calls), {"heatmap", "distribution", "balance", "summary"})
        for _, kwargs in backend.calls:
            self.assertEqual(
                kwargs["headers"]["Authorization"],
                "Bearer route-test-token",
            )

        heatmap_files = calls["heatmap"]["files"]
        self.assertEqual(
            set(heatmap_files),
            {"request", "leftPressureHeatmapImage", "rightPressureHeatmapImage"},
        )
        self.assertEqual(
            json.loads(heatmap_files["request"][1]),
            {
                "measurementSessionId": 37,
                "leftPressureValues": left_values,
                "rightPressureValues": right_values,
            },
        )

        distribution_files = calls["distribution"]["files"]
        for field_name, expected_png in (
            ("leftPressureHeatmapImage", saved.left_png),
            ("rightPressureHeatmapImage", saved.right_png),
        ):
            self.assertEqual(heatmap_files[field_name][1], expected_png)
            self.assertEqual(heatmap_files[field_name][2], "image/png")
        self.assertEqual(
            set(distribution_files),
            {"request", "leftPressureImage", "rightPressureImage"},
        )
        distribution_json = json.loads(distribution_files["request"][1])
        self.assertEqual(
            distribution_json,
            {
                "measurementSessionId": 37,
                "leftPressurePercent": facts["leftLoadPercent"],
                "rightPressurePercent": facts["rightLoadPercent"],
            },
        )
        self.assertEqual(
            distribution_json["leftPressurePercent"]
            + distribution_json["rightPressurePercent"],
            100.0,
        )
        expected_left_percent = round(
            sum(left_values) / (sum(left_values) + sum(right_values)) * 100.0,
            1,
        )
        self.assertEqual(
            distribution_json["leftPressurePercent"],
            expected_left_percent,
        )
        self.assertEqual(
            distribution_json["rightPressurePercent"],
            round(100.0 - expected_left_percent, 1),
        )
        for field_name, expected_png in (
            ("leftPressureImage", saved.left_png),
            ("rightPressureImage", saved.right_png),
        ):
            self.assertEqual(distribution_files[field_name][1], expected_png)
            self.assertEqual(distribution_files[field_name][2], "image/png")
            decoded = cv2.imdecode(
                np.frombuffer(expected_png, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(decoded.shape, (9, 7, 4))
            self.assertEqual(decoded[0, 0, 3], 0)
            self.assertEqual(decoded[4, 3, 3], 255)

        self.assertEqual(
            calls["balance"]["json"],
            {
                "measurementSessionId": 37,
                "balanceScore": facts["balanceScore"],
                "balanceComment": BALANCE_COMMENT,
            },
        )
        self.assertNotEqual(calls["balance"]["json"]["balanceScore"], 0.0)
        self.assertEqual(
            calls["summary"]["json"],
            {
                "measurementSessionId": 37,
                "metricType": "PRESSURE_BALANCE",
                "score": facts["balanceScore"],
                "advice": list(REPORT_ADVICE),
            },
        )

        body = response.json()
        self.assertEqual(body["analysis"]["pressureFacts"], facts)
        self.assertEqual(body["analysis"]["score"], facts["balanceScore"])
        self.assertEqual(body["analysis"]["balanceComment"], BALANCE_COMMENT)
        self.assertEqual(
            body["analysis"]["leftRegionalObservation"],
            LEFT_REGIONAL_OBSERVATION,
        )
        self.assertEqual(
            body["analysis"]["rightRegionalObservation"],
            RIGHT_REGIONAL_OBSERVATION,
        )
        self.assertEqual(body["analysis"]["advice"], list(REPORT_ADVICE))
        self.assertEqual(body["failedStages"], [])
        self.assertFalse(body["partialSave"])
        for report_name in (
            "pressureHeatmap",
            "pressureDistribution",
            "balance",
            "summary",
        ):
            self.assertEqual(body["reports"][report_name]["status"], "saved")
            self.assertEqual(body["reports"][report_name]["statusCode"], 200)

    def test_progress_logs_are_structured_complete_and_content_safe(self) -> None:
        backend = _Backend()
        patches = self._success_patches(backend, output_name="progress-success")
        request_dir, builder, renderer, generator, backend_patch, facts, _ = patches
        with (
            request_dir,
            builder,
            renderer,
            generator,
            backend_patch,
            patch.object(route.progress_logger, "info") as log_info,
        ):
            response = self._post(_valid_payload(measurement_session_id=37))

        self.assertEqual(response.status_code, 200, response.text)
        payloads = self._progress_payloads(log_info)
        self.assertGreaterEqual(len(payloads), 14)
        required = {
            "pipeline",
            "requestId",
            "measurementSessionId",
            "event",
            "stage",
            "message",
            "details",
        }
        for payload in payloads:
            self.assertEqual(set(payload), required)
            self.assertEqual(payload["pipeline"], route.PIPELINE_VERSION)
            self.assertEqual(payload["requestId"], "pressure-request")
            self.assertEqual(payload["measurementSessionId"], 37)
            self.assertIsInstance(payload["details"], dict)
            self.assertTrue(
                set(payload["details"]).issubset(route._PROGRESS_DETAIL_KEYS)
            )
            for value in payload["details"].values():
                self.assertIsInstance(value, (bool, int, float, str))

        facts_log = next(
            payload
            for payload in payloads
            if payload["stage"] == "pressure-balance-analysis"
            and payload["event"] == "completed"
        )
        self.assertEqual(facts_log["details"]["leftTotal"], facts["leftTotal"])
        self.assertEqual(
            facts_log["details"]["leftPressurePercent"],
            facts["leftLoadPercent"],
        )
        self.assertEqual(
            facts_log["details"]["leftHeelPercent"],
            facts["leftZonePercent"]["heel"],
        )
        self.assertEqual(
            facts_log["details"]["balanceScore"], facts["balanceScore"]
        )
        self.assertEqual(
            facts_log["details"]["qualityStatus"], facts["qualityStatus"]
        )

        completed_reports = {
            payload["details"].get("report")
            for payload in payloads
            if payload["event"] == "completed"
            and payload["details"].get("status") == "saved"
        }
        self.assertEqual(
            completed_reports,
            {"pressureHeatmap", "pressureDistribution", "balance", "summary"},
        )
        self.assertTrue(
            any(
                payload["stage"] == "pressure-comment-generation"
                and payload["event"] == "started"
                for payload in payloads
            )
        )
        self.assertTrue(
            any(
                payload["stage"] == "pressure-comment-generation"
                and payload["event"] == "completed"
                for payload in payloads
            )
        )
        terminal = payloads[-1]
        self.assertEqual(terminal["stage"], "pressure-pipeline")
        self.assertEqual(terminal["event"], "completed")
        self.assertTrue(terminal["details"]["allSaved"])

        messages = {
            (payload["stage"], payload["event"]): payload["message"]
            for payload in payloads
        }
        self.assertEqual(
            messages[("pressure-request", "started")],
            "좌우 압력 센서값을 정상적으로 받았습니다. 압력 분석을 시작합니다.",
        )
        self.assertIn(
            "각 발의 앞꿈치·중간·뒤꿈치 압력 분포를 분석하는 중",
            messages[("pressure-balance-analysis", "started")],
        )
        self.assertIn(
            "압력 분포 분석을 완료",
            messages[("pressure-balance-analysis", "completed")],
        )
        self.assertEqual(
            messages[("pressure-comment-generation", "started")],
            "분석 수치를 바탕으로 좌우 및 각 발의 압력 분포 코멘트를 생성하는 중입니다.",
        )
        self.assertEqual(
            messages[("pressure-pipeline", "completed")],
            "압력 분석과 모든 결과 저장을 완료했습니다.",
        )

        rendered_logs = self._render_progress_logs(log_info)
        self.assertTrue(
            all(
                message.startswith("[압력 분석][세션 37][요청 pressure]")
                for message in rendered_logs
            )
        )
        serialized = "\n".join(rendered_logs)
        for forbidden in (
            "route-test-token",
            "leftPressureValues",
            "rightPressureValues",
            json.dumps(list(range(12)), separators=(",", ":")),
            LEFT_REGIONAL_OBSERVATION,
            RIGHT_REGIONAL_OBSERVATION,
            BALANCE_COMMENT,
            '"body"',
            '"authorization"',
            '"api_token"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_progress_log_backend_failure_excludes_upstream_body(self) -> None:
        backend = _Backend(outcomes={"heatmap": 503})
        patches = self._success_patches(backend, output_name="progress-failure")
        request_dir, builder, renderer, generator, backend_patch, _, _ = patches
        with (
            request_dir,
            builder,
            renderer,
            generator,
            backend_patch,
            patch.object(route.progress_logger, "info") as log_info,
        ):
            response = self._post(_valid_payload(measurement_session_id=44))

        self.assertEqual(response.status_code, 207, response.text)
        payloads = self._progress_payloads(log_info)
        failed = next(
            payload
            for payload in payloads
            if payload["stage"] == "pressure-heatmap-backend-forward"
            and payload["event"] == "failed"
        )
        self.assertEqual(failed["details"]["report"], "pressureHeatmap")
        self.assertEqual(failed["details"]["status"], "http-error")
        self.assertEqual(failed["details"]["statusCode"], 503)
        self.assertTrue(failed["details"]["retryable"])
        self.assertEqual(payloads[-1]["event"], "partial")

        serialized = "\n".join(self._render_progress_logs(log_info))
        self.assertNotIn("route-test-token", serialized)
        self.assertNotIn('"authorization"', serialized)
        self.assertNotIn('"api_token"', serialized)
        self.assertNotIn('"failed":"heatmap"', serialized)

    def test_invalid_frames_are_422_before_local_or_network_work(self) -> None:
        invalid_payloads: list[dict[str, Any]] = []
        for field, value in (
            ("leftPressureValues", [0] * 11),
            ("rightPressureValues", [0] * 13),
        ):
            payload = _valid_payload()
            payload[field] = value
            invalid_payloads.append(payload)
        for field, index, value in (
            ("leftPressureValues", 4, -0.01),
            ("rightPressureValues", 3, "3"),
            ("leftPressureValues", 2, True),
            ("rightPressureValues", 5, float("nan")),
            ("leftPressureValues", 7, float("inf")),
        ):
            payload = _valid_payload()
            payload[field][index] = value
            invalid_payloads.append(payload)

        with (
            patch.object(route, "build_pressure_balance_facts") as build,
            patch.object(route, "create_pressure_heatmaps") as create,
            patch.object(
                route,
                "generate_pressure_balance_report",
                new=AsyncMock(),
            ) as generate,
            patch.object(route.httpx, "AsyncClient") as backend,
        ):
            for invalid in invalid_payloads:
                self.pressure_attempt_registry_path.unlink(missing_ok=True)
                response = self.client.post(
                    ENDPOINT,
                    content=json.dumps(invalid, allow_nan=True),
                    headers={
                        "Authorization": "Bearer route-test-token",
                        "Content-Type": "application/json",
                    },
                )
                self.assertEqual(response.status_code, 422, response.text)

        build.assert_not_called()
        create.assert_not_called()
        generate.assert_not_awaited()
        backend.assert_not_called()

    def test_no_load_and_single_side_load_request_retry_without_saving(self) -> None:
        cases = (
            ("no-load", [0.0] * 12, [0.0] * 12, ["heatmap"]),
            (
                "single-side-no-load",
                [1.0] * 12,
                [0.0] * 12,
                ["heatmap", "distribution"],
            ),
        )
        for expected_status, left, right, expected_events in cases:
            with self.subTest(expected_status=expected_status):
                output_dir = Path("route-test-output") / expected_status
                backend = _Backend()
                with (
                    patch.object(
                        route,
                        "_request_dir",
                        return_value=(expected_status, output_dir),
                    ),
                    patch.object(
                        route,
                        "create_pressure_heatmaps",
                        return_value=_saved_heatmaps(output_dir),
                    ),
                    patch.object(
                        route,
                        "generate_pressure_balance_report",
                        new=AsyncMock(),
                    ) as generate,
                    patch.object(route.httpx, "AsyncClient", return_value=backend),
                ):
                    response = self._post(
                        {
                            "measurementSessionId": 4,
                            "pressureMeasurementAttempt": 0,
                            "leftPressureValues": left,
                            "rightPressureValues": right,
                        }
                    )

                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(backend.events, [])
                generate.assert_not_awaited()
                body = response.json()["detail"]
                self.assertEqual(body["qualityStatus"], expected_status)
                self.assertTrue(body["pressureRetryRequired"])
                self.assertEqual(body["measurementStatus"], "WAITING_FOR_PRESSURE_RETRY")
                self.assertEqual(body["reports"], {})
                self.notify_pressure_analysis_started.assert_not_awaited()

    def test_local_failures_are_sanitized_and_traceable(self) -> None:
        output_dir = Path("route-test-output") / "local-failure"
        cases = (
            (
                "pressure-balance-analysis",
                patch.object(
                    route,
                    "build_pressure_balance_facts",
                    side_effect=ValueError("Bearer route-test-token"),
                ),
                422,
            ),
            (
                "pressure-heatmap-render",
                patch.object(
                    route,
                    "create_pressure_heatmaps",
                    side_effect=PressureHeatmapError("Bearer route-test-token"),
                ),
                500,
            ),
        )
        for expected_stage, failure_patch, expected_status in cases:
            with self.subTest(stage=expected_stage):
                with (
                    patch.object(
                        route,
                        "_request_dir",
                        return_value=("local-failure", output_dir),
                    ),
                    failure_patch,
                    patch.object(
                        route,
                        "generate_pressure_balance_report",
                        new=AsyncMock(),
                    ) as generate,
                    patch.object(route.httpx, "AsyncClient") as backend,
                    patch.object(Path, "mkdir", autospec=True),
                    patch.object(Path, "write_text", autospec=True),
                ):
                    response = self._post(_valid_payload())

                self.assertEqual(response.status_code, expected_status, response.text)
                self.assertEqual(response.json()["detail"]["stage"], expected_stage)
                self.assertNotIn("route-test-token", response.text)
                self.assertEqual(
                    response.headers["X-Feetfit-Request-Id"],
                    "local-failure",
                )
                self.assertEqual(
                    response.headers["X-Feetfit-Pipeline"],
                    route.PIPELINE_VERSION,
                )
                generate.assert_not_awaited()
                backend.assert_not_called()

    def test_each_required_core_http_failure_is_207_and_gates_downstream(self) -> None:
        cases = (
            (
                "heatmap",
                "pressureHeatmap",
                "pressure-heatmap-backend-forward",
                503,
                True,
            ),
            (
                "distribution",
                "pressureDistribution",
                "pressure-distribution-backend-forward",
                307,
                False,
            ),
        )
        for key, report_name, stage, status_code, retryable in cases:
            with self.subTest(key=key):
                backend = _Backend(outcomes={key: status_code})
                patches = self._success_patches(backend, output_name=f"core-{key}")
                request_dir, builder, renderer, generator, backend_patch, _, _ = patches
                with request_dir, builder, renderer, generator as generate, backend_patch:
                    response = self._post(_valid_payload())

                self.assertEqual(response.status_code, 207, response.text)
                generate.assert_not_awaited()
                self.assertEqual(backend.events, ["heatmap", "distribution"])
                body = response.json()
                self.assertEqual(body["failedStages"], [stage])
                self.assertTrue(body["partialSave"])
                self.assertEqual(body["reports"][report_name]["status"], "http-error")
                self.assertEqual(body["reports"][report_name]["statusCode"], status_code)
                self.assertEqual(body["errors"][0]["retryable"], retryable)
                self.assertEqual(body["reports"]["balance"]["status"], "skipped")
                self.assertEqual(body["reports"]["summary"]["status"], "skipped")
                self.assertNotIn("route-test-token", response.text)

    def test_balance_and_summary_http_failures_are_independent_207_results(self) -> None:
        cases = (
            ("balance", "pressure-balance-backend-forward", 503, True),
            ("summary", "pressure-summary-backend-forward", 409, False),
        )
        for key, stage, status_code, retryable in cases:
            with self.subTest(key=key):
                backend = _Backend(outcomes={key: status_code})
                patches = self._success_patches(
                    backend,
                    output_name=f"downstream-{key}",
                )
                request_dir, builder, renderer, generator, backend_patch, _, _ = patches
                with request_dir, builder, renderer, generator as generate, backend_patch:
                    response = self._post(_valid_payload())

                self.assertEqual(response.status_code, 207, response.text)
                generate.assert_awaited_once()
                self.assertEqual(
                    backend.events,
                    ["heatmap", "distribution", "gpt", "balance", "summary"],
                )
                body = response.json()
                self.assertEqual(body["failedStages"], [stage])
                self.assertTrue(body["partialSave"])
                self.assertEqual(body["reports"][key]["status"], "http-error")
                self.assertEqual(body["reports"][key]["statusCode"], status_code)
                self.assertEqual(body["errors"][0]["retryable"], retryable)
                other = "summary" if key == "balance" else "balance"
                self.assertEqual(body["reports"][other]["status"], "saved")
                self.assertNotIn("route-test-token", response.text)

    def test_each_backend_transport_failure_is_502_with_stage_and_partial_state(self) -> None:
        cases = (
            ("heatmap", "pressureHeatmap", "pressure-heatmap-backend-forward"),
            (
                "distribution",
                "pressureDistribution",
                "pressure-distribution-backend-forward",
            ),
            ("balance", "balance", "pressure-balance-backend-forward"),
            ("summary", "summary", "pressure-summary-backend-forward"),
        )
        for key, report_name, expected_stage in cases:
            with self.subTest(key=key):
                failure = httpx.ConnectError("Bearer route-test-token must not leak")
                backend = _Backend(outcomes={key: failure})
                patches = self._success_patches(backend, output_name=f"transport-{key}")
                request_dir, builder, renderer, generator, backend_patch, _, _ = patches
                with request_dir, builder, renderer, generator as generate, backend_patch:
                    response = self._post(_valid_payload())

                self.assertEqual(response.status_code, 502, response.text)
                detail = response.json()["detail"]
                self.assertEqual(detail["stage"], expected_stage)
                self.assertTrue(detail["retryable"])
                self.assertTrue(detail["partialSave"])
                self.assertEqual(
                    detail["reports"][report_name]["status"],
                    "transport-error",
                )
                self.assertNotIn("route-test-token", response.text)
                self.assertEqual(
                    response.headers["X-Feetfit-Request-Id"],
                    "pressure-request",
                )
                if key in {"heatmap", "distribution"}:
                    generate.assert_not_awaited()
                else:
                    generate.assert_awaited_once()

    def test_gpt_failure_is_500_after_both_core_saves_without_secret_leak(self) -> None:
        backend = _Backend()
        patches = self._success_patches(backend, output_name="gpt-failure")
        request_dir, builder, renderer, generator, backend_patch, _, _ = patches
        with (
            request_dir,
            builder,
            renderer,
            generator,
            backend_patch,
            patch.object(
                route,
                "generate_pressure_balance_report",
                new=AsyncMock(
                    side_effect=RuntimeError("sk-secret Bearer route-test-token")
                ),
            ) as failing_generate,
        ):
            response = self._post(_valid_payload())

        self.assertEqual(response.status_code, 500, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["stage"], "pressure-comment-generation")
        self.assertTrue(detail["partialSave"])
        self.assertTrue(detail["retryable"])
        self.assertEqual(detail["reports"]["pressureHeatmap"]["status"], "saved")
        self.assertEqual(
            detail["reports"]["pressureDistribution"]["status"],
            "saved",
        )
        self.assertNotIn("route-test-token", response.text)
        failing_generate.assert_awaited_once()
        self.assertEqual(backend.events, ["heatmap", "distribution"])


if __name__ == "__main__":
    unittest.main()
