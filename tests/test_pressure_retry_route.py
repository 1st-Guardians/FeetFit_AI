from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.api.routes import pressure_heatmaps as route
from app.services.pressure_measurement_status import PressureStatusOutcome
from app.services import pressure_measurement_status as status_service
from tests import test_pressure_heatmap_route as fixtures


class PressureRetryRouteTests(unittest.TestCase):
    setUp = fixtures.PressureHeatmapRouteTests.setUp
    tearDown = fixtures.PressureHeatmapRouteTests.tearDown
    _post = fixtures.PressureHeatmapRouteTests._post
    _intake_post = fixtures.PressureHeatmapRouteTests._intake_post
    _success_patches = fixtures.PressureHeatmapRouteTests._success_patches

    def test_sensor_schema_failures_request_retry_without_queue_render_or_save(self):
        cases = [[], [1.0] * 11, [1.0] * 13, [-1.0] + [1.0] * 11,
                 [True] + [1.0] * 11, ["1"] + [1.0] * 11, None, "invalid"]
        for value in cases:
            for side in ("leftPressureValues", "rightPressureValues"):
                self.pressure_attempt_registry_path.unlink(missing_ok=True)
                with self.subTest(side=side, value=value):
                    payload = fixtures._valid_payload(34)
                    payload["pressureMeasurementAttempt"] = 2
                    payload[side] = value
                    with (patch.object(route.BackgroundTasks, "add_task") as jobs,
                          patch.object(route, "create_pressure_heatmaps") as render,
                          patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
                        response = self._intake_post(payload)
                    self.assertEqual(response.status_code, 422, response.text)
                    detail = response.json()["detail"]
                    self.assertTrue(detail["pressureRetryRequired"])
                    self.assertEqual(detail["measurementStatus"], "WAITING_FOR_PRESSURE_RETRY")
                    self.assertEqual(detail["failureReason"], "INVALID_PRESSURE_DATA")
                    self.assertEqual(detail["pressureMeasurementAttempt"], 2)
                    self.assertEqual(self.notify_pressure_retry_required.await_args.kwargs["pressure_measurement_attempt"], 2)
                    jobs.assert_not_called()
                    render.assert_not_called()
                    terminal.assert_not_awaited()
                    self.assertEqual(route._active_pressure_analysis_sessions, set())
        self.pressure_attempt_registry_path.unlink(missing_ok=True)
        payload = fixtures._valid_payload(34)
        del payload["rightPressureValues"]
        self.assertEqual(self._intake_post(payload).status_code, 422)
        self.assertEqual(self.notify_pressure_retry_required.await_args.kwargs["pressure_measurement_attempt"], 0)

    def test_invalid_identity_attempt_or_extra_field_never_changes_backend_state(self):
        for fields in ({"measurementSessionId": 0}, {"pressureMeasurementAttempt": -1},
                       {"pressureMeasurementAttempt": True}, {"pressureMeasurementAttempt": "1"},
                       {"extra": "private-value"}):
            with self.subTest(fields=fields):
                self.notify_pressure_retry_required.reset_mock()
                payload = {**fixtures._valid_payload(34), "leftPressureValues": [], **fields}
                response = self._intake_post(payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.notify_pressure_retry_required.assert_not_awaited()

    def test_missing_attempt_is_rejected_without_defaulting_or_changing_state(self):
        payload = fixtures._valid_payload(9)
        del payload["pressureMeasurementAttempt"]
        with (patch.object(route.BackgroundTasks, "add_task") as jobs,
              patch.object(route, "claim_pressure_attempt") as claim):
            response = self._intake_post(payload)
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "pressureMeasurementAttempt"])
        jobs.assert_not_called()
        claim.assert_not_called()
        self.notify_pressure_measurement_received.assert_not_awaited()
        self.notify_pressure_analysis_started.assert_not_awaited()
        self.notify_pressure_retry_required.assert_not_awaited()

    def test_all_twelve_nonzero_weak_sensors_request_retry_before_analysis_or_save(self):
        for side in ("left", "right"):
            self.notify_pressure_analysis_started.reset_mock()
            self.notify_pressure_retry_required.reset_mock()
            payload = {"measurementSessionId": 34, "pressureMeasurementAttempt": 2,
                       "leftPressureValues": [1000.0] * 12, "rightPressureValues": [1000.0] * 12}
            payload[side + "PressureValues"] = [0.01] * 12
            with (self.subTest(side=side),
                  patch.object(route, "create_pressure_heatmaps") as render,
                  patch.object(route, "generate_pressure_balance_report", new=AsyncMock()) as generate,
                  patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
                response = self._post(payload)
                self.assertEqual(response.status_code, 422, response.text)
                detail = response.json()["detail"]
                self.assertEqual(detail["qualityStatus"], "single-side-low-load")
                self.assertEqual(detail["lowPressureSide"], side)
                self.assertEqual(detail[side + "EffectiveSensorCount"], 0)
                self.assertTrue(detail["pressureRetryRequired"])
                self.notify_pressure_retry_required.assert_awaited_once()
                self.assertEqual(self.notify_pressure_retry_required.await_args.kwargs["pressure_measurement_attempt"], 2)
                self.notify_pressure_analysis_started.assert_not_awaited()
                render.assert_not_called()
                generate.assert_not_awaited()
                terminal.assert_not_awaited()

    def test_one_effective_sensor_keeps_the_frame_in_analysis_even_below_five_percent_load(self):
        backend = fixtures._Backend()
        patches = self._success_patches(backend)
        payload = {"measurementSessionId": 34, "pressureMeasurementAttempt": 2,
                   "leftPressureValues": [50.0] + [0.01] * 11, "rightPressureValues": [1000.0] * 12}
        with ExitStack() as stack:
            for index, context in enumerate(patches[:5]):
                patched = stack.enter_context(context)
                if index == 1:
                    patched.return_value = fixtures.real_build_pressure_balance_facts(
                        payload["leftPressureValues"], payload["rightPressureValues"],
                    )
            response = self._post(payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.notify_pressure_retry_required.assert_not_awaited()
        self.notify_pressure_analysis_started.assert_awaited_once()
        self.assertEqual(len(backend.calls), 4)

    def test_retry_exhaustion_unconfirmed_callback_and_conflict_never_send_failed(self):
        outcomes = [
            (PressureStatusOutcome(confirmed=True, status="FAILED"), 422, False),
            (PressureStatusOutcome(), 502, False),
            (PressureStatusOutcome(attempt_conflict=True, backend_status_code=409), 409, False),
            (RuntimeError("private-callback-error"), 502, False),
        ]
        for outcome, expected, retry in outcomes:
            self.pressure_attempt_registry_path.unlink(missing_ok=True)
            callback = AsyncMock(**({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}))
            with (self.subTest(outcome=outcome),
                  patch.object(route, "notify_pressure_retry_required", new=callback),
                  patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal,
                  patch.object(route.BackgroundTasks, "add_task") as jobs):
                payload = fixtures._valid_payload(34)
                payload["leftPressureValues"] = []
                response = self._intake_post(payload)
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(response.json()["detail"]["pressureRetryRequired"], retry)
                self.assertNotIn("private-callback-error", response.text)
                callback.assert_awaited_once()
                jobs.assert_not_called()
                terminal.assert_not_awaited()

    def test_successful_remeasurement_keeps_attempt_in_status_but_not_report_saves(self):
        backend = fixtures._Backend()
        patches = self._success_patches(backend)
        payload = {**fixtures._valid_payload(34), "pressureMeasurementAttempt": 2}
        with ExitStack() as stack:
            for context in patches[:5]:
                stack.enter_context(context)
            response = self._post(payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.notify_pressure_analysis_started.await_args.kwargs["pressure_measurement_attempt"], 2)
        self.notify_pressure_retry_required.assert_not_awaited()
        self.assertEqual(len(backend.calls), 4)
        for _, kwargs in backend.calls:
            self.assertNotIn("pressureMeasurementAttempt", kwargs.get("params", {}))
            if "json" in kwargs:
                self.assertNotIn("pressureMeasurementAttempt", kwargs["json"])
            if "files" in kwargs:
                self.assertNotIn("pressureMeasurementAttempt", json.loads(kwargs["files"]["request"][1]))

    def test_received_attempt_survives_queue_and_actual_status_http_calls_only(self):
        class StatusBackend(fixtures._Backend):
            def __init__(self):
                super().__init__()
                self.status_calls = []

            async def patch(self, url, **kwargs):
                self.status_calls.append((url, kwargs))
                params = kwargs["params"]
                return fixtures.httpx.Response(200, json={"result": {
                    "status": params["status"], "measurementSessionId": 34,
                    "pressureMeasurementAttempt": params["pressureMeasurementAttempt"],
                }})

        for attempt in (0, 1):
            with self.subTest(attempt=attempt):
                backend = StatusBackend()
                patches = self._success_patches(backend)
                payload = {**fixtures._valid_payload(34), "pressureMeasurementAttempt": attempt}
                with ExitStack() as stack:
                    for context in patches[:5]:
                        stack.enter_context(context)
                    stack.enter_context(patch.object(route, "notify_pressure_measurement_received",
                                                    side_effect=status_service.notify_pressure_measurement_received))
                    stack.enter_context(patch.object(route, "notify_pressure_analysis_started",
                                                    side_effect=status_service.notify_pressure_analysis_started))
                    terminal = stack.enter_context(patch.object(route, "notify_measurement_failed", new=AsyncMock()))
                    with patch.object(route.BackgroundTasks, "add_task") as jobs:
                        response = self._intake_post(payload)
                    self.assertEqual(response.status_code, 202, response.text)
                    self.assertEqual(response.json()["pressureMeasurementAttempt"], attempt)
                    jobs.assert_called_once()
                    _, work_item = jobs.call_args.args
                    self.assertEqual(work_item.pressure_measurement_attempt, attempt)
                    payload["pressureMeasurementAttempt"] = attempt + 10
                    asyncio.run(route._run_pressure_analysis_job(work_item))
                    terminal.assert_not_awaited()
                self.assertEqual([kwargs["params"]["status"] for _, kwargs in backend.status_calls],
                                 ["MEASURING_PRESSURE", "ANALYZING"])
                self.assertTrue(all(kwargs["params"]["pressureMeasurementAttempt"] == attempt
                                    for _, kwargs in backend.status_calls))
                self.assertEqual(len(backend.calls), 4)
                for _, kwargs in backend.calls:
                    self.assertNotIn("pressureMeasurementAttempt", kwargs.get("params", {}))
                    if "json" in kwargs:
                        self.assertNotIn("pressureMeasurementAttempt", kwargs["json"])
                    if "files" in kwargs:
                        self.assertNotIn("pressureMeasurementAttempt", json.loads(kwargs["files"]["request"][1]))
                self.assertEqual(route._active_pressure_analysis_sessions, set())

    def test_unconfirmed_measuring_is_not_queued_and_same_attempt_is_not_patched_again(self):
        for index, outcome in enumerate((PressureStatusOutcome(), PressureStatusOutcome(attempt_conflict=True),
                                         PressureStatusOutcome(confirmed=True, status="ANALYZING"),
                                         RuntimeError("private-status-error"))):
            callback = AsyncMock(**({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}))
            payload = {**fixtures._valid_payload(900 + index), "pressureMeasurementAttempt": 1}
            with (self.subTest(outcome=outcome),
                  patch.object(route, "notify_pressure_measurement_received", new=callback),
                  patch.object(route.BackgroundTasks, "add_task") as jobs,
                  patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
                response = self._intake_post(payload)
                expected = 409 if isinstance(outcome, PressureStatusOutcome) and outcome.attempt_conflict else 502
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(response.json()["detail"]["pressureMeasurementAttempt"], 1)
                self.assertNotIn("private-status-error", response.text)
                repeated = self._intake_post(payload)
                self.assertEqual(repeated.status_code, 409, repeated.text)
                callback.assert_awaited_once()
                jobs.assert_not_called()
                terminal.assert_not_awaited()
                self.assertEqual(route._active_pressure_analysis_sessions, set())

    def test_failed_job_does_not_allow_same_attempt_to_reenter_queue(self):
        payload = {**fixtures._valid_payload(9), "pressureMeasurementAttempt": 1}
        with (patch.object(route.BackgroundTasks, "add_task") as jobs,
              patch.object(route, "_execute_pressure_pipeline", new=AsyncMock(side_effect=RuntimeError("failed"))),
              patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
            first = self._intake_post(payload)
            self.assertEqual(first.status_code, 202, first.text)
            _, work = jobs.call_args.args
            asyncio.run(route._run_pressure_analysis_job(work))
            self.assertEqual(terminal.await_args.kwargs["pressure_measurement_attempt"], 1)
            self.assertEqual(route._active_pressure_analysis_sessions, set())
            repeated = self._intake_post(payload)
            self.assertEqual(repeated.status_code, 409, repeated.text)
            self.assertEqual(repeated.json()["detail"]["errorType"], "PressureMeasurementAttemptAlreadyConsumed")
            jobs.assert_called_once()
            self.notify_pressure_measurement_received.assert_awaited_once()
            terminal.assert_awaited_once()
            next_attempt = self._intake_post({**payload, "pressureMeasurementAttempt": 2})
            self.assertEqual(next_attempt.status_code, 202, next_attempt.text)
            self.assertEqual(jobs.call_args.args[1].pressure_measurement_attempt, 2)
            self.assertEqual(self.notify_pressure_measurement_received.await_args.kwargs["pressure_measurement_attempt"], 2)
            self.assertEqual(jobs.call_count, 2)

    def test_retry_uses_received_attempt_and_duplicate_frame_has_no_second_callback(self):
        payload = {"measurementSessionId": 9, "pressureMeasurementAttempt": 1,
                   "leftPressureValues": [0.0] * 12, "rightPressureValues": [1000.0] * 12}
        with (patch.object(route.BackgroundTasks, "add_task") as jobs,
              patch.object(route, "build_pressure_balance_facts", side_effect=fixtures.real_build_pressure_balance_facts),
              patch.object(route, "create_pressure_heatmaps") as render,
              patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
            accepted = self._intake_post(payload)
            self.assertEqual(accepted.status_code, 202, accepted.text)
            _, work = jobs.call_args.args
            asyncio.run(route._run_pressure_analysis_job(work))
            self.assertEqual(self.notify_pressure_retry_required.await_args.kwargs["pressure_measurement_attempt"], 1)
            repeated = self._intake_post(payload)
            self.assertEqual(repeated.status_code, 409, repeated.text)
            jobs.assert_called_once()
            self.notify_pressure_measurement_received.assert_awaited_once()
            self.notify_pressure_retry_required.assert_awaited_once()
            self.notify_pressure_analysis_started.assert_not_awaited()
            render.assert_not_called()
            terminal.assert_not_awaited()

    def test_attempt_registry_failure_never_sends_status_or_queues_work(self):
        with (patch.object(route, "claim_pressure_attempt", side_effect=OSError("private-path")),
              patch.object(route.BackgroundTasks, "add_task") as jobs):
            response = self._intake_post({**fixtures._valid_payload(9), "pressureMeasurementAttempt": 1})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("private-path", response.text)
        self.assertFalse(response.json()["detail"]["retryable"])
        jobs.assert_not_called()
        self.notify_pressure_measurement_received.assert_not_awaited()
        self.assertEqual(route._active_pressure_analysis_sessions, set())

    def test_hardware_aliases_preserve_arrays_and_attempt_in_owned_work(self):
        for use_snake_identity in (False, True):
            self.pressure_attempt_registry_path.unlink(missing_ok=True)
            with self.subTest(snake_identity=use_snake_identity):
                payload = {"left_pressure_values": list(range(12)),
                           "right_pressure_values": list(range(12, 24))}
                if use_snake_identity:
                    payload.update(measurement_session_id=34, pressure_measurement_attempt=2)
                else:
                    payload.update(measurementSessionId=34, pressureMeasurementAttempt=2)
                with (patch.object(route.settings, "pressure_sensor_input_mode", "hardware"),
                      patch.object(route.BackgroundTasks, "add_task") as jobs):
                    response = self._intake_post(payload)
                self.assertEqual(response.status_code, 202, response.text)
                _, work = jobs.call_args.args
                self.assertEqual(work.pressure_measurement_attempt, 2)
                self.assertEqual(work.left_pressure_values, tuple(range(12)))
                self.assertEqual(work.right_pressure_values, tuple(range(12, 24)))
                route._active_pressure_analysis_sessions.clear()

    def test_malformed_snake_case_hardware_frame_requests_retry_with_received_attempt(self):
        for use_snake_identity in (False, True):
            self.pressure_attempt_registry_path.unlink(missing_ok=True)
            self.notify_pressure_retry_required.reset_mock()
            payload = {"left_pressure_values": [], "right_pressure_values": list(range(12))}
            if use_snake_identity:
                payload.update(measurement_session_id=34, pressure_measurement_attempt=2)
            else:
                payload.update(measurementSessionId=34, pressureMeasurementAttempt=2)
            with (self.subTest(snake_identity=use_snake_identity),
                  patch.object(route.BackgroundTasks, "add_task") as jobs):
                response = self._intake_post(payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"]["pressureMeasurementAttempt"], 2)
                self.assertEqual(response.json()["detail"]["measurementStatus"], "WAITING_FOR_PRESSURE_RETRY")
                self.notify_pressure_retry_required.assert_awaited_once()
                jobs.assert_not_called()
                self.assertEqual(route._active_pressure_analysis_sessions, set())

    def test_unconfirmed_analyzing_never_renders_or_saves_or_sends_failed(self):
        for outcome in (PressureStatusOutcome(), PressureStatusOutcome(attempt_conflict=True),
                        PressureStatusOutcome(confirmed=True, status="WAITING_FOR_PRESSURE_RETRY"),
                        RuntimeError("private-status-error")):
            callback = AsyncMock(**({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}))
            with (self.subTest(outcome=outcome),
                  patch.object(route, "notify_pressure_analysis_started", new=callback),
                  patch.object(route, "create_pressure_heatmaps") as render,
                  patch.object(route, "notify_measurement_failed", new=AsyncMock()) as terminal):
                response = self._post({**fixtures._valid_payload(34), "pressureMeasurementAttempt": 1})
            expected = 409 if isinstance(outcome, PressureStatusOutcome) and outcome.attempt_conflict else 502
            self.assertEqual(response.status_code, expected, response.text)
            render.assert_not_called()
            terminal.assert_not_awaited()
            self.assertNotIn("private-status-error", response.text)

    def test_report_conflict_is_a_save_error_and_failed_status_keeps_attempt(self):
        backend = fixtures._Backend(outcomes={"heatmap": 409})
        patches = self._success_patches(backend)
        with ExitStack() as stack:
            for context in patches[:5]:
                stack.enter_context(context)
            terminal = stack.enter_context(patch.object(route, "notify_measurement_failed", new=AsyncMock()))
            response = self._post({**fixtures._valid_payload(34), "pressureMeasurementAttempt": 1})
        self.assertEqual(response.status_code, 207, response.text)
        self.assertEqual(backend.events, ["heatmap", "distribution"])
        terminal.assert_awaited_once()
        self.assertEqual(terminal.await_args.kwargs["failure_reason"], "REPORT_SAVE_ERROR")
        self.assertEqual(terminal.await_args.kwargs["pressure_measurement_attempt"], 1)


if __name__ == "__main__":
    unittest.main()
