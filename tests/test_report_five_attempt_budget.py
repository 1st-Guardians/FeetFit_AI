from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.services import report_text_generation as service
from tests.test_full_photo_analysis_texts import (
    ANALYSIS_PNG_BYTES,
    JPEG_BYTES,
    LEFT_PNG_BYTES,
    RIGHT_WEBP_BYTES,
    _facts,
    _valid_generated,
)
from tests.test_pressure_balance_report_text import (
    LEFT_PNG,
    RIGHT_PNG,
    VALID_ADVICE,
    _proportional_values,
)
from tests.test_report_text_resilience import _FakeClient, _response


def _generated(data: dict) -> httpx.Response:
    return _response(body={"status": "completed", "output_text": json.dumps(data)})


def _invalid_photo() -> dict:
    invalid = _valid_generated()
    invalid["hallux_score_analysis_text"] = "점수는 100입니다."
    return invalid


def _invalid_pressure() -> dict:
    return {**VALID_ADVICE, "score": 100}


def _request_payload(request: dict) -> dict:
    return json.loads(request["body"]["input"][1]["content"][0]["text"])


def _request_images(request: dict) -> list[dict]:
    return [
        item for item in request["body"]["input"][1]["content"]
        if item["type"] == "input_image"
    ]


class ReportFiveAttemptBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sleep = AsyncMock()
        values = {
            "tinea_inflammation_enabled": True,
            "openai_api_key": "unit-test-five-attempt-key",
            "openai_report_model": "unit-test-five-attempt-model",
            "openai_report_include_images": True,
            "openai_report_text_enabled": True,
            "openai_report_fallback_enabled": True,
            "openai_report_max_attempts": 5,
            "openai_report_timeout_seconds": 30.0,
            "openai_report_full_photo_timeout_seconds": 240.0,
            "openai_report_retry_max_delay_seconds": 30.0,
        }
        for name, value in values.items():
            active = patch.object(service.settings, name, value)
            active.start()
            self.addCleanup(active.stop)
        for active in (
            patch.object(service.asyncio, "sleep", self.sleep),
            patch.object(service.random, "uniform", return_value=0.0),
        ):
            active.start()
            self.addCleanup(active.stop)

    async def _direct(self, **overrides):
        return await service._create_structured_response(
            system_prompt="Return the requested JSON object.",
            user_payload={"fact": "unchanged"},
            schema_name="five_attempt_unit_test",
            json_schema={"type": "object"},
            **overrides,
        )

    async def _photo(self, facts=None):
        return await service.generate_full_photo_analysis_texts(
            dorsal_raw_image_bytes=JPEG_BYTES,
            left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
            right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
            facts=_facts() if facts is None else facts,
            related_analysis_images={"ANALYSIS": ANALYSIS_PNG_BYTES},
        )

    async def _pressure(self, facts=None):
        left, right = _proportional_values()
        return await service.generate_pressure_balance_report(
            left_pressure_values=left,
            right_pressure_values=right,
            pressure_facts=(
                service.build_pressure_balance_facts(left, right) if facts is None else facts
            ),
            left_heatmap_png=LEFT_PNG,
            right_heatmap_png=RIGHT_PNG,
        )

    def _assert_unchanged_images_and_model(self, requests: list[dict], expected: list[bytes]) -> None:
        expected_images = [
            {"type": "input_image", "image_url": service._image_data_url(image), "detail": "high"}
            for image in expected
        ]
        for request in requests:
            self.assertEqual(request["body"]["model"], "unit-test-five-attempt-model")
            self.assertIs(request["body"]["store"], False)
            self.assertIs(request["body"]["text"]["format"]["strict"], True)
            self.assertEqual(_request_images(request), expected_images)

    async def test_transport_failures_can_recover_on_fifth_http_request(self) -> None:
        failure = _response(500, {"error": {"code": "server_error"}})
        client = _FakeClient([failure] * 4 + [_response()])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await self._direct(), {"message": "완료"})
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(self.sleep.await_count, 4)
        for request in client.requests:
            self.assertEqual(request, client.requests[0])

    async def test_persistent_transport_failure_uses_exactly_five_http_requests(self) -> None:
        client = _FakeClient([_response(500, {"error": {"code": "server_error"}})] * 5)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(service.ReportGenerationError) as raised:
                await self._direct()
        self.assertEqual(raised.exception.code, "API_HTTP_ERROR")
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(self.sleep.await_count, 4)

    async def test_full_photo_semantic_repair_can_succeed_on_fifth_http_request(self) -> None:
        client = _FakeClient([_generated(_invalid_photo())] * 4 + [_generated(_valid_generated())])
        facts = _facts()
        original_facts = copy.deepcopy(facts)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            result = await self._photo(facts)
        self.assertEqual(result.generation_source, "OPENAI")
        self.assertEqual(result.hallux_score_analysis_text,
                         _valid_generated()["hallux_score_analysis_text"])
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(facts, original_facts)
        self._assert_unchanged_images_and_model(
            client.requests, [JPEG_BYTES, service.combine_raw_plantar_photos(LEFT_PNG_BYTES, RIGHT_WEBP_BYTES), ANALYSIS_PNG_BYTES]
        )
        first_payload = _request_payload(client.requests[0])
        self.assertNotIn("validation_retry", first_payload)
        for request in client.requests:
            self.assertEqual(_request_payload(request)["authoritative_facts_do_not_modify"], original_facts)
        for request in client.requests[1:]:
            correction = _request_payload(request)["validation_retry"]
            self.assertEqual(correction["maximum_attempts"], 5)
            self.assertEqual(correction["failed_field"], "hallux_score_analysis_text")

    async def test_mixed_transport_json_and_semantic_errors_share_five_request_limit(self) -> None:
        client = _FakeClient([
            httpx.ConnectError("unit-test-offline"),
            httpx.Response(200, content=b"broken API JSON"),
            _generated(_invalid_photo()),
            _response(500, {"error": {"code": "server_error"}}),
            _generated(_invalid_photo()),
        ])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            result = await self._photo()
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        self.assertEqual(len(client.requests), 5)
        self.assertTrue(result.generation_failure_code)
        self._assert_unchanged_images_and_model(
            client.requests, [JPEG_BYTES, service.combine_raw_plantar_photos(LEFT_PNG_BYTES, RIGHT_WEBP_BYTES), ANALYSIS_PNG_BYTES]
        )

    async def test_full_photo_persistent_semantic_failure_falls_back_after_five_requests(self) -> None:
        client = _FakeClient([_generated(_invalid_photo())] * 5)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            result = await self._photo()
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")

    async def test_pressure_semantic_repair_reaches_fifth_request_without_changing_facts(self) -> None:
        client = _FakeClient([_generated(_invalid_pressure())] * 4 + [_generated(VALID_ADVICE)])
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)
        original_facts = copy.deepcopy(facts)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            result = await self._pressure(facts)
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(result.generation_source, "OPENAI")
        self.assertEqual(result.left_regional_observation, VALID_ADVICE["leftRegionalObservation"])
        self.assertEqual(result.score, original_facts["balanceScore"])
        self.assertEqual(facts, original_facts)
        self._assert_unchanged_images_and_model(client.requests, [LEFT_PNG, RIGHT_PNG])
        for request in client.requests:
            self.assertEqual(
                _request_payload(request)["numeric_source_of_truth"]["serverComputedPressureFacts"],
                original_facts,
            )
        # A retry must explicitly correct the rejected contract rather than
        # resending the same semantic instructions without feedback.
        self.assertNotEqual(client.requests[0]["body"]["input"], client.requests[1]["body"]["input"])

    async def test_pressure_mixed_failures_cannot_multiply_transport_and_semantic_budgets(self) -> None:
        client = _FakeClient([
            _generated(_invalid_pressure()),
            httpx.ReadTimeout("unit-test-timeout"),
            _response(500, {"error": {"code": "server_error"}}),
            _generated(_invalid_pressure()),
            _generated(_invalid_pressure()),
        ])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            result = await self._pressure()
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        left, right = _proportional_values()
        self.assertEqual(result.score, service.build_pressure_balance_facts(left, right)["balanceScore"])

    async def test_auth_and_quota_failures_stop_generators_after_one_http_request(self) -> None:
        for generator in (self._photo, self._pressure):
            for status, code in ((401, "invalid_api_key"), (429, "insufficient_quota")):
                with self.subTest(generator=generator.__name__, status=status):
                    client = _FakeClient([_response(status, {"error": {"code": code}})])
                    with patch.object(service.httpx, "AsyncClient", return_value=client):
                        result = await generator()
                    self.assertEqual(len(client.requests), 1)
                    self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        self.sleep.assert_not_awaited()

    async def test_one_explicit_request_budget_caps_multiple_structured_calls(self) -> None:
        budget = service._ReportRequestBudget.for_report()
        client = _FakeClient([_response()] * 5)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            for _ in range(5):
                self.assertEqual(await self._direct(request_budget=budget), {"message": "완료"})
            with self.assertRaises(service.ReportGenerationError):
                await self._direct(request_budget=budget)
        self.assertEqual(budget.maximum_attempts, 5)
        self.assertEqual(budget.attempts, 5)
        self.assertEqual(len(client.requests), 5)

    async def test_shared_deadline_is_not_reset_after_full_photo_semantic_failure(self) -> None:
        instant = [100.0]
        clock = SimpleNamespace(monotonic=lambda: instant[0])

        async def late_invalid_response():
            instant[0] = 341.0
            return _generated(_invalid_photo())

        client = _FakeClient([late_invalid_response])
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client),
            patch.object(service, "time", clock),
        ):
            result = await self._photo()
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        self.sleep.assert_not_awaited()

    async def test_shared_deadline_is_not_reset_between_structured_calls(self) -> None:
        instant = [100.0]
        budget = service._ReportRequestBudget(maximum_attempts=5, deadline=105.0)
        clock = SimpleNamespace(monotonic=lambda: instant[0])
        client = _FakeClient([_response()])
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client),
            patch.object(service, "time", clock),
        ):
            await self._direct(request_budget=budget)
            instant[0] = 106.0
            with self.assertRaises(service.ReportGenerationError) as raised:
                await self._direct(request_budget=budget)
        self.assertEqual(raised.exception.code, "API_TIMEOUT")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(budget.attempts, 1)


if __name__ == "__main__":
    unittest.main()
