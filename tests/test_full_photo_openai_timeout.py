from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service
from tests.photo_image_fixtures import encoded_photo


PNG_BYTES = encoded_photo(".png", (80, 100, 120))


class _FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {"output_text": '{"value":"ok"}', "status": "completed"}


class _CapturingAsyncClient:
    last_timeout: object | None = None

    def __init__(self, *, timeout: object, trust_env: bool = False):
        type(self).last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url: str, *, headers: dict, json: dict):
        return _FakeResponse()


class FullPhotoOpenAITimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_photo_socket_wait_keeps_connect_and_pool_bounded(self) -> None:
        with (
            patch.object(service.settings, "openai_api_key", "test-key"),
            patch.object(service.settings, "openai_report_include_images", False),
            patch.object(service.settings, "openai_report_connect_timeout_seconds", 7.0),
            patch.object(service.settings, "openai_report_pool_timeout_seconds", 9.0),
            patch.object(service.httpx, "AsyncClient", _CapturingAsyncClient),
        ):
            result = await service._create_structured_response(
                system_prompt="system",
                user_payload={},
                schema_name="timeout_test",
                json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                wait_for_response_without_timeout=True,
            )

        self.assertEqual(result, {"value": "ok"})
        timeout = _CapturingAsyncClient.last_timeout
        self.assertIsInstance(timeout, service.httpx.Timeout)
        self.assertEqual(timeout.connect, 7.0)
        self.assertIsNone(timeout.read)
        self.assertIsNone(timeout.write)
        self.assertEqual(timeout.pool, 9.0)

    async def test_full_photo_repairs_share_one_request_budget_and_deadline(self) -> None:
        create_response = AsyncMock(return_value={})
        with (
            patch.object(service.settings, "openai_report_fallback_enabled", False),
            patch.object(service.settings, "openai_report_max_attempts", 5),
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaises(ValueError):
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=PNG_BYTES,
                    left_plantar_raw_image_bytes=PNG_BYTES,
                    right_plantar_raw_image_bytes=PNG_BYTES,
                    facts={},
                    related_analysis_images={},
                )

        self.assertIs(
            create_response.await_args.kwargs["wait_for_response_without_timeout"],
            True,
        )
        self.assertEqual(create_response.await_count, 5)
        budgets = [call.kwargs["request_budget"] for call in create_response.await_args_list]
        self.assertTrue(all(budget is budgets[0] for budget in budgets))
        self.assertEqual(budgets[0].maximum_attempts, 5)
