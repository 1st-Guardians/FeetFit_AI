from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.services import report_text_generation as service


def _response(status: int = 200, body: object = None, **kwargs) -> httpx.Response:
    if body is None:
        body = {"status": "completed", "output_text": '{"message":"완료"}'}
    return httpx.Response(status, json=body, **kwargs)


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.exited = True

    async def post(self, url, *, headers, json):
        self.requests.append({"url": url, "body": copy.deepcopy(json)})
        if not self.outcomes:
            raise AssertionError("unexpected extra provider request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class ReportTextResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sleep = AsyncMock()
        settings = {
            "openai_api_key": "unit-test-api-key-never-log",
            "openai_report_model": "unit-test-report-model",
            "openai_report_include_images": True,
            "openai_report_max_attempts": 3,
            "openai_report_timeout_seconds": 30.0,
            "openai_report_full_photo_timeout_seconds": 240.0,
            "openai_report_retry_max_delay_seconds": 30.0,
        }
        for name, value in settings.items():
            active = patch.object(service.settings, name, value)
            active.start()
            self.addCleanup(active.stop)
        for active in (
            patch.object(service.asyncio, "sleep", self.sleep),
            patch.object(service.random, "uniform", return_value=0.0),
        ):
            active.start()
            self.addCleanup(active.stop)

    async def _request(self, **overrides):
        arguments = {
            "system_prompt": "Return the required JSON object.",
            "user_payload": {"private_input": "unit-test-patient-detail-never-log"},
            "schema_name": "unit_test_report",
            "json_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        }
        arguments.update(overrides)
        return await service._create_structured_response(**arguments)

    async def test_server_error_retries_without_changing_model_images_or_retention(self) -> None:
        client = _FakeClient([
            _response(500, {"error": {"code": "server_error", "message": "private-provider-body"}}),
            _response(),
        ])
        image = b"\x89PNG\r\n\x1a\nunit-test-image"
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client),
            self.assertLogs(service.logger, level="WARNING") as captured,
        ):
            result = await self._request(images=[("LEFT", image)], image_detail="high")
        self.assertEqual(result, {"message": "완료"})
        self.assertEqual(len(client.requests), 2)
        first, second = client.requests
        self.assertEqual(first, second)
        self.assertEqual(first["url"], service.OPENAI_RESPONSES_URL)
        body = first["body"]
        self.assertEqual(body["model"], "unit-test-report-model")
        self.assertIs(body["store"], False)
        self.assertIs(body["text"]["format"]["strict"], True)
        image_content = body["input"][1]["content"][-1]
        self.assertEqual(image_content["detail"], "high")
        self.assertEqual(image_content["image_url"], service._image_data_url(image))
        self.sleep.assert_awaited_once_with(1.0)
        logged = "\n".join(captured.output)
        self.assertIn("API_HTTP_ERROR", logged)
        for private_value in (
            "unit-test-api-key-never-log", "unit-test-patient-detail-never-log",
            "private-provider-body", image_content["image_url"],
        ):
            self.assertNotIn(private_value, logged)
        self.assertTrue(client.exited)

    async def test_persistent_server_failure_stops_after_three_attempts(self) -> None:
        client = _FakeClient([_response(503, {"error": {"code": "server_error"}})] * 3)
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(service.ReportGenerationError) as raised:
                await self._request()
        self.assertEqual(raised.exception.code, "API_HTTP_ERROR")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual([call.args[0] for call in self.sleep.await_args_list], [1.0, 2.0])

    async def test_retryable_http_statuses_recover(self) -> None:
        for status in (408, 409, 429, 502, 504):
            with self.subTest(status=status):
                self.sleep.reset_mock()
                client = _FakeClient([_response(status, {"error": {"code": "temporary"}}), _response()])
                with patch.object(service.httpx, "AsyncClient", return_value=client):
                    self.assertEqual(await self._request(), {"message": "완료"})
                self.assertEqual(len(client.requests), 2)
                self.sleep.assert_awaited_once_with(1.0)

    async def test_network_protocol_and_read_timeout_errors_recover(self) -> None:
        for failure in (
            httpx.ConnectError("private-network-detail"),
            httpx.ReadTimeout("private-timeout-detail"),
            httpx.RemoteProtocolError("private-protocol-detail"),
        ):
            with self.subTest(error_type=type(failure).__name__):
                self.sleep.reset_mock()
                client = _FakeClient([failure, _response()])
                with patch.object(service.httpx, "AsyncClient", return_value=client):
                    self.assertEqual(await self._request(), {"message": "완료"})
                self.assertEqual(len(client.requests), 2)
                self.sleep.assert_awaited_once_with(1.0)

    async def test_auth_and_quota_failures_are_not_retried(self) -> None:
        for status, code in ((401, "invalid_api_key"), (429, "insufficient_quota"),
                             (429, "billing_hard_limit_reached")):
            with self.subTest(status=status, provider_code=code):
                client = _FakeClient([_response(status, {"error": {"code": code}})])
                with patch.object(service.httpx, "AsyncClient", return_value=client):
                    with self.assertRaises(service.ReportGenerationError) as raised:
                        await self._request()
                self.assertFalse(raised.exception.retryable)
                self.assertEqual(raised.exception.status_code, status)
                self.assertEqual(len(client.requests), 1)
        self.sleep.assert_not_awaited()

    async def test_rate_limit_obeys_retry_after_seconds(self) -> None:
        client = _FakeClient([
            _response(429, {"error": {"code": "rate_limit_exceeded"}}, headers={"Retry-After": "7"}),
            _response(),
        ])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            await self._request()
        self.sleep.assert_awaited_once_with(7.0)
        self.assertEqual(len(client.requests), 2)

    async def test_rate_limit_obeys_retry_after_http_date(self) -> None:
        when = datetime.now(timezone.utc) + timedelta(seconds=10)
        client = _FakeClient([
            _response(429, {"error": {"code": "rate_limit_exceeded"}},
                      headers={"Retry-After": format_datetime(when, usegmt=True)}),
            _response(),
        ])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            await self._request()
        self.sleep.assert_awaited_once()
        delay = self.sleep.await_args.args[0]
        self.assertGreater(delay, 8.0)
        self.assertLessEqual(delay, 10.0)

    async def test_excessive_retry_after_fails_instead_of_retrying_too_early(self) -> None:
        for wait, budget in ((45, 60.0), (7, 5.0)):
            with self.subTest(retry_after=wait, budget=budget):
                client = _FakeClient([
                    _response(429, {"error": {"code": "rate_limit_exceeded"}},
                              headers={"Retry-After": str(wait)}),
                ])
                with (
                    patch.object(service.httpx, "AsyncClient", return_value=client),
                    patch.object(service.settings, "openai_report_timeout_seconds", budget),
                ):
                    with self.assertRaises(service.ReportGenerationError):
                        await self._request()
                self.assertEqual(len(client.requests), 1)
        self.sleep.assert_not_awaited()

    async def test_invalid_api_json_output_json_and_empty_output_can_recover(self) -> None:
        bad_responses = (
            httpx.Response(200, content=b"not API JSON"),
            _response(body={"status": "completed", "output_text": "not output JSON"}),
            _response(body={"status": "completed", "output": []}),
            _response(body={"status": "completed", "output_text": "[]"}),
        )
        for bad in bad_responses:
            with self.subTest(response=bad.content[:65]):
                client = _FakeClient([bad, _response()])
                with patch.object(service.httpx, "AsyncClient", return_value=client):
                    self.assertEqual(await self._request(), {"message": "완료"})
                self.assertEqual(len(client.requests), 2)

    async def test_token_truncation_increases_only_token_budget_up_to_cap(self) -> None:
        incomplete = _response(body={
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": '{"message":',
        })
        client = _FakeClient([incomplete, incomplete, _response()])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await self._request(max_output_tokens=2400), {"message": "완료"})
        self.assertEqual([item["body"]["max_output_tokens"] for item in client.requests],
                         [2400, 4800, 6000])
        bodies = [dict(item["body"]) for item in client.requests]
        for body in bodies:
            body.pop("max_output_tokens")
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0], bodies[2])

    async def test_refusal_and_content_filter_are_not_retried(self) -> None:
        cases = (
            ({"status": "completed", "output": [{"content": [
                {"type": "refusal", "refusal": "private-refusal-text"}
            ]}]}, "MODEL_REFUSAL"),
            ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
             "INCOMPLETE_OUTPUT"),
        )
        for body, code in cases:
            with self.subTest(code=code):
                client = _FakeClient([_response(body=body)])
                with patch.object(service.httpx, "AsyncClient", return_value=client):
                    with self.assertRaises(service.ReportGenerationError) as raised:
                        await self._request()
                self.assertEqual(raised.exception.code, code)
                self.assertFalse(raised.exception.retryable)
                self.assertNotIn("private-refusal-text", str(raised.exception))
                self.assertEqual(len(client.requests), 1)
        self.sleep.assert_not_awaited()

    async def test_full_photo_request_uses_finite_overall_deadline(self) -> None:
        client = _FakeClient([_response()])
        original_wait_for = asyncio.wait_for
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client) as factory,
            patch.object(service.asyncio, "wait_for", wraps=original_wait_for) as wait,
        ):
            await self._request(wait_for_response_without_timeout=True)
        wait.assert_called_once()
        self.assertGreater(wait.call_args.kwargs["timeout"], 230.0)
        self.assertLessEqual(wait.call_args.kwargs["timeout"], 240.0)
        self.assertIsNone(factory.call_args.kwargs["timeout"].read)

    async def test_exhausted_deadline_does_not_start_another_attempt(self) -> None:
        client = _FakeClient([_response(500, {"error": {"code": "server_error"}})])
        clock = SimpleNamespace(monotonic=Mock(side_effect=[100.0, 100.0, 101.0, 104.0]))
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client),
            patch.object(service, "time", clock),
            patch.object(service.settings, "openai_report_timeout_seconds", 3.0),
        ):
            with self.assertRaises(service.ReportGenerationError) as raised:
                await self._request()
        self.assertEqual(raised.exception.code, "API_TIMEOUT")
        self.assertEqual(len(client.requests), 1)
        self.sleep.assert_awaited_once_with(1.0)

    async def test_overall_deadline_cancels_hung_post_without_retry(self) -> None:
        cancelled = asyncio.Event()

        async def hung_post():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        client = _FakeClient([hung_post])
        with (
            patch.object(service.httpx, "AsyncClient", return_value=client),
            patch.object(service.settings, "openai_report_full_photo_timeout_seconds", 0.01),
        ):
            with self.assertRaises(service.ReportGenerationError) as raised:
                await self._request(wait_for_response_without_timeout=True)
        self.assertEqual(raised.exception.code, "API_TIMEOUT")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(client.requests), 1)
        self.sleep.assert_not_awaited()

    async def test_caller_cancellation_propagates_without_retry_or_result(self) -> None:
        client = _FakeClient([asyncio.CancelledError()])
        with patch.object(service.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(asyncio.CancelledError):
                await self._request()
        self.assertEqual(len(client.requests), 1)
        self.assertTrue(client.exited)
        self.sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
