from __future__ import annotations

import base64
import json
import re
import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service
from tests.photo_image_fixtures import encoded_photo


JPEG_BYTES = encoded_photo(".jpg", (30, 70, 160))
PNG_BYTES = encoded_photo(".png", (160, 70, 30))


def _facts(*, regions: int = 2, ratio: float = 0.42, status: str = "refined") -> dict:
    return {
        "region_count": regions,
        "contact_area_ratio": ratio,
        "status": status,
    }


_PLANTAR_SHAPE_TERMS = ("요족", "정상", "평발")
_PLANTAR_FORBIDDEN_TERMS = (
    "압력",
    "하중",
    "체중",
    "착지",
    "접촉 분포",
    "접촉 면적",
    "영역 수",
    "균형",
    "과내전",
    "과외전",
    "진단",
)
_PLANTAR_RESULT_PATTERN = re.compile(
    r"정상\s*형태(?:이며|이고|입니다)|"
    r"(?:요족|평발)\s*경향(?:이며|이고|이\s*(?:보입니다|있습니다))|"
    r"판단하기\s*어렵습니다"
)
_PLANTAR_HEDGE_PATTERN = re.compile(r"수\s*있|가능성")


def _assert_plantar_text_contract(
    testcase: unittest.TestCase,
    text: str,
) -> None:
    testcase.assertEqual(len(text.splitlines()), 1)
    testcase.assertEqual(sum(text.count(mark) for mark in ".!?"), 1)
    testcase.assertLessEqual(len(text), 180)
    testcase.assertTrue(text.endswith((".", "!", "?")))
    testcase.assertTrue(any(term in text for term in _PLANTAR_SHAPE_TERMS))
    testcase.assertRegex(text, _PLANTAR_RESULT_PATTERN)
    testcase.assertNotRegex(text, _PLANTAR_HEDGE_PATTERN)
    for forbidden in _PLANTAR_FORBIDDEN_TERMS:
        testcase.assertNotIn(forbidden, text)


class PlantarFootprintFallbackTests(unittest.TestCase):
    def test_fallback_is_one_explicit_abstention_sentence(self) -> None:
        result = service.build_fallback_plantar_footprint_analysis_text(
            left_segmentation_facts=_facts(ratio=0.41),
            right_segmentation_facts=_facts(ratio=0.43),
        )

        _assert_plantar_text_contract(self, result)
        self.assertIn("요족·정상·평발", result)
        self.assertEqual(
            result,
            "발바닥 형태는 요족·정상·평발 중 어느 경향인지 판단하기 어렵습니다.",
        )

    def test_segmentation_ratios_do_not_become_pressure_or_numeric_claims(self) -> None:
        result = service.build_fallback_plantar_footprint_analysis_text(
            left_segmentation_facts=_facts(ratio=42.0),
            right_segmentation_facts=_facts(ratio=38.0),
        )

        _assert_plantar_text_contract(self, result)
        self.assertNotRegex(result, r"\d")


class PlantarFootprintGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_openai_returns_fallback_without_request(self) -> None:
        create_response = AsyncMock()
        with (
            patch.object(service, "_openai_enabled", return_value=False),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=JPEG_BYTES,
                right_raw_image_bytes=PNG_BYTES,
                left_segmentation_facts=_facts(ratio=0.50),
                right_segmentation_facts=_facts(ratio=0.30),
            )

        create_response.assert_not_awaited()
        _assert_plantar_text_contract(self, result)

    async def test_sends_one_raw_photo_composite_and_allowlisted_facts(self) -> None:
        generated = "왼발은 정상 형태이며 오른발은 평발 경향이 보입니다."
        create_response = AsyncMock(
            return_value={"plantar_footprint_analysis_text": generated}
        )
        left_facts = {**_facts(ratio=0.52), "segmentation_png": "must-not-leak"}
        right_facts = {**_facts(ratio=0.37), "private_note": "must-not-leak"}

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=JPEG_BYTES,
                right_raw_image_bytes=PNG_BYTES,
                left_segmentation_facts=left_facts,
                right_segmentation_facts=right_facts,
            )

        self.assertEqual(result, generated)
        _assert_plantar_text_contract(self, result)
        create_response.assert_awaited_once()
        request = create_response.await_args.kwargs
        self.assertEqual(request["image_detail"], "high")
        self.assertIn("합본 한 장만", request["user_payload"]["image_contract"])
        self.assertIn("발등 사진, 압력 히트맵", request["system_prompt"])
        self.assertEqual(
            request["images"],
            [
                (service.PLANTAR_COMPOSITE_LABEL, service.combine_raw_plantar_photos(JPEG_BYTES, PNG_BYTES)),
            ],
        )
        facts_text = request["user_payload"]["deterministic_segmentation_facts_text"]
        self.assertIn("LEFT: region_count=2", facts_text)
        self.assertIn("RIGHT: region_count=2", facts_text)
        self.assertNotIn("must-not-leak", facts_text)
        self.assertNotIn(
            "segmentation_png",
            json.dumps(request["user_payload"], ensure_ascii=False),
        )
        self.assertTrue(request["json_schema"]["additionalProperties"] is False)
        self.assertIn("요족·정상·평발", request["system_prompt"])
        self.assertIn("한 문장", request["system_prompt"])
        self.assertIn("양발은 정상 형태입니다.", request["system_prompt"])
        self.assertIn("경향이 보입니다", request["system_prompt"])
        self.assertIn("'수 있습니다'와 '가능성'은 사용하지 않는다", request["system_prompt"])
        self.assertIn("판단하기 어렵습니다", request["system_prompt"])
        self.assertIn("압력, 하중, 체중 분포", request["system_prompt"])
        self.assertIn("과내전, 과외전", request["system_prompt"])
        self.assertIn("안내 문장도 작성하지 않는다", request["system_prompt"])
        policy = request["user_payload"]["plantar_shape_assessment_policy"]
        self.assertIn("같은 범주로 설명한다", policy["bilateral_consistency"])
        self.assertIn("명확히 확인되는 형태 차이", policy["separate_categories"])
        self.assertIn("뚜렷한 형태 근거가 있을 때만", policy["high_arch_evidence"])
        self.assertIn("뚜렷한 형태 근거가 있을 때만", policy["low_arch_evidence"])
        self.assertIn("요족이 아니라는 이유만으로 평발을 선택하지 않는다", policy["low_arch_evidence"])
        self.assertIn("정상 형태를 유지한다", policy["normal_or_insufficient_evidence"])
        self.assertIn("영역 수나 전체 면적 비율만으로", policy["normal_or_insufficient_evidence"])
        for rule in policy.values():
            self.assertIn(rule, request["system_prompt"])
            self.assertIn(
                rule,
                request["json_schema"]["properties"]["plantar_footprint_analysis_text"]["description"],
            )
        self.assertEqual(
            request["user_payload"]["output_contract"],
            "근거가 있으면 정상 형태 또는 요족·평발 경향을 관찰형으로 쓰고, "
            "근거가 부족하면 지정된 판단 보류 문장을 쓰는 한국어 한 문장을 작성한다.",
        )

    async def test_forbidden_or_malformed_output_is_replaced_by_safe_fallback(self) -> None:
        invalid_outputs = (
            "왼발은 평발입니다.",
            "왼발은 평발 경향일 수 있습니다.",
            "왼발은 평발 가능성이 있습니다.",
            "압력 분포상 평발 경향이 보입니다.",
            "하중이 고르게 실리는 정상 형태입니다.",
            "체중이 바깥쪽으로 실려 요족 경향이 보입니다.",
            "착지 부위가 넓어 평발 경향이 보입니다.",
            "접촉 분포와 영역 수를 보면 정상 형태입니다.",
            "평발 경향이 보이지만 의학적 진단이 아닙니다.",
            "과내전으로 평발 경향이 보입니다.",
            "과외전으로 요족 경향이 보입니다.",
            "**평발 경향이 보입니다.**",
            "발바닥 형태를 판단하기 어렵습니다.",
        )
        for invalid in invalid_outputs:
            with self.subTest(invalid=invalid):
                create_response = AsyncMock(
                    return_value={"plantar_footprint_analysis_text": invalid}
                )
                with (
                    patch.object(service, "_openai_enabled", return_value=True),
                    patch.object(service.settings, "openai_report_include_images", True),
                    patch.object(service, "_create_structured_response", create_response),
                ):
                    result = await service.generate_plantar_footprint_analysis_text(
                        left_raw_image_bytes=JPEG_BYTES,
                        right_raw_image_bytes=PNG_BYTES,
                        left_segmentation_facts=_facts(ratio=0.40),
                        right_segmentation_facts=_facts(ratio=0.40),
                    )

                create_response.assert_awaited_once()
                _assert_plantar_text_contract(self, result)
                self.assertNotEqual(result, invalid)

    async def test_api_failure_uses_fallback_after_one_request(self) -> None:
        create_response = AsyncMock(side_effect=RuntimeError("offline"))
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=JPEG_BYTES,
                right_raw_image_bytes=PNG_BYTES,
                left_segmentation_facts=_facts(),
                right_segmentation_facts=_facts(),
            )

        create_response.assert_awaited_once()
        _assert_plantar_text_contract(self, result)

    async def test_invalid_raw_image_mime_uses_fallback_without_http_helper(self) -> None:
        create_response = AsyncMock()
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=b"not-an-image",
                right_raw_image_bytes=PNG_BYTES,
                left_segmentation_facts=_facts(),
                right_segmentation_facts=_facts(),
            )

        create_response.assert_not_awaited()
        _assert_plantar_text_contract(self, result)

    async def test_truncated_png_uses_fallback_without_sending_a_partial_composite(self) -> None:
        create_response = AsyncMock()
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=b"\x89PNG\r\n\x1a\ntruncated",
                right_raw_image_bytes=PNG_BYTES,
                left_segmentation_facts=_facts(),
                right_segmentation_facts=_facts(),
            )
        create_response.assert_not_awaited()
        self.assertEqual(result, service._PLANTAR_ABSTENTION_TEXT)
        _assert_plantar_text_contract(self, result)


class _FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {"output_text": '{"value":"ok"}', "status": "completed"}


class _CapturingAsyncClient:
    last_request: dict | None = None
    last_timeout: object | None = None

    def __init__(self, *, timeout: object, trust_env: bool = False):
        self.timeout = timeout
        type(self).last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url: str, *, headers: dict, json: dict):
        type(self).last_request = {"url": url, "headers": headers, "json": json}
        return _FakeResponse()


class StructuredResponseImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_jpeg_magic_is_preserved_in_exact_data_url_and_high_detail(self) -> None:
        expected_url = "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode("ascii")
        with (
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service.settings, "openai_api_key", "test-key"),
            patch.object(service.settings, "openai_report_timeout_seconds", 180.0),
            patch.object(service.httpx, "AsyncClient", _CapturingAsyncClient),
        ):
            result = await service._create_structured_response(
                system_prompt="system",
                user_payload={"facts": "text-only"},
                schema_name="test_schema",
                json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                images=[("LEFT RAW PRE-SEGMENTATION", JPEG_BYTES)],
                image_detail="high",
            )

        self.assertEqual(result, {"value": "ok"})
        timeout = _CapturingAsyncClient.last_timeout
        self.assertIsInstance(timeout, service.httpx.Timeout)
        self.assertEqual(timeout.connect, 10.0)
        self.assertEqual(timeout.read, 180.0)
        self.assertEqual(timeout.write, 180.0)
        self.assertEqual(timeout.pool, 10.0)
        body = _CapturingAsyncClient.last_request["json"]
        image_inputs = [
            item
            for item in body["input"][1]["content"]
            if item["type"] == "input_image"
        ]
        self.assertEqual(image_inputs, [{"type": "input_image", "image_url": expected_url, "detail": "high"}])
        self.assertIs(body["store"], False)

    async def test_existing_helper_image_detail_defaults_to_low(self) -> None:
        with (
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service.settings, "openai_api_key", "test-key"),
            patch.object(service.httpx, "AsyncClient", _CapturingAsyncClient),
        ):
            await service._create_structured_response(
                system_prompt="system",
                user_payload={},
                schema_name="test_schema",
                json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                images=[("generated PNG", PNG_BYTES)],
            )

        body = _CapturingAsyncClient.last_request["json"]
        image_input = next(
            item
            for item in body["input"][1]["content"]
            if item["type"] == "input_image"
        )
        self.assertEqual(image_input["detail"], "low")
        self.assertTrue(image_input["image_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
