from __future__ import annotations

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service


def setUpModule() -> None:
    enabled = patch.object(service.settings, "tinea_inflammation_enabled", True)
    enabled.start()
    unittest.addModuleCleanup(enabled.stop)
from tests.test_full_photo_analysis_texts import (
    JPEG_BYTES, LEFT_PNG_BYTES, RIGHT_WEBP_BYTES, _facts, _valid_generated,
)


def _response_data(result: service.FullPhotoAnalysisTexts) -> dict:
    fields = (
        "hallux_score_analysis_text", "fungal_suspicion_safety_description",
        "skin_reaction_safety_description", "total_score_description",
        "plantar_footprint_analysis_text",
    )
    return {
        **{field: getattr(result, field) for field in fields},
        "metric_advice": {
            key: {"observation": value[0], "practical_tip": value[1]}
            for key, value in result.metric_advice.items()
        },
    }


class FactBasedPhotoFallbackTests(unittest.TestCase):
    def test_safe_scores_do_not_invent_skin_findings(self) -> None:
        facts = _facts()
        facts["tinea_pedis"]["final_scores"] = {
            "fungal_suspicion_safety_score": 100,
            "skin_reaction_safety_score": 95,
        }
        result = service.build_fallback_full_photo_analysis_texts(facts=facts)
        for text in (result.fungal_suspicion_safety_description,
                     result.skin_reaction_safety_description,
                     result.total_score_description):
            self.assertIn("관찰되지", text)
        service._validate_full_photo_response_data(_response_data(result))

    def test_low_safety_scores_have_signal_not_reversed_risk(self) -> None:
        facts = _facts()
        facts["tinea_pedis"]["final_scores"] = {
            "fungal_suspicion_safety_score": 20,
            "skin_reaction_safety_score": 30,
        }
        original = copy.deepcopy(facts)
        result = service.build_fallback_full_photo_analysis_texts(facts=facts)
        self.assertIn("신호가", result.fungal_suspicion_safety_description)
        self.assertIn("관찰됩니다", result.skin_reaction_safety_description)
        self.assertNotIn("관찰되지", result.total_score_description)
        self.assertIn("함께 관찰", result.total_score_description)
        self.assertEqual(facts, original)
        service._validate_full_photo_response_data(_response_data(result))

    def test_missing_or_disabled_skin_analysis_does_not_claim_clear_skin(self) -> None:
        disabled = _facts()
        for surface in ("dorsal", "plantar"):
            for side in ("left", "right"):
                disabled["tinea_pedis"][surface][side]["analysis_mode"] = "disabled_forced_clear"
        for facts in ({}, disabled):
            with self.subTest(facts=facts):
                result = service.build_fallback_full_photo_analysis_texts(facts=facts)
                self.assertIn("근거가 충분하지", result.fungal_suspicion_safety_description)
                self.assertIn("근거가 충분하지", result.skin_reaction_safety_description)
                self.assertNotIn("관찰되지", result.total_score_description)
                service._validate_full_photo_response_data(_response_data(result))

    def test_hallux_side_comparison_uses_actual_angles(self) -> None:
        for left, right, expected in (
            (12, 18, "오른발에서"), (22, 10, "왼발에서"),
            (12, 12.5, "비슷하게"), (None, 18, "근거가 충분하지"),
        ):
            with self.subTest(left=left, right=right):
                facts = _facts()
                facts["hallux_valgus"].update(left_angle_degree=left, right_angle_degree=right)
                result = service.build_fallback_full_photo_analysis_texts(facts=facts)
                self.assertIn(expected, result.hallux_score_analysis_text)
                self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
                self.assertEqual(result.metric_advice["HALLUX_VALGUS"][0], result.hallux_score_analysis_text)
                service._validate_full_photo_response_data(_response_data(result))

    def test_contact_mask_statistics_do_not_fabricate_arch_classification(self) -> None:
        result = service.build_fallback_full_photo_analysis_texts(facts=_facts())
        self.assertIn("판단하기 어렵", result.plantar_footprint_analysis_text)
        self.assertNotRegex(result.plantar_footprint_analysis_text, "압력|하중|접촉")


class ReportFallbackRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for patcher in (
            patch.object(service.settings, "openai_report_fallback_enabled", True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service.settings, "openai_report_max_attempts", 5),
            patch.object(service, "_openai_enabled", return_value=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _generate_photo(self, **overrides):
        kwargs = {
            "dorsal_raw_image_bytes": JPEG_BYTES,
            "left_plantar_raw_image_bytes": LEFT_PNG_BYTES,
            "right_plantar_raw_image_bytes": RIGHT_WEBP_BYTES,
            "facts": _facts(), "related_analysis_images": {},
        }
        kwargs.update(overrides)
        return await service.generate_full_photo_analysis_texts(**kwargs)

    async def test_exhausted_provider_failure_returns_fact_based_text_with_safe_source(self) -> None:
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock) as create:
            create.side_effect = service.ReportGenerationError("API_HTTP_ERROR", status_code=503)
            result = await self._generate_photo()
        self.assertEqual(create.await_count, 1)
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        self.assertEqual(result.generation_failure_code, "API_HTTP_ERROR")
        self.assertIn("오른발에서", result.hallux_score_analysis_text)
        service._validate_full_photo_response_data(_response_data(result))

    async def test_invalid_generated_text_exhausts_five_attempts_then_complete_fallback(self) -> None:
        bad = _valid_generated()
        bad["skin_reaction_safety_description"] = "invalid 999"
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock, return_value=bad) as create:
            result = await self._generate_photo()
        self.assertEqual(create.await_count, 5)
        repairs = [
            call.kwargs["user_payload"]["validation_retry"]
            for call in create.await_args_list[1:]
        ]
        self.assertEqual([repair["attempt"] for repair in repairs], [2, 3, 4, 5])
        self.assertTrue(all(repair["maximum_attempts"] == 5 for repair in repairs))
        self.assertEqual(result.generation_failure_code, "OUTPUT_VALIDATION_FAILED")
        self.assertNotIn("999", repr(result))
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        service._validate_full_photo_response_data(_response_data(result))

    async def test_successful_gpt_keeps_gpt_text_and_source(self) -> None:
        data = _valid_generated()
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock, return_value=data):
            result = await self._generate_photo()
        self.assertEqual(result.generation_source, "OPENAI")
        self.assertIsNone(result.generation_failure_code)
        self.assertEqual(_response_data(result), data)

    async def test_disabled_provider_or_images_uses_distinct_failure_code(self) -> None:
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock) as create:
            with patch.object(service, "_openai_enabled", return_value=False):
                result = await self._generate_photo()
                self.assertEqual(result.generation_failure_code, "OPENAI_DISABLED")
            with patch.object(service.settings, "openai_report_include_images", False):
                result = await self._generate_photo()
                self.assertEqual(result.generation_failure_code, "IMAGE_INPUT_DISABLED")
            create.assert_not_awaited()

    async def test_invalid_photo_inputs_are_not_converted_into_success(self) -> None:
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock) as create:
            for overrides in (
                {"facts": {"score": float("nan")}},
                {"facts": {"image": b"not-json"}},
                {"dorsal_raw_image_bytes": b"corrupt"},
            ):
                with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                    await self._generate_photo(**overrides)
            create.assert_not_awaited()

    async def test_cancellation_is_not_swallowed_by_fallback(self) -> None:
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self._generate_photo()

    async def test_pressure_fallback_preserves_measured_score_and_side(self) -> None:
        left, right = [7.0] + [0.0] * 11, [0.0] * 11 + [3.0]
        facts = service.build_pressure_balance_facts(left, right)
        with patch.object(service, "_create_structured_response", new_callable=AsyncMock, side_effect=service.ReportGenerationError("API_TIMEOUT")):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left, right_pressure_values=right, pressure_facts=facts,
                left_heatmap_png=LEFT_PNG_BYTES, right_heatmap_png=LEFT_PNG_BYTES,
            )
        self.assertEqual(result.score, facts["balanceScore"])
        self.assertEqual(result.generation_failure_code, "API_TIMEOUT")
        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
        self.assertIn("왼발", result.overall_balance_observation)

    async def test_invalid_derived_heatmap_uses_only_valid_sensor_facts(self) -> None:
        values = [1.0] * 12
        facts = service.build_pressure_balance_facts(values, values)
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(service, "_openai_enabled", return_value=enabled):
                with patch.object(service, "_create_structured_response", new_callable=AsyncMock) as create:
                    result = await service.generate_pressure_balance_report(
                        left_pressure_values=values, right_pressure_values=values,
                        pressure_facts=facts,
                        left_heatmap_png=JPEG_BYTES, right_heatmap_png=LEFT_PNG_BYTES,
                    )
                    create.assert_not_awaited()
                self.assertEqual(result.score, facts["balanceScore"])
                self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
                self.assertEqual(result.generation_failure_code, "INVALID_DERIVED_IMAGE" if enabled else "OPENAI_DISABLED")
                with self.assertRaises(ValueError):
                    await service.generate_pressure_balance_report(
                        left_pressure_values=values, right_pressure_values=values,
                        pressure_facts={**facts, "balanceScore": 0},
                        left_heatmap_png=JPEG_BYTES, right_heatmap_png=LEFT_PNG_BYTES,
                    )


if __name__ == "__main__":
    unittest.main()
