from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service
from tests.photo_image_fixtures import encoded_photo


def setUpModule() -> None:
    enabled = patch.object(service.settings, "tinea_inflammation_enabled", True)
    enabled.start()
    unittest.addModuleCleanup(enabled.stop)


JPEG_BYTES = b"\xff\xd8\xff\xe0dorsal-jpeg"
LEFT_PNG_BYTES = encoded_photo(".png", (30, 70, 160))
RIGHT_WEBP_BYTES = encoded_photo(".webp", (160, 70, 30))
ANALYSIS_PNG_BYTES = b"\x89PNG\r\n\x1a\nanalysis-result"


def _facts() -> dict:
    return {
        "hallux_valgus": {
            "left_angle_degree": 12.4,
            "right_angle_degree": 18.7,
            "safety_score": 83,
        },
        "metrics": {
            "measured_left_foot_size_mm": 253,
            "measured_right_foot_size_mm": 248,
            "left_foot_width_mm": 85,
            "right_foot_width_mm": 70,
        },
        "tinea_pedis": {
            "dorsal": {
                "left": {
                    "fungal_suspicion_safety_score": 91,
                    "skin_reaction_safety_score": 88,
                },
                "right": {
                    "fungal_suspicion_safety_score": 93,
                    "skin_reaction_safety_score": 90,
                },
            },
            "plantar": {
                "left": {
                    "fungal_suspicion_safety_score": 86,
                    "skin_reaction_safety_score": 84,
                },
                "right": {
                    "fungal_suspicion_safety_score": 89,
                    "skin_reaction_safety_score": 87,
                },
            },
            "final_scores": {
                "fungal_suspicion_safety_score": 90,
                "skin_reaction_safety_score": 87,
            },
        },
        "plantar_footprint": {
            "left": {
                "region_count": 2,
                "contact_area_ratio": 0.42,
                "status": "refined",
            },
            "right": {
                "region_count": 3,
                "contact_area_ratio": 0.38,
                "status": "refined",
            },
        },
    }


def _valid_generated() -> dict:
    return {
        "hallux_score_analysis_text": (
            "발가락 정렬에서 좌우 기울기 차이가 관찰되며 한쪽에 변화 경향이 보입니다."
        ),
        "fungal_suspicion_safety_description": (
            "양발 발등 일부에서 피부색과 각질 변화의 의심 경향이 보입니다."
        ),
        "skin_reaction_safety_description": (
            "일부 피부에서 자극처럼 보이는 색 변화 경향이 관찰됩니다."
        ),
        "total_score_description": (
            "발가락 정렬과 피부 및 접촉 분포를 함께 살펴보는 경향 안내입니다."
        ),
        "plantar_footprint_analysis_text": (
            "왼발은 정상 형태이며 오른발은 평발 경향이 보입니다."
        ),
        "metric_advice": {
            "HALLUX_VALGUS": {
                "observation": "발가락 정렬에서 좌우 차이가 관찰됩니다.",
                "practical_tip": "발 앞부분을 압박하지 않는 신발을 선택해 주세요.",
            },
            "ATHLETES_FOOT": {
                "observation": "발 피부의 색과 각질 변화 경향을 관찰할 필요가 있습니다.",
                "practical_tip": "발을 청결하고 건조하게 관리해 주세요.",
            },
            "SKIN_IRRITATION": {
                "observation": "피부 자극처럼 보이는 변화 경향이 관찰됩니다.",
                "practical_tip": "마찰을 줄이고 보습 상태를 유지해 주세요.",
            },
        },
    }


class FullPhotoAnalysisFallbackTests(unittest.TestCase):
    def test_fallback_has_exact_fields_and_two_part_metric_advice(self) -> None:
        result = service.build_fallback_full_photo_analysis_texts(facts=_facts())

        self.assertIsInstance(result, service.FullPhotoAnalysisTexts)
        self.assertEqual(
            set(result.metric_advice),
            {"HALLUX_VALGUS", "ATHLETES_FOOT", "SKIN_IRRITATION"},
        )
        self.assertTrue(all(len(value) == 2 for value in result.metric_advice.values()))
        self.assertEqual(len(result.plantar_footprint_analysis_text.splitlines()), 1)
        self.assertIn("요족·정상·평발", result.plantar_footprint_analysis_text)
        self.assertIn("판단하기 어렵습니다", result.plantar_footprint_analysis_text)
        self.assertNotRegex(result.plantar_footprint_analysis_text, r"수\s*있|가능성")
        self.assertNotIn("진단", result.plantar_footprint_analysis_text)
        self.assertNotRegex(result.plantar_footprint_analysis_text, r"압력|하중")
        self.assertNotIn("확진", repr(result))


        self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")


class FullPhotoPlantarTextValidationTests(unittest.TestCase):
    def test_one_shape_result_sentence_is_normalized_without_disclaimer(self) -> None:
        result = service._require_full_photo_plantar_footprint_analysis_text(
            "  왼발과   오른발은 정상 형태입니다  "
        )

        self.assertEqual(
            result,
            "왼발과 오른발은 정상 형태입니다.",
        )
        self.assertNotIn("진단", result)

    def test_left_and_right_may_differ_within_one_result_sentence(self) -> None:
        result = service._require_full_photo_plantar_footprint_analysis_text(
            "왼발은 정상 형태이며 오른발은 평발 경향이 보입니다."
        )

        self.assertEqual(len(result.splitlines()), 1)
        self.assertIn("왼발은 정상", result)
        self.assertIn("오른발은 평발", result)

    def test_tendency_is_present_wording_is_already_sufficiently_uncertain(self) -> None:
        result = service._require_full_photo_plantar_footprint_analysis_text(
            "왼발은 요족 경향이 있으며 오른발은 평발 경향이 있습니다."
        )

        self.assertEqual(
            result,
            "왼발은 요족 경향이 있으며 오른발은 평발 경향이 있습니다.",
        )
        self.assertNotRegex(result, r"수\s*있|가능성")

    def test_abstention_sentence_is_accepted_when_shape_is_unclear(self) -> None:
        result = service._require_full_photo_plantar_footprint_analysis_text(
            "발바닥 형태는 요족·정상·평발 중 어느 경향인지 판단하기 어렵습니다."
        )

        self.assertEqual(
            result,
            "발바닥 형태는 요족·정상·평발 중 어느 경향인지 판단하기 어렵습니다.",
        )

    def test_unsafe_values_report_specific_rule_codes(self) -> None:
        invalid_values = (
            (
                "왼발은 평발입니다.",
                "DIAGNOSTIC_OR_CERTAINTY_NOT_ALLOWED",
            ),
            ("발바닥 형태는 평발 경향일 가능성이 2배 높습니다.", "DIGITS_NOT_ALLOWED"),
            (
                "왼발은 평발 경향일 수 있습니다.",
                "PLANTAR_HEDGE_LANGUAGE_NOT_ALLOWED",
            ),
            (
                "오른발은 평발 가능성이 있습니다.",
                "PLANTAR_HEDGE_LANGUAGE_NOT_ALLOWED",
            ),
            (
                "왼발은 정상 경향일 수 있습니다. 오른발은 평발 경향일 가능성이 있습니다.",
                "SENTENCE_COUNT_OUT_OF_RANGE",
            ),
            (
                "발바닥 형태는 정상 형태이지만 이 내용은 의학적 진단이 아닙니다.",
                "PLANTAR_DISCLAIMER_NOT_ALLOWED",
            ),
            (
                "발바닥 압력 분포에서 평발 경향이 보입니다.",
                "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
            ),
            (
                "발바닥 하중 형태에서 요족 경향이 보입니다.",
                "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
            ),
            (
                "발바닥 착지 부위가 넓어 평발 경향이 보입니다.",
                "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
            ),
            (
                "발바닥 접촉 분포와 영역 수를 보면 정상 형태입니다.",
                "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
            ),
            (
                "발바닥 형태는 과내전 경향이 보입니다.",
                "PLANTAR_ARCH_CATEGORY_NOT_ALLOWED",
            ),
            (
                "발바닥 형태는 과외전 경향이 보입니다.",
                "PLANTAR_ARCH_CATEGORY_NOT_ALLOWED",
            ),
            (
                "발바닥 형태는 좌우가 비슷해 보입니다.",
                "PLANTAR_ARCH_CATEGORY_REQUIRED",
            ),
            (
                "발바닥 형태에서 평발 특징이 관찰됩니다.",
                "PLANTAR_EVIDENCE_STATEMENT_REQUIRED",
            ),
            (
                "발바닥 형태를 판단하기 어렵습니다.",
                "PLANTAR_ABSTENTION_FORMAT_INVALID",
            ),
        )

        for value, expected_rule in invalid_values:
            with self.subTest(rule=expected_rule):
                with self.assertRaises(
                    service.FullPhotoOutputValidationError
                ) as caught:
                    service._require_full_photo_plantar_footprint_analysis_text(
                        value
                    )

                self.assertEqual(
                    caught.exception.field,
                    "plantar_footprint_analysis_text",
                )
                self.assertEqual(caught.exception.rule_code, expected_rule)


class FullPhotoAnalysisSchemaTests(unittest.TestCase):
    def test_schema_describes_every_generated_text_contract(self) -> None:
        schema = service._full_photo_report_schema()
        properties = schema["properties"]

        self.assertIn("description", schema)
        for field in (
            "hallux_score_analysis_text",
            "fungal_suspicion_safety_description",
            "skin_reaction_safety_description",
            "total_score_description",
        ):
            with self.subTest(field=field):
                self.assertEqual(properties[field]["type"], "string")
                self.assertNotIn("maxLength", properties[field])
                self.assertIn("삼백오십 자 이내", properties[field]["description"])

        plantar = properties["plantar_footprint_analysis_text"]
        self.assertEqual(plantar["type"], "string")
        self.assertNotIn("maxLength", plantar)
        self.assertIn("요족·정상·평발 세 범주 안에서만", plantar["description"])
        self.assertIn("한국어 한 문장", plantar["description"])
        self.assertIn("정상 형태", plantar["description"])
        self.assertIn("경향이 보입니다", plantar["description"])
        self.assertIn("판단하기 어렵습니다", plantar["description"])
        self.assertIn("'수 있습니다'와 '가능성'", plantar["description"])
        self.assertIn("백팔십 자 이내", plantar["description"])
        self.assertIn("압력", plantar["description"])
        self.assertIn("하중", plantar["description"])
        self.assertIn("과내전", plantar["description"])
        self.assertIn("과외전", plantar["description"])
        self.assertIn("비진단 안내 문장을 사용하지 않는다", plantar["description"])
        self.assertNotIn("정확히 두 문장", plantar["description"])

        metric_advice = properties["metric_advice"]
        for metric_type in service._FULL_REPORT_ADVICE_KEYS:
            advice = metric_advice["properties"][metric_type]
            self.assertFalse(advice["additionalProperties"])
            for field in ("observation", "practical_tip"):
                with self.subTest(metric_type=metric_type, field=field):
                    text_schema = advice["properties"][field]
                    self.assertEqual(text_schema["type"], "string")
                    self.assertNotIn("maxLength", text_schema)
                    self.assertIn("한 문장", text_schema["description"])
                    self.assertIn("백팔십 자 이내", text_schema["description"])


class FullPhotoAnalysisGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # These repair-contract fixtures use two attempts; the default five-request
        # budget and exhaustion are covered by the recovery/budget test suites.
        for patcher in (
            patch.object(service.settings, "openai_report_fallback_enabled", False),
            patch.object(service.settings, "openai_report_max_attempts", 2),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_one_request_contains_all_raw_and_valid_analysis_images(self) -> None:
        facts = _facts()
        original_facts = copy.deepcopy(facts)
        create_response = AsyncMock(return_value=_valid_generated())
        related = {
            "HALLUX_RESULT": ANALYSIS_PNG_BYTES,
            "DORSAL_TINEA_MAP": ANALYSIS_PNG_BYTES + b"-dorsal-map",
            "EMPTY_RESULT": b"",
            "INVALID_RESULT": b"not-an-image",
            "PLANTAR_TINEA_OVERLAY": ANALYSIS_PNG_BYTES + b"-plantar-overlay",
        }

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_full_photo_analysis_texts(
                dorsal_raw_image_bytes=JPEG_BYTES,
                left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                facts=facts,
                related_analysis_images=related,
            )

        create_response.assert_awaited_once()
        request = create_response.await_args.kwargs
        self.assertEqual(request["image_detail"], "high")
        self.assertEqual(request["schema_name"], "full_photo_analysis_texts")
        self.assertEqual(
            [label for label, _ in request["images"]],
            [
                "DORSAL RAW ORIGINAL PHOTO",
                service.PLANTAR_COMPOSITE_LABEL,
                "HALLUX_RESULT",
                "DORSAL_TINEA_MAP",
                "PLANTAR_TINEA_OVERLAY",
            ],
        )
        self.assertEqual(
            [image for _, image in request["images"][:2]],
            [JPEG_BYTES, service.combine_raw_plantar_photos(LEFT_PNG_BYTES, RIGHT_WEBP_BYTES)],
        )
        self.assertEqual(
            request["user_payload"]["authoritative_facts_do_not_modify"],
            original_facts,
        )
        self.assertEqual(facts, original_facts)
        schema = request["json_schema"]
        self.assertTrue(schema["additionalProperties"] is False)
        prompt = request["system_prompt"]
        self.assertIn("발바닥 형태를 요족·정상·평발 세 범주 안에서만", prompt)
        self.assertIn("양발은 평발 경향이 보입니다.", prompt)
        self.assertIn("명확한 좌우 형태 차이가 있을 때만", prompt)
        policy = request["user_payload"]["plantar_shape_assessment_policy"]
        self.assertIn("같은 범주로 설명한다", policy["bilateral_consistency"])
        self.assertIn("한쪽 근거가 부족하면 다른 쪽 결과를 복사하지 않는다", policy["separate_categories"])
        self.assertIn("뚜렷한 형태 근거가 있을 때만", policy["high_arch_evidence"])
        self.assertIn("뚜렷한 형태 근거가 있을 때만", policy["low_arch_evidence"])
        self.assertIn("판단 보류 문장을 쓴다", policy["normal_or_insufficient_evidence"])
        for rule in policy.values():
            self.assertIn(rule, prompt)
            self.assertIn(
                rule,
                schema["properties"]["plantar_footprint_analysis_text"]["description"],
            )
        self.assertIn("'수 있습니다'와 '가능성'을 사용하지 않는다", prompt)
        self.assertIn("판단하기 어렵습니다", prompt)
        self.assertIn("압력, 하중, 체중 분포", prompt)
        self.assertIn("과내전, 과외전은 평가하거나 언급하지 않는다", prompt)
        self.assertIn("비진단 안내 문장도 작성하지 않는다", prompt)
        self.assertIn("백팔십 자 이내의 한국어 한 문장", prompt)
        self.assertIn("백 점에 가까울수록 의심 신호가 적고 안전", prompt)
        self.assertIn("영 점에 가까울수록 의심 신호가 강하다", prompt)
        self.assertEqual(
            request["user_payload"]["safety_score_semantics"],
            {
                "direction": "higher_is_safer_lower_is_stronger_signal",
                "aggregation": "arithmetic_mean_of_two_dorsal_feet_half_up",
                "instruction": (
                    "낮은 안전 점수를 건강하거나 의심 신호가 적다는 뜻으로 해석하지 않는다."
                ),
            },
        )
        plantar_checklist = request["user_payload"]["output_contract_checklist"][
            "plantar_footprint_analysis_text"
        ]
        self.assertIn("정상 형태 또는 요족·평발 경향", plantar_checklist)
        self.assertIn("판단 보류 문장", plantar_checklist)
        self.assertIn("'수 있습니다'와 '가능성'", plantar_checklist)
        self.assertIn("압력 평가", plantar_checklist)
        self.assertIn("비진단 안내 문장은 쓰지 않는다", plantar_checklist)
        self.assertNotIn("서버가 마지막 문장으로 정규화", prompt)
        self.assertIn("출력 계약은 권고가 아니라", prompt)
        self.assertIn("종결 부호는 맨 끝에 한 번만", prompt)
        self.assertNotIn("validation_retry", request["user_payload"])
        self.assertEqual(
            set(schema["properties"]),
            {
                "hallux_score_analysis_text",
                "fungal_suspicion_safety_description",
                "skin_reaction_safety_description",
                "total_score_description",
                "plantar_footprint_analysis_text",
                "metric_advice",
            },
        )
        self.assertEqual(
            result.metric_advice["HALLUX_VALGUS"],
            (
                "발가락 정렬에서 좌우 차이가 관찰됩니다.",
                "발 앞부분을 압박하지 않는 신발을 선택해 주세요.",
            ),
        )

        self.assertEqual(result.generation_source, "OPENAI")

    async def test_shape_policy_is_shared_without_rewriting_valid_results(self) -> None:
        for shape_text in (
            "양발은 평발 경향이 보입니다.",
            "양발은 정상 형태입니다.",
            "왼발은 요족 경향이 있으며 오른발은 평발 경향이 있습니다.",
        ):
            with self.subTest(shape_text=shape_text):
                generated = _valid_generated()
                generated["plantar_footprint_analysis_text"] = shape_text
                create_response = AsyncMock(side_effect=[
                    {"plantar_footprint_analysis_text": shape_text},
                    generated,
                ])
                facts = _facts()
                with (
                    patch.object(service, "_openai_enabled", return_value=True),
                    patch.object(service.settings, "openai_report_include_images", True),
                    patch.object(service, "_create_structured_response", create_response),
                ):
                    standalone = await service.generate_plantar_footprint_analysis_text(
                        left_raw_image_bytes=LEFT_PNG_BYTES,
                        right_raw_image_bytes=RIGHT_WEBP_BYTES,
                        left_segmentation_facts=facts["plantar_footprint"]["left"],
                        right_segmentation_facts=facts["plantar_footprint"]["right"],
                    )
                    full = await service.generate_full_photo_analysis_texts(
                        dorsal_raw_image_bytes=JPEG_BYTES,
                        left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                        right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                        facts=facts,
                        related_analysis_images={},
                    )

                self.assertEqual(standalone, shape_text)
                self.assertEqual(full.plantar_footprint_analysis_text, shape_text)
                self.assertEqual(create_response.await_count, 2)
                standalone_request, full_request = (
                    call.kwargs for call in create_response.await_args_list
                )
                self.assertEqual(
                    standalone_request["user_payload"]["plantar_shape_assessment_policy"],
                    full_request["user_payload"]["plantar_shape_assessment_policy"],
                )
                self.assertEqual(standalone_request["images"], [
                    (service.PLANTAR_COMPOSITE_LABEL,
                     service.combine_raw_plantar_photos(LEFT_PNG_BYTES, RIGHT_WEBP_BYTES)),
                ])
                self.assertEqual(full_request["images"][1], standalone_request["images"][0])
                self.assertEqual(
                    full_request["user_payload"]["plantar_image_contract"],
                    standalone_request["user_payload"]["image_contract"],
                )

    async def test_disabled_openai_is_rejected_without_request(self) -> None:
        create_response = AsyncMock()
        with (
            patch.object(service, "_openai_enabled", return_value=False),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaisesRegex(RuntimeError, "required"):
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=JPEG_BYTES,
                    left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                    right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                    facts=_facts(),
                    related_analysis_images={},
                )

        create_response.assert_not_awaited()

    async def test_api_failure_propagates_after_exactly_one_attempt(self) -> None:
        create_response = AsyncMock(side_effect=RuntimeError("offline"))
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=JPEG_BYTES,
                    left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                    right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                    facts=_facts(),
                    related_analysis_images={},
                )

        create_response.assert_awaited_once()

    async def test_validation_failure_retries_once_with_safe_corrective_contract(
        self,
    ) -> None:
        invalid = _valid_generated()
        invalid["plantar_footprint_analysis_text"] = (
            "발바닥 압력 분포에서 평발 경향이 보이며 원문비밀표식은 재사용하지 않습니다."
        )
        create_response = AsyncMock(side_effect=[invalid, _valid_generated()])

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_full_photo_analysis_texts(
                dorsal_raw_image_bytes=JPEG_BYTES,
                left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                facts=_facts(),
                related_analysis_images={"HALLUX_RESULT": ANALYSIS_PNG_BYTES},
            )

        self.assertEqual(create_response.await_count, 2)
        first_request = create_response.await_args_list[0].kwargs
        retry_request = create_response.await_args_list[1].kwargs
        self.assertNotIn("validation_retry", first_request["user_payload"])
        self.assertEqual(
            retry_request["user_payload"]["validation_retry"],
            {
                "attempt": 2,
                "maximum_attempts": 2,
                "failed_field": "plantar_footprint_analysis_text",
                "failed_rule_code": "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
                "instruction": (
                    "이전 응답 본문을 추측하거나 부분 수정하지 말고, 같은 facts와 이미지를 근거로 "
                    "전체 JSON을 처음부터 다시 생성한 뒤 모든 출력 규칙을 재검사한다."
                ),
            },
        )
        self.assertIn("교정 재시도", retry_request["system_prompt"])
        self.assertIn(
            "plantar_footprint_analysis_text",
            retry_request["system_prompt"],
        )
        self.assertIn(
            "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
            retry_request["system_prompt"],
        )
        retry_contract = json.dumps(
            {
                "system_prompt": retry_request["system_prompt"],
                "validation_retry": retry_request["user_payload"]["validation_retry"],
            },
            ensure_ascii=False,
        )
        self.assertNotIn("원문비밀표식", retry_contract)
        self.assertEqual(first_request["images"], retry_request["images"])
        self.assertEqual(first_request["json_schema"], retry_request["json_schema"])
        self.assertEqual(
            first_request["user_payload"]["plantar_shape_assessment_policy"],
            retry_request["user_payload"]["plantar_shape_assessment_policy"],
        )
        self.assertEqual(result.generation_source, "OPENAI")

    async def test_natural_metric_wording_does_not_require_keyword_whitelists(self) -> None:
        generated = _valid_generated()
        generated["metric_advice"]["ATHLETES_FOOT"] = {
            "observation": "무좀 의심 징후가 거의 나타나지 않습니다.",
            "practical_tip": "통풍이 잘되는 신발을 착용해 주세요.",
        }
        generated["metric_advice"]["SKIN_IRRITATION"] = {
            "observation": "피부 변화 영역이 거의 나타나지 않습니다.",
            "practical_tip": "마찰이 적은 신발을 착용해 주세요.",
        }
        create_response = AsyncMock(return_value=generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_full_photo_analysis_texts(
                dorsal_raw_image_bytes=JPEG_BYTES,
                left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                facts=_facts(),
                related_analysis_images={},
            )

        self.assertEqual(
            result.metric_advice["ATHLETES_FOOT"],
            (
                "무좀 의심 징후가 거의 나타나지 않습니다.",
                "통풍이 잘되는 신발을 착용해 주세요.",
            ),
        )
        self.assertEqual(result.generation_source, "OPENAI")

    async def test_one_plantar_shape_result_sentence_is_accepted_without_disclaimer(
        self,
    ) -> None:
        generated = _valid_generated()
        generated["plantar_footprint_analysis_text"] = (
            "왼발은 정상 형태이며 오른발은 평발 경향이 보입니다."
        )
        create_response = AsyncMock(return_value=generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_full_photo_analysis_texts(
                dorsal_raw_image_bytes=JPEG_BYTES,
                left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                facts=_facts(),
                related_analysis_images={},
            )

        self.assertEqual(
            result.plantar_footprint_analysis_text,
            "왼발은 정상 형태이며 오른발은 평발 경향이 보입니다.",
        )
        self.assertEqual(len(result.plantar_footprint_analysis_text.splitlines()), 1)
        self.assertNotIn("진단", result.plantar_footprint_analysis_text)
        self.assertEqual(result.generation_source, "OPENAI")

    async def test_invalid_raw_mime_is_rejected_without_request(self) -> None:
        create_response = AsyncMock()
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaises(ValueError):
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=b"invalid",
                    left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                    right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                    facts=_facts(),
                    related_analysis_images={},
                )

        create_response.assert_not_awaited()

    async def test_numeric_or_diagnostic_generated_text_is_rejected(self) -> None:
        invalid_variants = []
        numeric = _valid_generated()
        numeric["total_score_description"] = "새 안전 점수는 91점으로 보입니다."
        invalid_variants.append(
            (numeric, "total_score_description", "DIGITS_NOT_ALLOWED")
        )
        diagnostic = _valid_generated()
        diagnostic["hallux_score_analysis_text"] = "무지외반증으로 확진되었습니다."
        invalid_variants.append(
            (
                diagnostic,
                "hallux_score_analysis_text",
                "DIAGNOSTIC_OR_CERTAINTY_NOT_ALLOWED",
            )
        )
        markup = _valid_generated()
        markup["metric_advice"]["ATHLETES_FOOT"]["practical_tip"] = (
            "**통풍이 잘되는 신발을 착용해 주세요.**"
        )
        invalid_variants.append(
            (
                markup,
                "metric_advice.ATHLETES_FOOT.practical_tip",
                "MARKUP_NOT_ALLOWED",
            )
        )

        for generated, expected_field, expected_rule in invalid_variants:
            with self.subTest(field=expected_field, rule=expected_rule):
                create_response = AsyncMock(return_value=generated)
                with (
                    patch.object(service, "_openai_enabled", return_value=True),
                    patch.object(service.settings, "openai_report_include_images", True),
                    patch.object(service, "_create_structured_response", create_response),
                ):
                    with self.assertRaises(
                        service.FullPhotoOutputValidationError
                    ) as caught:
                        await service.generate_full_photo_analysis_texts(
                            dorsal_raw_image_bytes=JPEG_BYTES,
                            left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                            right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                            facts=_facts(),
                            related_analysis_images={},
                        )

                self.assertEqual(caught.exception.field, expected_field)
                self.assertEqual(caught.exception.rule_code, expected_rule)
                self.assertNotIn("91점", str(caught.exception))
                self.assertEqual(create_response.await_count, 2)

    async def test_metric_advice_requires_exact_keys_and_one_sentence_per_part(self) -> None:
        invalid = _valid_generated()
        invalid["metric_advice"]["HALLUX_VALGUS"]["observation"] = (
            "발가락 정렬 차이가 관찰됩니다. 변화 경향도 보입니다."
        )
        create_response = AsyncMock(return_value=invalid)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaises(
                service.FullPhotoOutputValidationError
            ) as caught:
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=JPEG_BYTES,
                    left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                    right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                    facts=_facts(),
                    related_analysis_images={},
                )

        self.assertEqual(
            caught.exception.field,
            "metric_advice.HALLUX_VALGUS.observation",
        )
        self.assertEqual(
            caught.exception.rule_code,
            "SENTENCE_COUNT_OUT_OF_RANGE",
        )
        self.assertEqual(create_response.await_count, 2)

    async def test_corrective_retry_stops_after_second_validation_failure(self) -> None:
        first_invalid = _valid_generated()
        first_invalid["metric_advice"]["HALLUX_VALGUS"]["observation"] = (
            "발가락 정렬 차이가 관찰됩니다. 변화 경향도 보입니다."
        )
        second_invalid = _valid_generated()
        second_invalid["metric_advice"]["ATHLETES_FOOT"]["practical_tip"] = (
            "발을 건조하게 관리해 주세요. 통풍도 유지해 주세요."
        )
        create_response = AsyncMock(side_effect=[first_invalid, second_invalid])

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaises(
                service.FullPhotoOutputValidationError
            ) as caught:
                await service.generate_full_photo_analysis_texts(
                    dorsal_raw_image_bytes=JPEG_BYTES,
                    left_plantar_raw_image_bytes=LEFT_PNG_BYTES,
                    right_plantar_raw_image_bytes=RIGHT_WEBP_BYTES,
                    facts=_facts(),
                    related_analysis_images={},
                )

        self.assertEqual(create_response.await_count, 2)
        self.assertEqual(
            caught.exception.field,
            "metric_advice.ATHLETES_FOOT.practical_tip",
        )
        self.assertEqual(
            caught.exception.rule_code,
            "SENTENCE_COUNT_OUT_OF_RANGE",
        )


if __name__ == "__main__":
    unittest.main()
