from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service


LEFT_PNG = b"\x89PNG\r\n\x1a\nleft-pressure-heatmap"
RIGHT_PNG = b"\x89PNG\r\n\x1a\nright-pressure-heatmap"
VALID_ADVICE = {
    "leftRegionalObservation": "왼발은 앞꿈치 구역의 압력 비중이 상대적으로 크고 특정 구역에 압력이 집중되는 경향이 관찰됩니다.",
    "rightRegionalObservation": "오른발은 앞꿈치 구역의 압력 비중이 상대적으로 크고 특정 구역에 압력이 집중되는 경향이 관찰됩니다.",
    "overallBalanceObservation": "왼발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다.",
    "practicalTip": "서 있을 때 양발에 천천히 고르게 힘을 실어 주세요.",
}


def _proportional_values() -> tuple[list[float], list[float]]:
    return [7.0] + [0.0] * 11, _right_hardware([3.0] + [0.0] * 11)


def _right_hardware(canonical_values: list[float]) -> list[float]:
    return list(reversed(canonical_values))


class PressureBalanceFactsTests(unittest.TestCase):
    def test_heel_first_right_hardware_sample_is_canonicalized_for_zone_facts(self) -> None:
        left = [
            64.0, 44.7, 39.3, 27.3, 16.6, 574.3,
            924.3, 66.5, 77.5, 11.6, 11376.8, 10670.5,
        ]
        right = [
            8088.3, 3086.2, 3184.6, 3754.0, 17.5, 1724.6,
            256.8, 0.0, 0.0, 8.3, 9.3, 0.0,
        ]
        original_right = right.copy()

        facts = service.build_pressure_balance_facts(left, right)

        self.assertEqual(right, original_right)
        self.assertEqual(facts["rightRawZoneLoadPercent"]["forefoot"], 0.1)
        self.assertEqual(facts["rightRawZoneLoadPercent"]["heel"], 71.3)
        self.assertEqual(facts["rightDominantZones"], ["HEEL"])
        self.assertEqual(facts["leftLoadPercent"], 54.3)
        self.assertEqual(facts["rightLoadPercent"], 45.7)

    def test_balanced_identical_frames_have_exact_facts_and_zone_groups(self) -> None:
        facts = service.build_pressure_balance_facts([1.0] * 12, [1.0] * 12)

        self.assertEqual(
            facts,
            {
                "leftTotal": 12.0,
                "rightTotal": 12.0,
                "leftLoadPercent": 50.0,
                "rightLoadPercent": 50.0,
                "loadDifferencePercentagePoints": 0.0,
                "loadImbalancePercent": 0.0,
                "profileAsymmetryPercent": 0.0,
                "leftActiveSensorCount": 12,
                "rightActiveSensorCount": 12,
                "bilateralBalanceScore": 100.0,
                "leftDistributionScore": 100.0,
                "rightDistributionScore": 100.0,
                "distributionScore": 100.0,
                "balanceScore": 100.0,
                "scoreComponents": {
                    "bilateralBalanceScore": 100.0,
                    "leftDistributionScore": 100.0,
                    "rightDistributionScore": 100.0,
                },
                "scoreWeights": {
                    "bilateralBalance": 0.6,
                    "leftDistribution": 0.2,
                    "rightDistribution": 0.2,
                },
                "scoreFormula": (
                    "0.6 * bilateralBalanceScore + 0.2 * leftDistributionScore "
                    "+ 0.2 * rightDistributionScore"
                ),
                "distributionScoreFormula": (
                    "100 * max(0, 1 - sum(abs(sensorCountAdjustedZoneShare - 1/3)) / (4/3))"
                ),
                "formulaVersion": "pressure-balance-v2",
                "zoneAdjustmentFormula": (
                    "zoneMean = zoneSum / zoneSensorCount; adjustedZoneShare = "
                    "zoneMean / sum(allZoneMeans)"
                ),
                "distributionScoreMetadata": {
                    "provisional": True,
                    "nonClinical": True,
                    "meaning": "sensor-count-adjusted three-zone spread index",
                },
                "loadBalanceLevel": "BALANCED",
                "qualityStatus": "valid",
                "leftRawZoneLoadPercent": {
                    "forefoot": 41.7,
                    "midfoot": 33.3,
                    "heel": 25.0,
                },
                "rightRawZoneLoadPercent": {
                    "forefoot": 41.7,
                    "midfoot": 33.3,
                    "heel": 25.0,
                },
                "leftZonePercent": {
                    "forefoot": 33.3,
                    "midfoot": 33.3,
                    "heel": 33.4,
                },
                "rightZonePercent": {
                    "forefoot": 33.3,
                    "midfoot": 33.3,
                    "heel": 33.4,
                },
                "leftSensorCountAdjustedZonePercent": {
                    "forefoot": 33.3,
                    "midfoot": 33.3,
                    "heel": 33.4,
                },
                "rightSensorCountAdjustedZonePercent": {
                    "forefoot": 33.3,
                    "midfoot": 33.3,
                    "heel": 33.4,
                },
                "leftDominantZones": ["FOREFOOT", "MIDFOOT", "HEEL"],
                "rightDominantZones": ["FOREFOOT", "MIDFOOT", "HEEL"],
                "leftDistributionLevel": "BROADLY_DISTRIBUTED",
                "rightDistributionLevel": "BROADLY_DISTRIBUTED",
                "dominantFoot": "BALANCED",
            },
        )
        left_fallback, right_fallback, _, _ = service._pressure_fallback_components(
            facts
        )
        self.assertEqual(
            left_fallback,
            "왼발은 앞꿈치·중족부·뒤꿈치 세 구역에 압력이 비교적 고르게 분산되는 경향이 관찰됩니다.",
        )
        self.assertEqual(
            right_fallback,
            "오른발은 앞꿈치·중족부·뒤꿈치 세 구역에 압력이 비교적 고르게 분산되는 경향이 관찰됩니다.",
        )
        self.assertNotIn("상대적으로 크", left_fallback)

    def test_score_is_explainable_weighted_bilateral_and_per_foot_composite(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)

        self.assertEqual(facts["leftLoadPercent"], 70.0)
        self.assertEqual(facts["rightLoadPercent"], 30.0)
        self.assertEqual(facts["loadDifferencePercentagePoints"], 40.0)
        self.assertEqual(facts["bilateralBalanceScore"], 60.0)
        self.assertEqual(facts["leftDistributionScore"], 0.0)
        self.assertEqual(facts["rightDistributionScore"], 0.0)
        self.assertEqual(facts["distributionScore"], 0.0)
        self.assertEqual(facts["balanceScore"], 36.0)
        self.assertEqual(
            facts["balanceScore"],
            round(
                0.6 * facts["bilateralBalanceScore"]
                + 0.2 * facts["leftDistributionScore"]
                + 0.2 * facts["rightDistributionScore"],
                1,
            ),
        )
        self.assertEqual(facts["profileAsymmetryPercent"], 0.0)
        self.assertEqual(facts["dominantFoot"], "LEFT")
        self.assertEqual(facts["loadBalanceLevel"], "STRONG_ONE_SIDED")

    def test_user_sample_has_complementary_percentages_and_v2_composite(self) -> None:
        left = [
            50.5,
            0.0,
            793.7,
            157.6,
            26.4,
            7908.9,
            3204.6,
            56.7,
            1859.7,
            14810.9,
            9935.6,
            10556.9,
        ]
        right = _right_hardware([
            48.5,
            0.0,
            812.3,
            160.1,
            30.0,
            7800.2,
            3100.5,
            60.0,
            1900.0,
            14000.1,
            9700.4,
            10100.8,
        ])

        facts = service.build_pressure_balance_facts(left, right)

        self.assertEqual(facts["leftLoadPercent"] + facts["rightLoadPercent"], 100.0)
        self.assertEqual(facts["loadDifferencePercentagePoints"], 1.6)
        self.assertEqual(facts["loadImbalancePercent"], 1.6)
        self.assertEqual(facts["bilateralBalanceScore"], 98.4)
        self.assertEqual(facts["leftDistributionScore"], 34.1)
        self.assertEqual(facts["rightDistributionScore"], 35.0)
        self.assertEqual(facts["balanceScore"], 72.9)
        self.assertEqual(facts["loadBalanceLevel"], "BALANCED")
        self.assertEqual(facts["leftDominantZones"], ["HEEL"])
        self.assertEqual(facts["rightDominantZones"], ["HEEL"])

    def test_loaded_percentages_are_exact_complements_after_rounding(self) -> None:
        facts = service.build_pressure_balance_facts(
            [1.0] + [0.0] * 11,
            [2.0] + [0.0] * 11,
        )

        self.assertEqual(facts["leftLoadPercent"], 33.3)
        self.assertEqual(facts["rightLoadPercent"], 66.7)
        self.assertEqual(
            facts["leftLoadPercent"] + facts["rightLoadPercent"],
            100.0,
        )
        self.assertEqual(facts["loadDifferencePercentagePoints"], 33.4)
        self.assertEqual(facts["bilateralBalanceScore"], 66.6)

    def test_profile_asymmetry_is_observation_only_while_zone_spread_affects_score(self) -> None:
        left = [1.0] * 6 + [0.0] * 6
        right = _right_hardware([0.0] * 6 + [1.0] * 6)

        facts = service.build_pressure_balance_facts(left, right)

        self.assertEqual(facts["leftLoadPercent"], 50.0)
        self.assertEqual(facts["rightLoadPercent"], 50.0)
        self.assertEqual(facts["profileAsymmetryPercent"], 100.0)
        self.assertEqual(facts["bilateralBalanceScore"], 100.0)
        self.assertEqual(facts["leftDistributionScore"], 30.0)
        self.assertEqual(facts["rightDistributionScore"], 50.0)
        self.assertEqual(facts["balanceScore"], 76.0)

    def test_extreme_one_sided_load_has_fixed_low_score_and_explicit_level(self) -> None:
        facts = service.build_pressure_balance_facts(
            [99.0] + [0.0] * 11,
            [1.0] + [0.0] * 11,
        )

        self.assertEqual(facts["leftLoadPercent"], 99.0)
        self.assertEqual(facts["rightLoadPercent"], 1.0)
        self.assertEqual(facts["loadDifferencePercentagePoints"], 98.0)
        self.assertEqual(facts["bilateralBalanceScore"], 2.0)
        self.assertEqual(facts["balanceScore"], 1.2)
        self.assertEqual(facts["loadBalanceLevel"], "EXTREME_ONE_SIDED")
        self.assertEqual(facts["dominantFoot"], "LEFT")

    def test_three_percentage_point_difference_is_still_balanced(self) -> None:
        boundary = service.build_pressure_balance_facts(
            [51.5] + [0.0] * 11,
            [48.5] + [0.0] * 11,
        )
        outside = service.build_pressure_balance_facts(
            [51.6] + [0.0] * 11,
            [48.4] + [0.0] * 11,
        )

        self.assertEqual(boundary["loadDifferencePercentagePoints"], 3.0)
        self.assertEqual(boundary["dominantFoot"], "BALANCED")
        self.assertEqual(outside["loadDifferencePercentagePoints"], 3.2)
        self.assertEqual(outside["dominantFoot"], "LEFT")

    def test_zone_gap_within_five_points_is_reported_as_mixed_dominance(self) -> None:
        # Forefoot and midfoot differ by less than five percentage points;
        # heel remains clearly below both.
        values = [10.0] + [0.0] * 4 + [7.8] + [0.0] * 3 + [1.0] + [0.0] * 2

        facts = service.build_pressure_balance_facts(
            values,
            _right_hardware(values),
        )

        self.assertEqual(facts["leftDominantZones"], ["FOREFOOT", "MIDFOOT"])
        self.assertEqual(facts["rightDominantZones"], ["FOREFOOT", "MIDFOOT"])

    def test_no_load_and_single_side_load_are_explicitly_inconclusive(self) -> None:
        no_load = service.build_pressure_balance_facts([0.0] * 12, [0.0] * 12)
        one_side = service.build_pressure_balance_facts([0.0] * 12, [1.0] * 12)

        self.assertEqual(no_load["qualityStatus"], "no-load")
        self.assertEqual(no_load["loadBalanceLevel"], "INCONCLUSIVE")
        self.assertIsNone(no_load["balanceScore"])
        self.assertEqual(no_load["leftLoadPercent"], 0.0)
        self.assertEqual(no_load["rightLoadPercent"], 0.0)
        self.assertEqual(no_load["profileAsymmetryPercent"], 0.0)
        self.assertEqual(no_load["leftDominantZones"], [])
        self.assertEqual(no_load["rightDominantZones"], [])
        self.assertEqual(no_load["leftZonePercent"], {
            "forefoot": 0.0,
            "midfoot": 0.0,
            "heel": 0.0,
        })

        self.assertEqual(one_side["qualityStatus"], "single-side-no-load")
        self.assertEqual(one_side["loadBalanceLevel"], "INCONCLUSIVE")
        self.assertIsNone(one_side["balanceScore"])
        self.assertEqual(one_side["leftLoadPercent"], 0.0)
        self.assertEqual(one_side["rightLoadPercent"], 100.0)
        self.assertEqual(one_side["loadImbalancePercent"], 100.0)
        self.assertEqual(one_side["profileAsymmetryPercent"], 100.0)
        self.assertEqual(one_side["dominantFoot"], "RIGHT")
        self.assertEqual(one_side["leftDominantZones"], [])
        self.assertEqual(
            one_side["rightDominantZones"],
            ["FOREFOOT", "MIDFOOT", "HEEL"],
        )
        self.assertIsNone(one_side["bilateralBalanceScore"])
        self.assertIsNone(one_side["leftDistributionScore"])
        self.assertEqual(one_side["rightDistributionScore"], 100.0)
        self.assertEqual(one_side["leftDistributionLevel"], "INCONCLUSIVE")
        self.assertEqual(
            one_side["rightDistributionLevel"],
            "BROADLY_DISTRIBUTED",
        )

    def test_invalid_sensor_frames_are_rejected(self) -> None:
        invalid_left_frames = (
            [0.0] * 11,
            [0.0] * 13,
            [-0.1] + [0.0] * 11,
            [True] + [0.0] * 11,
            ["1"] + [0.0] * 11,
            [math.nan] + [0.0] * 11,
            [math.inf] + [0.0] * 11,
            [1e308] * 12,
        )

        for invalid in invalid_left_frames:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    service.build_pressure_balance_facts(invalid, [1.0] * 12)


class PressureBalanceGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for patcher in (
            patch.object(service.settings, "openai_report_max_attempts", 5),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service.settings, "openai_report_fallback_enabled", True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_one_high_detail_request_contains_facts_and_heatmap_reading_contract(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)
        original_facts = copy.deepcopy(facts)
        create_response = AsyncMock(return_value=VALID_ADVICE)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )

        create_response.assert_awaited_once()
        request = create_response.await_args.kwargs
        self.assertEqual(request["schema_name"], "pressure_balance_report")
        self.assertEqual(request["image_detail"], "high")
        self.assertEqual(
            request["images"],
            [
                ("LEFT DERIVED PRESSURE HEATMAP", LEFT_PNG),
                ("RIGHT DERIVED PRESSURE HEATMAP", RIGHT_PNG),
            ],
        )
        numeric = request["user_payload"]["numeric_source_of_truth"]
        self.assertEqual(set(numeric), {"serverComputedPressureFacts"})
        self.assertNotIn("leftRawPressureBySensor", str(request["user_payload"]))
        self.assertNotIn("rightRawPressureBySensor", str(request["user_payload"]))
        self.assertEqual(numeric["serverComputedPressureFacts"], original_facts)
        self.assertEqual(facts, original_facts)
        reading = request["user_payload"]["heatmap_reading_contract"]
        self.assertEqual(reading["source_role"], "spatial_visual_evidence")
        self.assertEqual(reading["orientation"], {
            "top": "FOREFOOT", "middle": "MIDFOOT", "bottom": "HEEL",
            "right_display_transform": "horizontal_flip_only",
        })
        self.assertEqual(
            reading["color_scale"]["normalization"],
            "independent_foot_peak_and_mask_minmax",
        )
        self.assertFalse(reading["color_scale"]["linear_pressure_scale"])
        self.assertIn("Multiquadric RBF", request["user_payload"]["image_contract"])
        self.assertIn("투명 배경", request["user_payload"]["image_contract"])
        self.assertIn("같은 압력 크기로 비교하지 않는다", reading["compare_feet"])
        self.assertEqual(reading["color_scale"]["low_to_high"], [
            "blue", "cyan", "green", "yellow", "orange", "red", "dark_red",
        ])
        schema = request["json_schema"]
        self.assertEqual(
            set(schema["properties"]),
            {
                "leftRegionalObservation",
                "rightRegionalObservation",
                "overallBalanceObservation",
                "practicalTip",
            },
        )
        self.assertEqual(schema["required"], list(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("score", schema["properties"])
        self.assertEqual(result.score, 36.0)
        self.assertEqual(result.generation_source, "OPENAI")
        self.assertEqual(
            result.left_regional_observation,
            VALID_ADVICE["leftRegionalObservation"],
        )
        self.assertEqual(
            result.right_regional_observation,
            VALID_ADVICE["rightRegionalObservation"],
        )
        self.assertEqual(
            result.overall_balance_observation,
            VALID_ADVICE["overallBalanceObservation"],
        )
        self.assertEqual(result.practical_tip, VALID_ADVICE["practicalTip"])
        self.assertEqual(
            result.advice,
            (
                "\n".join(
                    (
                        VALID_ADVICE["leftRegionalObservation"],
                        VALID_ADVICE["rightRegionalObservation"],
                        VALID_ADVICE["overallBalanceObservation"],
                    )
                ),
                VALID_ADVICE["practicalTip"],
            ),
        )
        self.assertEqual(
            result.balance_comment,
            "\n".join(
                (
                    VALID_ADVICE["leftRegionalObservation"],
                    VALID_ADVICE["rightRegionalObservation"],
                    VALID_ADVICE["overallBalanceObservation"],
                    VALID_ADVICE["practicalTip"],
                )
            ),
        )

    async def test_distinct_heatmap_hotspot_descriptions_survive_without_changing_score(self) -> None:
        left = [7.0] + [0.0] * 11
        right = _right_hardware([1.0, 0.0, 1.0, 0.0, 1.0] + [0.0] * 7)
        facts = service.build_pressure_balance_facts(left, right)
        generated = {
            **VALID_ADVICE,
            "leftRegionalObservation": (
                "왼발은 앞꿈치에 압력이 집중되고 히트맵의 높은 강도 영역이 좁게 모여 관찰됩니다."
            ),
            "rightRegionalObservation": (
                "오른발은 앞꿈치에 압력이 집중되며 히트맵의 높은 강도 영역이 여러 지점으로 떨어져 관찰됩니다."
            ),
        }
        create_response = AsyncMock(return_value=generated)
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service.settings, "openai_report_include_images", True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left, right_pressure_values=right, pressure_facts=facts,
                left_heatmap_png=LEFT_PNG, right_heatmap_png=RIGHT_PNG,
            )
        self.assertEqual(facts["leftDistributionLevel"], "CONCENTRATED")
        self.assertEqual(facts["rightDistributionLevel"], "CONCENTRATED")
        self.assertEqual(facts["leftDominantZones"], ["FOREFOOT"])
        self.assertEqual(facts["rightDominantZones"], ["FOREFOOT"])
        create_response.assert_awaited_once()
        self.assertEqual(result.generation_source, "OPENAI")
        self.assertIsNone(result.generation_failure_code)
        self.assertEqual(result.score, facts["balanceScore"])
        self.assertEqual(result.left_regional_observation, generated["leftRegionalObservation"])
        self.assertEqual(result.right_regional_observation, generated["rightRegionalObservation"])
        self.assertIn(generated["leftRegionalObservation"], result.balance_comment)
        self.assertIn(generated["rightRegionalObservation"], result.advice[0])

    async def test_visual_detail_does_not_override_measured_dominant_foot_or_zone(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)
        contradictions = (
            ("overallBalanceObservation", "오른발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다."),
            (
                "leftRegionalObservation",
                "왼발은 뒤꿈치에 압력이 집중되고 히트맵의 높은 강도 영역이 좁게 모여 관찰됩니다.",
            ),
        )
        for field, invented in contradictions:
            with self.subTest(field=field):
                generated = {**VALID_ADVICE, field: invented}
                create_response = AsyncMock(return_value=generated)
                with (
                    patch.object(service, "_openai_enabled", return_value=True),
                    patch.object(service.settings, "openai_report_include_images", True),
                    patch.object(service.settings, "openai_report_fallback_enabled", True),
                    patch.object(service, "_create_structured_response", create_response),
                ):
                    result = await service.generate_pressure_balance_report(
                        left_pressure_values=left, right_pressure_values=right, pressure_facts=facts,
                        left_heatmap_png=LEFT_PNG, right_heatmap_png=RIGHT_PNG,
                    )
                self.assertEqual(create_response.await_count, 5)
                self.assertEqual(result.generation_source, "DETERMINISTIC_FALLBACK")
                self.assertEqual(result.score, facts["balanceScore"])
                self.assertIn("앞꿈치", result.left_regional_observation)
                self.assertIn("왼발 쪽으로 압력이 크게 치우", result.overall_balance_observation)
                self.assertNotIn(invented, result.balance_comment)

    async def test_disabled_openai_returns_deterministic_fallback_without_request(self) -> None:
        left = [99.0] + [0.0] * 11
        right = [1.0] + [0.0] * 11
        facts = service.build_pressure_balance_facts(left, right)
        create_response = AsyncMock()

        with (
            patch.object(service, "_openai_enabled", return_value=False),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=b"invalid-but-unused",
                right_heatmap_png=b"invalid-but-unused",
            )

        create_response.assert_not_awaited()
        self.assertEqual(result.score, facts["balanceScore"])
        self.assertEqual(len(result.advice), 2)
        self.assertIn("왼발 쪽으로 압력이 크게 치우치는 경향", result.advice[0])

    async def test_spoofed_or_inconclusive_facts_raise_before_openai(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)
        spoofed = copy.deepcopy(facts)
        spoofed["balanceScore"] = 100.0
        inconclusive_left = [0.0] * 12
        inconclusive_right = [1.0] * 12
        inconclusive = service.build_pressure_balance_facts(
            inconclusive_left,
            inconclusive_right,
        )
        create_response = AsyncMock()

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            with self.assertRaisesRegex(ValueError, "exactly match"):
                await service.generate_pressure_balance_report(
                    left_pressure_values=left,
                    right_pressure_values=right,
                    pressure_facts=spoofed,
                    left_heatmap_png=LEFT_PNG,
                    right_heatmap_png=RIGHT_PNG,
                )
            with self.assertRaisesRegex(ValueError, "positive load on both sides"):
                await service.generate_pressure_balance_report(
                    left_pressure_values=inconclusive_left,
                    right_pressure_values=inconclusive_right,
                    pressure_facts=inconclusive,
                    left_heatmap_png=LEFT_PNG,
                    right_heatmap_png=RIGHT_PNG,
                )

        create_response.assert_not_awaited()

    async def test_invalid_heatmap_or_api_failure_attempts_at_most_once_and_falls_back(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)

        invalid_image_call = AsyncMock(return_value=VALID_ADVICE)
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", invalid_image_call),
        ):
            invalid_image_result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=b"not-a-png",
                right_heatmap_png=RIGHT_PNG,
            )
        invalid_image_call.assert_not_awaited()
        self.assertEqual(invalid_image_result.score, facts["balanceScore"])

        failed_call = AsyncMock(side_effect=RuntimeError("refusal"))
        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", failed_call),
        ):
            failed_result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )
        failed_call.assert_awaited_once()
        self.assertEqual(failed_result.score, facts["balanceScore"])

    async def test_malformed_diagnostic_or_score_bearing_output_uses_whole_fallback(self) -> None:
        left, right = _proportional_values()
        facts = service.build_pressure_balance_facts(left, right)
        invalid_outputs = (
            {"leftRegionalObservation": VALID_ADVICE["leftRegionalObservation"]},
            {**VALID_ADVICE, "score": 100},
            {
                **VALID_ADVICE,
                "leftRegionalObservation": (
                    "왼발은 앞꿈치 압력 비중이 가장 큰 경향이 관찰됩니다. "
                    "다른 경향도 보입니다."
                ),
            },
            {
                **VALID_ADVICE,
                "rightRegionalObservation": (
                    "오른발은 앞꿈치 압력 비중이 70으로 가장 큰 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "overallBalanceObservation": "압력 불균형으로 진단됩니다.",
            },
            {
                **VALID_ADVICE,
                "leftRegionalObservation": (
                    "왼발은 **앞꿈치 압력 비중**이 가장 큰 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "practicalTip": "고르게 힘을 싣는 방법은 https://example.com 에 있습니다.",
            },
            {
                **VALID_ADVICE,
                "overallBalanceObservation": (
                    "오른발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "overallBalanceObservation": (
                    "왼발 압력이 상대적으로 큰 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "leftRegionalObservation": (
                    "왼발은 뒤꿈치 구역의 압력 비중이 가장 큰 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "leftRegionalObservation": (
                    "왼발은 구역 압력 비중이 가장 큰 경향이 관찰됩니다."
                ),
            },
            {
                **VALID_ADVICE,
                "practicalTip": "과내전 가능성을 줄이도록 양발에 고르게 힘을 실어 주세요.",
            },
            {
                **VALID_ADVICE,
                "leftRegionalObservation": (
                    "왼발은 앞꿈치 구역이 정상 범위이며 압력이 집중되는 경향이 관찰됩니다."
                ),
            },
        )

        for generated in invalid_outputs:
            with self.subTest(generated=generated):
                create_response = AsyncMock(return_value=generated)
                with (
                    patch.object(service, "_openai_enabled", return_value=True),
                    patch.object(service, "_create_structured_response", create_response),
                ):
                    result = await service.generate_pressure_balance_report(
                        left_pressure_values=left,
                        right_pressure_values=right,
                        pressure_facts=facts,
                        left_heatmap_png=LEFT_PNG,
                        right_heatmap_png=RIGHT_PNG,
                    )
                self.assertEqual(create_response.await_count, 5)
                self.assertEqual(result.score, facts["balanceScore"])
                self.assertIn("앞꿈치", result.left_regional_observation)
                self.assertIn("앞꿈치", result.right_regional_observation)
                self.assertIn("왼발 쪽으로 압력이 크게 치우", result.overall_balance_observation)

    async def test_each_foot_regional_observation_uses_its_own_dominant_zone(self) -> None:
        left = [0.0] * 9 + [10.0, 0.0, 0.0]
        right = _right_hardware([0.0] * 5 + [10.0] + [0.0] * 6)
        facts = service.build_pressure_balance_facts(left, right)
        generated = {
            "leftRegionalObservation": (
                "왼발은 뒤꿈치 구역의 압력 비중이 상대적으로 크고 특정 구역에 압력이 집중되는 경향이 관찰됩니다."
            ),
            "rightRegionalObservation": (
                "오른발은 중족부 구역의 압력 비중이 상대적으로 크고 특정 구역에 압력이 집중되는 경향이 관찰됩니다."
            ),
            "overallBalanceObservation": (
                "좌우 발에 전달된 압력이 비교적 고르게 분포하는 경향이 관찰됩니다."
            ),
            "practicalTip": "현재의 체중 분배를 유지하며 양발에 고르게 힘을 실어 주세요.",
        }
        create_response = AsyncMock(return_value=generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )

        self.assertEqual(facts["leftDominantZones"], ["HEEL"])
        self.assertEqual(facts["rightDominantZones"], ["MIDFOOT"])
        self.assertEqual(result.left_regional_observation, generated["leftRegionalObservation"])
        self.assertEqual(result.right_regional_observation, generated["rightRegionalObservation"])
        self.assertEqual(len(result.advice), 2)
        self.assertEqual(result.advice[0].count("\n"), 2)

    async def test_reversed_left_right_dominant_zones_use_fallback(self) -> None:
        left = [0.0] * 9 + [10.0, 0.0, 0.0]
        right = _right_hardware([0.0] * 5 + [10.0] + [0.0] * 6)
        facts = service.build_pressure_balance_facts(left, right)
        reversed_generated = {
            "leftRegionalObservation": (
                "왼발은 중족부 구역의 압력 비중이 가장 큰 경향이 관찰됩니다."
            ),
            "rightRegionalObservation": (
                "오른발은 뒤꿈치 구역의 압력 비중이 가장 큰 경향이 관찰됩니다."
            ),
            "overallBalanceObservation": (
                "좌우 발에 전달된 압력이 비교적 고르게 분포하는 경향이 관찰됩니다."
            ),
            "practicalTip": "현재의 체중 분배를 유지하며 양발에 고르게 힘을 실어 주세요.",
        }
        create_response = AsyncMock(return_value=reversed_generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )

        self.assertEqual(create_response.await_count, 5)
        self.assertIn("뒤꿈치", result.left_regional_observation)
        self.assertIn("중족부", result.right_regional_observation)

    async def test_balanced_facts_reject_invented_excessive_one_sided_claim(self) -> None:
        left = [1.0] * 12
        right = [1.0] * 12
        facts = service.build_pressure_balance_facts(left, right)
        generated = {
            "leftRegionalObservation": (
                "왼발은 앞꿈치 구역의 압력 비중이 가장 큰 경향이 관찰됩니다."
            ),
            "rightRegionalObservation": (
                "오른발은 앞꿈치 구역의 압력 비중이 가장 큰 경향이 관찰됩니다."
            ),
            "overallBalanceObservation": (
                "왼발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다."
            ),
            "practicalTip": "서 있을 때 양발에 천천히 고르게 힘을 실어 주세요.",
        }
        create_response = AsyncMock(return_value=generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )

        self.assertEqual(create_response.await_count, 5)
        self.assertEqual(result.score, 100.0)
        self.assertIn("좌우 발", result.overall_balance_observation)
        self.assertNotIn("크게 치우", result.overall_balance_observation)

    async def test_broad_distribution_wording_is_accepted_only_with_supporting_facts(self) -> None:
        left = [1.0] * 12
        right = [1.0] * 12
        facts = service.build_pressure_balance_facts(left, right)
        generated = {
            "leftRegionalObservation": (
                "왼발은 앞꿈치·중족부·뒤꿈치 세 구역에 압력이 비교적 고르게 분산되는 경향이 관찰됩니다."
            ),
            "rightRegionalObservation": (
                "오른발은 앞꿈치·중족부·뒤꿈치 세 구역에 압력이 비교적 고르게 분산되는 경향이 관찰됩니다."
            ),
            "overallBalanceObservation": (
                "좌우 발에 전달된 압력이 비교적 고르게 분포하는 경향이 관찰됩니다."
            ),
            "practicalTip": (
                "현재의 체중 분배를 유지하며 양발에 고르게 힘을 실어 주세요."
            ),
        }
        create_response = AsyncMock(return_value=generated)

        with (
            patch.object(service, "_openai_enabled", return_value=True),
            patch.object(service, "_create_structured_response", create_response),
        ):
            result = await service.generate_pressure_balance_report(
                left_pressure_values=left,
                right_pressure_values=right,
                pressure_facts=facts,
                left_heatmap_png=LEFT_PNG,
                right_heatmap_png=RIGHT_PNG,
            )

        create_response.assert_awaited_once()
        self.assertIn(
            "앞꿈치·중족부·뒤꿈치 세 구역",
            create_response.await_args.kwargs["user_payload"]["regional_contract"],
        )
        self.assertEqual(result.left_regional_observation, generated["leftRegionalObservation"])
        self.assertEqual(result.right_regional_observation, generated["rightRegionalObservation"])
        self.assertNotIn("정상", result.balance_comment)


if __name__ == "__main__":
    unittest.main()
