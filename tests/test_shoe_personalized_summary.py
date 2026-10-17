from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from app.schemas.shoe_server import ServerShoe, ServerUserFootState
from app.services.shoe.shoe_fit_policy import AreaScore, ComponentScore
from app.services.shoe.shoe_personalized_summary import (
    build_personalized_point_fallback,
    build_personalized_reason_summaries,
)


@dataclass(frozen=True)
class _Candidate:
    review_id: int
    sentence: str


def _state(*, measurements: bool = True) -> ServerUserFootState:
    return ServerUserFootState.model_validate(
        {
            "dailyFootAnalysis": (
                {
                    "balanceScore": 70.0,
                    "leftPressurePercent": 42.0,
                    "rightPressurePercent": 58.0,
                    "measuredLeftFootSizeMm": 260.0,
                    "measuredRightFootSizeMm": 262.0,
                    "leftFootWidthMm": 110.0,
                    "rightFootWidthMm": 111.0,
                    "avgTemperatureCelsius": 27.0,
                    "avgHumidityPercent": 68.0,
                    "typeText": None,
                }
                if measurements
                else None
            ),
            "tinaPedisAnalysis": (
                {
                    "fungalSuspicionSafetyScore": 95,
                    "skinReactionSafetyScore": 95,
                }
                if measurements
                else None
            ),
            "halluxValgusAnalysis": None,
            "staticPressureAnalyses": (
                [
                    {
                        "analysisId": 1,
                        "footSide": "LEFT",
                        "leftPressureRatio": 50.0,
                        "rightPressureRatio": 50.0,
                        "forefootPressureRatio": 78.0,
                        "rearfootPressureRatio": 22.0,
                        "centerOfPressureX": 0.0,
                        "centerOfPressureY": 0.0,
                        "balanceScore": 70.0,
                        "balanceStatus": "ATTENTION_NEEDED",
                        "analysisText": None,
                    },
                    {
                        "analysisId": 2,
                        "footSide": "RIGHT",
                        "leftPressureRatio": 50.0,
                        "rightPressureRatio": 50.0,
                        "forefootPressureRatio": 18.0,
                        "rearfootPressureRatio": 82.0,
                        "centerOfPressureX": 0.0,
                        "centerOfPressureY": 0.0,
                        "balanceScore": 70.0,
                        "balanceStatus": "ATTENTION_NEEDED",
                        "analysisText": None,
                    },
                ]
                if measurements
                else []
            ),
            "pressureSensorReadings": [],
        }
    )


def _shoe() -> ServerShoe:
    return ServerShoe.model_validate(
        {
            "shoeId": 90,
            "brandName": "호카",
            "shoeName": "클리프톤 9",
            "modelCode": "1127895-SSEG",
            "musinsaUrl": "https://example.com/90",
            "price": 169990,
            "imageUrl": None,
            "overallRating": 4.9,
            "reviewCount": 0,
            "reviews": [],
            "labMeasurements": [
                {
                    "measurementId": 900,
                    "source": "RUNREPEAT",
                    "sourceUrl": "https://runrepeat.com/example",
                    "sourceBrandName": "Hoka",
                    "sourceShoeName": "Clifton 9",
                    "sourceModelCode": None,
                    "testedSize": "US 9",
                    "capturedAt": "2026-09-01T00:00:00",
                    "parserVersion": "test-v1",
                    "internalLengthMm": 270.0,
                    "widthMm": 96.0,
                    "toeboxWidthMm": 72.0,
                    "toeboxHeightMm": 25.0,
                    "insoleThicknessMm": 4.0,
                    "heelStackMm": 35.0,
                    "forefootStackMm": 27.0,
                    "rawMetrics": [],
                }
            ],
        }
    )


def _areas(
    *,
    forefoot_risk: str = "HIGH",
    heel_risk: str = "HIGH",
    insole_risk: str = "MEDIUM",
) -> tuple[AreaScore, ...]:
    scores = {"LOW": 85.0, "MEDIUM": 65.0, "HIGH": 35.0}
    return (
        AreaScore(
            reason_type="FOREFOOT",
            score=scores[forefoot_risk],
            risk_level=forefoot_risk,
            title="발볼 적합도 주의",
            components=(
                ComponentScore("WIDTH_SPACE", 25.0, 0.55),
                ComponentScore("TOEBOX_SPACE", 45.0, 0.45),
            ),
        ),
        AreaScore(
            reason_type="HEEL",
            score=scores[heel_risk],
            risk_level=heel_risk,
            title="뒤꿈치 적합도 주의",
            components=(
                ComponentScore("SHOCK_ABSORPTION", 20.0, 0.35),
                ComponentScore("CUSHION", 55.0, 0.4),
            ),
        ),
        AreaScore(
            reason_type="INSOLE",
            score=scores[insole_risk],
            risk_level=insole_risk,
            title="깔창 적합도 보통",
            components=(
                ComponentScore("BREATHABILITY", 60.0, 1.0),
            ),
        ),
    )


def _candidates() -> dict[str, tuple[_Candidate, ...]]:
    return {
        "FOREFOOT": (
            _Candidate(1, "신발 발볼은 좁지만 반 사이즈 업하니 편했습니다."),
            _Candidate(2, "신발 발볼이 좁고 반 사이즈 업하니 잘 맞습니다."),
        ),
        "HEEL": (
            _Candidate(3, "바닥 충격이 크게 느껴지고 오래 신으면 발이 피로합니다."),
            _Candidate(4, "쿠션감이 부족해 장시간 착용하면 불편합니다."),
        ),
        "INSOLE": (
            _Candidate(5, "통풍이 잘 되어 신발 내부가 시원하고 쾌적합니다."),
            _Candidate(6, "통기성이 부족해 내부에 열감이 느껴집니다."),
        ),
    }


class PersonalizedShoeSummaryTests(unittest.TestCase):
    def test_uses_measurement_compatibility_and_consensus_as_natural_copy(self) -> None:
        summaries = build_personalized_reason_summaries(
            _state(), _areas(), _shoe(), _candidates()
        )

        self.assertEqual(set(summaries), {"FOREFOOT", "HEEL", "INSOLE"})
        self.assertIn("발볼", summaries["FOREFOOT"])
        self.assertIn("발볼", summaries["FOREFOOT"])
        self.assertIn("반 사이즈 업", summaries["FOREFOOT"])
        self.assertIn("뒤꿈치 쪽 하중", summaries["HEEL"])
        self.assertIn("충격", summaries["HEEL"])
        self.assertIn("발 내부 습도", summaries["INSOLE"])
        self.assertIn("통기", summaries["INSOLE"])
        for summary in summaries.values():
            self.assertGreaterEqual(summary.count("."), 2)
            self.assertLessEqual(summary.count("."), 3)
            self.assertNotIn("정량 분석 기준", summary)
            self.assertNotRegex(summary, r"리뷰\s*\d+개")

    def test_missing_measurements_do_not_create_pressure_or_humidity_claims(self) -> None:
        summaries = build_personalized_reason_summaries(
            _state(measurements=False), _areas(), _shoe(), {}
        )

        combined = " ".join(summaries.values())
        self.assertNotIn("측정 결과", combined)
        self.assertNotIn("하중", combined)
        self.assertNotIn("좌우 압력", combined)
        self.assertNotIn("습도", combined)

    def test_no_review_consensus_does_not_invent_size_or_thin_insole(self) -> None:
        candidates = {
            "FOREFOOT": (
                _Candidate(1, "발볼이 좁게 느껴졌습니다."),
                _Candidate(2, "발볼은 넉넉하게 느껴졌습니다."),
            ),
            "HEEL": (),
            "INSOLE": (_Candidate(3, "깔창이 얇게 느껴졌습니다."),),
        }

        summaries = build_personalized_reason_summaries(
            _state(), _areas(), _shoe(), candidates
        )

        self.assertNotIn("반 사이즈", summaries["FOREFOOT"])
        self.assertNotIn("한 사이즈", summaries["FOREFOOT"])
        self.assertEqual(
            summaries["INSOLE"],
            "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다.",
        )
        self.assertNotIn("얇", summaries["INSOLE"])

    def test_risk_levels_change_guidance_without_changing_input_facts(self) -> None:
        summaries = build_personalized_reason_summaries(
            _state(),
            _areas(forefoot_risk="HIGH", heel_risk="MEDIUM", insole_risk="LOW"),
            _shoe(),
            {},
        )

        expected = "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다."
        self.assertEqual(summaries["FOREFOOT"], expected)
        self.assertEqual(summaries["HEEL"], expected)
        self.assertEqual(summaries["INSOLE"], expected)

    def test_low_insole_risk_excludes_negative_and_cross_domain_reviews(
        self,
    ) -> None:
        areas = tuple(
            replace(area, title="깔창 적합도 좋음")
            if area.reason_type == "INSOLE"
            else area
            for area in _areas(
                forefoot_risk="LOW", heel_risk="LOW", insole_risk="LOW"
            )
        )
        candidates = {
            "INSOLE": (
                _Candidate(
                    16901,
                    "통풍이 잘 되고 신발 내부가 시원하고 쾌적합니다.",
                ),
                _Candidate(
                    16902,
                    "통기성이 좋아 오래 신어도 내부에 습기가 적습니다.",
                ),
                _Candidate(
                    16903,
                    "통풍이 부족해 신발 내부가 덥고 습기가 차서 답답합니다.",
                ),
            )
        }

        summary = build_personalized_reason_summaries(
            _state(), areas, _shoe(), candidates
        )["INSOLE"]
        insole_area = next(area for area in areas if area.reason_type == "INSOLE")

        # Review evidence must not rewrite the independently computed score/title.
        self.assertEqual(insole_area.risk_level, "LOW")
        self.assertEqual(insole_area.title, "깔창 적합도 좋음")

        self.assertIn("실제 착용 리뷰", summary)
        self.assertRegex(summary, r"통풍|통기")
        self.assertNotRegex(summary, r"덥|습기가 차|답답")
        self.assertNotIn("현재 발 상태와 비교적 잘 맞는 편", summary)
        self.assertNotIn("큰 주의점은 없지만", summary)
        self.assertEqual(summary.count("."), 3)

    def test_output_is_deterministic(self) -> None:
        inputs = (_state(), _areas(), _shoe(), _candidates())
        first = build_personalized_reason_summaries(*inputs)
        second = build_personalized_reason_summaries(*inputs)
        self.assertEqual(first, second)

    def test_point_fallback_is_natural_non_null_and_does_not_invent_style(self) -> None:
        summary = build_personalized_point_fallback(_areas())

        self.assertIn("발볼과 앞코", summary)
        self.assertIn("충격 완화·반발력·쿠션", summary)
        self.assertIn("제품 통기성", summary)
        self.assertNotIn("리뷰", summary)
        self.assertNotIn("청바지", summary)
        self.assertNotIn("반 사이즈", summary)
        self.assertEqual(summary.count("."), 3)


if __name__ == "__main__":
    unittest.main()
