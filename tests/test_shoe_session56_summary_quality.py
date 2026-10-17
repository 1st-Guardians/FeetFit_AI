from __future__ import annotations

from dataclasses import dataclass
import re
import unittest
from unittest.mock import patch

from app.schemas.shoe_server import (
    ServerRecommendationContext,
    ServerShoe,
    ServerUserFootState,
)
from app.services.shoe import shoe_recommendation
from app.services.shoe.shoe_fit_policy import AreaScore, ComponentScore
from app.services.shoe.shoe_personalized_summary import (
    build_personalized_reason_summaries,
)
from app.services.shoe.shoe_recommendation import ReviewCandidate, ShoeFacts


_REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")
_MECHANICAL_POINT_COPY = re.compile(
    r"정량 분석 기준|관련 착화 리뷰\s*\d+개|"
    r"(?:발볼 공간|앞코 공간|쿠션감|반발력)은\s*"
    r"(?:낮은 편|보통 수준|높은 편)입니다|"
    r"현재 발 치수와 제품 실측값을 비교하면|"
    r"뒤꿈치 지지와 충격 완화는|"
    r"깔창의 쿠션과 통기성은"
)


@dataclass(frozen=True)
class _Session56Case:
    label: str
    shoe: ServerShoe
    shortlists: dict[str, tuple[ReviewCandidate, ...]]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _review(review_id: int, review_text: str) -> dict:
    return {
        "reviewId": review_id,
        "rating": 5.0,
        "reviewText": review_text,
        "source": "MUSINSA",
        "collectedAt": None,
    }


def _shoe(shoe_id: int, shoe_name: str, reviews: list[dict]) -> ServerShoe:
    return ServerShoe.model_validate(
        {
            "shoeId": shoe_id,
            "brandName": "테스트 브랜드",
            "shoeName": shoe_name,
            "modelCode": f"SESSION56-{shoe_id}",
            "musinsaUrl": f"https://example.com/shoes/{shoe_id}",
            "price": 100000,
            "imageUrl": None,
            "overallRating": 4.8,
            "reviewCount": len(reviews),
            "reviews": reviews,
            "labMeasurements": [
                {
                    "measurementId": 56_000 + shoe_id,
                    "source": "RUNREPEAT",
                    "sourceUrl": f"https://runrepeat.com/session56-{shoe_id}",
                    "sourceBrandName": "Test Brand",
                    "sourceShoeName": shoe_name,
                    "sourceModelCode": None,
                    "testedSize": "US 9",
                    "capturedAt": "2026-09-06T00:00:00",
                    "parserVersion": "session56-regression",
                    "internalLengthMm": 270.0,
                    "widthMm": 98.0,
                    "toeboxWidthMm": 72.0,
                    "toeboxHeightMm": 25.0,
                    "insoleThicknessMm": 5.0,
                    "heelStackMm": 34.0,
                    "forefootStackMm": 25.0,
                    "rawMetrics": [],
                }
            ],
        }
    )


def _foot_state_without_pressure() -> ServerUserFootState:
    return ServerUserFootState.model_validate(
        {
            "dailyFootAnalysis": {
                "balanceScore": 70.0,
                "leftPressurePercent": None,
                "rightPressurePercent": None,
                "measuredLeftFootSizeMm": 260.0,
                "measuredRightFootSizeMm": 262.0,
                "leftFootWidthMm": 110.0,
                "rightFootWidthMm": 111.0,
                "avgTemperatureCelsius": 27.0,
                "avgHumidityPercent": 40.0,
                "typeText": None,
            },
            "tinaPedisAnalysis": None,
            "halluxValgusAnalysis": None,
            "staticPressureAnalyses": [],
            "pressureSensorReadings": [],
        }
    )


def _foot_state_with_split_pressure() -> ServerUserFootState:
    raw = _foot_state_without_pressure().model_dump(by_alias=True)
    raw["staticPressureAnalyses"] = [
        {
            "analysisId": 5601,
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
            "analysisId": 5602,
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
    return ServerUserFootState.model_validate(raw)


def _areas() -> tuple[AreaScore, ...]:
    return (
        AreaScore(
            reason_type="FOREFOOT",
            score=35.0,
            risk_level="HIGH",
            title="발볼 적합도 주의",
            components=(
                ComponentScore("WIDTH_SPACE", 25.0, 0.55),
                ComponentScore("TOEBOX_SPACE", 45.0, 0.45),
            ),
        ),
        AreaScore(
            reason_type="HEEL",
            score=65.0,
            risk_level="MEDIUM",
            title="뒤꿈치 적합도 보통",
            components=(
                ComponentScore("SHOCK_ABSORPTION", 60.0, 0.5),
                ComponentScore("ENERGY_RETURN", 68.0, 0.5),
            ),
        ),
        AreaScore(
            reason_type="INSOLE",
            score=65.0,
            risk_level="MEDIUM",
            title="깔창 적합도 보통",
            components=(
                ComponentScore("BREATHABILITY", 68.0, 1.0),
            ),
        ),
    )


def _candidate(shoe: ServerShoe, review_id: int) -> ReviewCandidate:
    review = next(review for review in shoe.reviews if review.id == review_id)
    return ReviewCandidate(
        review.id,
        review.review_text,
        0.95,
        source_text=review.review_text,
    )


def _session56_cases() -> tuple[_Session56Case, ...]:
    shoe_2 = _shoe(
        2,
        "세션 56 신발 2",
        [
            _review(
                201,
                "발볼이 여유롭고 편하며 신발이 가볍고 쿠션이 훅신해서 데일리로 신기 좋습니다.",
            ),
            _review(
                202,
                "신발이 가볍고 쿠션이 훅신해 오래 신어도 편했습니다.",
            ),
            _review(
                203,
                "쿠션이 부드럽고 가벼워서 오래 걸어도 편하며 데일리로 활용하기 좋습니다.",
            ),
        ],
    )
    shoe_46 = _shoe(
        46,
        "세션 56 신발 46",
        [
            _review(4601, "발볼이 좁게 느껴졌지만 디자인은 깔끔하고 마음에 듭니다."),
            _review(4602, "D 와이드라 발볼이 여유롭고 디자인도 예쁩니다."),
            _review(4603, "쿠션이 훅신하고 오래 신어도 편하며 반발감도 좋습니다."),
            _review(4604, "쿠션이 부드럽고 장시간 착용해도 편하고 반발감이 좋습니다."),
            _review(4605, "훅신한 쿠션 덕분에 오래 걸어도 편하고 탄력감이 좋습니다."),
        ],
    )
    shoe_189 = _shoe(
        189,
        "세션 56 신발 189",
        [
            _review(18901, "발볼이 여유롭고 전체적으로 딱 맞는 핏이며 디자인이 예쁘고 신발이 가볍습니다."),
            _review(18902, "발볼 공간이 넉넉하고 핏이 잘 맞으며 디자인이 깔끔하고 가벼습니다."),
            _review(18903, "발볼이 편안할 정도로 여유가 있고 핏과 디자인이 마음에 들며 무게도 가볍습니다."),
            _review(18904, "깔창이 걸을 때마다 밀려서 발 안에서 미끄럽게 느껴졌습니다."),
            _review(18905, "쿠션이 훅신하고 신발도 가볍습니다."),
            _review(18906, "쿠션이 부드럽고 신을 때 가볍게 느껴집니다."),
            _review(18907, "쿠션은 부드럽지만 밑창이 불편하고 깔창이 한 번 밀렸습니다."),
        ],
    )

    return (
        _Session56Case(
            "shoe-2-single-roomy-review",
            shoe_2,
            {
                "FOREFOOT": (_candidate(shoe_2, 201),),
                "HEEL": (),
                "INSOLE": (_candidate(shoe_2, 201),),
            },
        ),
        _Session56Case(
            "shoe-46-conflicting-width",
            shoe_46,
            {
                "FOREFOOT": (
                    _candidate(shoe_46, 4601),
                    _candidate(shoe_46, 4602),
                ),
                "HEEL": (),
                "INSOLE": (
                    _candidate(shoe_46, 4603),
                    _candidate(shoe_46, 4604),
                    _candidate(shoe_46, 4605),
                ),
            },
        ),
        _Session56Case(
            "shoe-189-single-slip-report",
            shoe_189,
            {
                "FOREFOOT": (
                    _candidate(shoe_189, 18901),
                    _candidate(shoe_189, 18902),
                    _candidate(shoe_189, 18903),
                ),
                "HEEL": (_candidate(shoe_189, 18904),),
                "INSOLE": (
                    _candidate(shoe_189, 18905),
                    _candidate(shoe_189, 18906),
                    _candidate(shoe_189, 18907),
                ),
            },
        ),
    )


def _score_case(case: _Session56Case) -> ShoeFacts:
    state = _foot_state_without_pressure()
    context = ServerRecommendationContext(
        measurement_session_id=56,
        user_id=7,
        measurement_status="COMPLETED",
        foot_state=state,
        shoes=[case.shoe],
    )
    with (
        patch(
            "app.services.shoe.shoe_recommendation.score_shoe_fit",
            return_value=_areas(),
        ),
        patch(
            "app.services.shoe.shoe_recommendation._semantic_shortlists",
            return_value=case.shortlists,
        ),
    ):
        return shoe_recommendation._score_shoe_facts(
            case.shoe,
            context,
            query_vectors={reason_type: object() for reason_type in _REASON_TYPES},
        )


class Session56PointSummaryQualityTests(unittest.TestCase):
    def test_point_summary_integrates_repeated_reviews_in_two_or_three_sentences(
        self,
    ) -> None:
        for case in _session56_cases():
            with self.subTest(case=case.label):
                point_summary = _score_case(case).point_summary
                self.assertIsNotNone(point_summary)
                sentences = _sentences(point_summary or "")
                self.assertGreaterEqual(len(sentences), 2)
                self.assertLessEqual(len(sentences), 3)
                self.assertNotRegex(point_summary or "", _MECHANICAL_POINT_COPY)
                self.assertNotIn("현재 발 상태", point_summary or "")

    def test_point_summary_drops_single_or_conflicting_review_claims(self) -> None:
        results = {case.shoe.id: _score_case(case).point_summary or "" for case in _session56_cases()}

        # shoe 2 has only one roomy-width report; the repeated cushion, lightness,
        # comfort and daily-use evidence may be summarized, but width may not.
        self.assertNotIn("발볼", results[2])
        repeated_topics = sum(
            term in results[2] for term in ("쿠션", "편안", "가볍", "데일리")
        )
        self.assertGreaterEqual(repeated_topics, 2)

        # shoe 46 has one narrow and one roomy report. Neither direction is a
        # product-level consensus, while cushion and design do repeat.
        self.assertNotIn("발볼", results[46])
        self.assertNotIn("좁", results[46])
        self.assertNotIn("여유", results[46])
        self.assertIn("쿠션", results[46])
        self.assertIn("디자인", results[46])

        # shoe 189 has one slipping/discomfort report among otherwise repeated
        # roomy, light and cushioned feedback. The singleton must not leak.
        self.assertIn("발볼", results[189])
        self.assertIn("딱 맞는 핏", results[189])
        self.assertIn("가볍", results[189])
        for singleton_claim in ("밀려", "밀림", "미끄", "불편"):
            self.assertNotIn(singleton_claim, results[189])


class Session56ReasonSummaryQualityTests(unittest.TestCase):
    def test_insole_summary_uses_humidity_and_breathability_reviews_only(self) -> None:
        shoe = _shoe(
            560,
            "통기성 회귀 신발",
            [
                _review(5601, "통풍이 잘 되어 신발 내부가 시원하고 쾌적했어요."),
                _review(5602, "통기성이 좋아 내부에 땀이 적고 편안했습니다."),
                _review(5603, "통기성이 부족해 신발 내부에 열감이 느껴졌습니다."),
            ],
        )
        candidates = {
            "FOREFOOT": (),
            "HEEL": (),
            "INSOLE": tuple(_candidate(shoe, review.id) for review in shoe.reviews),
        }

        state_raw = _foot_state_with_split_pressure().model_dump(by_alias=True)
        state_raw["dailyFootAnalysis"]["avgHumidityPercent"] = 80.0
        summary = build_personalized_reason_summaries(
            ServerUserFootState.model_validate(state_raw), _areas(), shoe, candidates
        )["INSOLE"]
        sentences = _sentences(summary)

        self.assertEqual(len(sentences), 3)
        self.assertIn("발 내부 습도", sentences[0])
        self.assertIn("통기성", sentences[1])
        self.assertIn("실제 착용 리뷰", sentences[2])
        self.assertRegex(sentences[2], r"통풍|통기")
        self.assertNotRegex(summary, r"하중|충격|뒤꿈치")

    def test_each_reason_has_measurement_product_and_action_sentences(self) -> None:
        for case in _session56_cases():
            facts = _score_case(case)
            for reason in facts.reasons:
                with self.subTest(case=case.label, reason=reason.reason_type):
                    summary = reason.review_summary or ""
                    sentences = _sentences(summary)
                    if not reason.review_ids:
                        self.assertEqual(
                            summary,
                            "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다.",
                        )
                        continue
                    self.assertEqual(len(sentences), 3)
                    self.assertTrue(sentences[0].startswith("측정 결과"))
                    self.assertTrue(sentences[1].startswith("제품의 "))
                    self.assertRegex(sentences[1], r"비교한 결과|계산 결과")
                    self.assertIn("실제 착용 리뷰", sentences[2])

    def test_missing_pressure_thin_insole_and_sizing_evidence_is_not_invented(
        self,
    ) -> None:
        for case in _session56_cases():
            facts = _score_case(case)
            summaries = " ".join(
                [
                    facts.point_summary or "",
                    *(reason.review_summary or "" for reason in facts.reasons),
                ]
            )
            with self.subTest(case=case.label):
                for unsupported in (
                    "하중",
                    "압력",
                    "깔창이 얇",
                    "얇은 깔창",
                    "바닥이 얇",
                    "반 사이즈",
                    "한 사이즈",
                    "사이즈 업",
                    "사이즈 다운",
                    "여유 있는 사이즈",
                ):
                    self.assertNotIn(unsupported, summaries)

    def test_pressure_is_not_used_by_forefoot_or_insole_without_reviews(self) -> None:
        case = _session56_cases()[0]
        raw = _foot_state_without_pressure().model_dump(by_alias=True)
        daily = raw["dailyFootAnalysis"]
        daily["measuredLeftFootSizeMm"] = None
        daily["measuredRightFootSizeMm"] = None
        daily["leftFootWidthMm"] = None
        daily["rightFootWidthMm"] = None
        daily["avgHumidityPercent"] = None
        raw["staticPressureAnalyses"] = [
            {
                "analysisId": 5603,
                "footSide": "LEFT",
                "leftPressureRatio": 50.0,
                "rightPressureRatio": 50.0,
                "forefootPressureRatio": 40.0,
                "rearfootPressureRatio": 40.0,
                "centerOfPressureX": 0.0,
                "centerOfPressureY": 0.0,
                "balanceScore": 70.0,
                "balanceStatus": "VERY_GOOD",
                "analysisText": None,
            }
        ]
        summaries = build_personalized_reason_summaries(
            ServerUserFootState.model_validate(raw),
            _areas(),
            case.shoe,
            case.shortlists,
        )
        forefoot_summary = summaries["FOREFOOT"]
        insole_summary = summaries["INSOLE"]

        expected = "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다."
        self.assertEqual(forefoot_summary, expected)
        self.assertEqual(insole_summary, expected)
        self.assertNotRegex(forefoot_summary + insole_summary, r"전족부|뒤꿈치|하중")

    def test_conflict_and_singleton_reason_reviews_are_not_generalized(self) -> None:
        results = {case.shoe.id: _score_case(case) for case in _session56_cases()}
        by_reason = {
            shoe_id: {reason.reason_type: reason.review_summary or "" for reason in facts.reasons}
            for shoe_id, facts in results.items()
        }

        # The two shoe-46 width reviews disagree, so neither direction can be
        # introduced as a repeated wearing-experience conclusion.
        shoe_46_forefoot = by_reason[46]["FOREFOOT"]
        self.assertNotIn("좁게 느껴", shoe_46_forefoot)
        self.assertNotIn("여유롭게 느껴", shoe_46_forefoot)

        # An insole that slipped once is not heel-hold evidence and cannot be
        # presented as a repeated HEEL product property.
        shoe_189_heel = by_reason[189]["HEEL"]
        for unsupported in ("깔창", "밀려", "미끄", "들뜨", "헐렁"):
            self.assertNotIn(unsupported, shoe_189_heel)


if __name__ == "__main__":
    unittest.main()
