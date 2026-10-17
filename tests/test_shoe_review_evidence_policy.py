from __future__ import annotations

import unittest

from app.schemas.shoe_server import ServerShoe, ServerUserFootState
from app.services.shoe.shoe_feature_rules import default_title, polarity_score
from app.services.shoe.shoe_fit_policy import AreaScore, ComponentScore
from app.services.shoe.shoe_personalized_summary import (
    build_personalized_reason_summaries,
)
from app.services.shoe.shoe_recommendation import (
    ReviewCandidate,
    select_review_evidence,
)


_NO_REVIEW_SUMMARY = (
    "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다."
)
_REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")


def _candidate(review_id: int, sentence: str, similarity: float = 0.9) -> ReviewCandidate:
    return ReviewCandidate(
        review_id=review_id,
        sentence=sentence,
        similarity=similarity,
    )


def _positive(reason_type: str, review_id: int, similarity: float = 0.9) -> ReviewCandidate:
    text = {
        "FOREFOOT": "발볼이 넉넉하고 발가락도 편해요.",
        "HEEL": "쿠션감이 좋고 오래 걸어도 발 부담이 적어서 편해요.",
        "INSOLE": "통풍이 잘 되고 신발 내부가 시원하고 쾌적해서 좋아요.",
    }[reason_type]
    return _candidate(review_id, text, similarity)


def _negative(reason_type: str, review_id: int, similarity: float = 0.9) -> ReviewCandidate:
    text = {
        "FOREFOOT": "발볼이 좁아서 발가락이 아파요.",
        "HEEL": "바닥 충격이 크게 느껴지고 오래 걸으면 발이 피로하고 불편해요.",
        "INSOLE": "통기성이 부족해 신발 내부에 열감과 습기가 차서 답답해요.",
    }[reason_type]
    return _candidate(review_id, text, similarity)


def _areas(risk_level: str = "MEDIUM") -> tuple[AreaScore, ...]:
    score = {"LOW": 85.0, "MEDIUM": 65.0, "HIGH": 35.0}[risk_level]
    component = {
        "FOREFOOT": "WIDTH_SPACE",
        "HEEL": "SHOCK_ABSORPTION",
        "INSOLE": "BREATHABILITY",
    }
    return tuple(
        AreaScore(
            reason_type=reason_type,
            score=score,
            risk_level=risk_level,
            title=default_title(reason_type, risk_level),
            components=(ComponentScore(component[reason_type], score, 1.0),),
        )
        for reason_type in _REASON_TYPES
    )


def _empty_state() -> ServerUserFootState:
    return ServerUserFootState.model_validate(
        {
            "dailyFootAnalysis": None,
            "tinaPedisAnalysis": None,
            "halluxValgusAnalysis": None,
            "staticPressureAnalyses": [],
            "pressureSensorReadings": [],
        }
    )


def _shoe() -> ServerShoe:
    return ServerShoe.model_validate(
        {
            "shoeId": 901,
            "brandName": "테스트",
            "shoeName": "근거 정책 테스트 신발",
            "modelCode": "POLICY-901",
            "musinsaUrl": "https://example.com/shoes/901",
            "price": 100000,
            "imageUrl": None,
            "overallRating": 4.5,
            "reviewCount": 0,
            "reviews": [],
            "labMeasurements": [
                {
                    "measurementId": 9010,
                    "source": "RUNREPEAT",
                    "sourceUrl": "https://runrepeat.com/policy-901",
                    "sourceBrandName": "Test",
                    "sourceShoeName": "Policy 901",
                    "sourceModelCode": None,
                    "testedSize": "US 9",
                    "capturedAt": "2026-09-06T00:00:00",
                    "parserVersion": "policy-test-v1",
                    "internalLengthMm": 270.0,
                    "widthMm": 100.0,
                    "toeboxWidthMm": 75.0,
                    "toeboxHeightMm": 25.0,
                    "insoleThicknessMm": 4.0,
                    "heelStackMm": 30.0,
                    "forefootStackMm": 22.0,
                    "rawMetrics": [],
                }
            ],
        }
    )


class ReviewEvidenceSelectionPolicyTests(unittest.TestCase):
    def test_each_reason_keeps_only_its_own_review_clause(self) -> None:
        mixed = _candidate(
            1,
            "발볼이 넉넉해서 편해요. 뒤꿈치는 쓸려서 아파요. "
            "통풍은 잘 되고 신발 내부가 쾌적해요.",
        )

        selected = {
            reason_type: select_review_evidence(reason_type, "MEDIUM", (mixed,))
            for reason_type in _REASON_TYPES
        }

        self.assertEqual([item.review_id for item in selected["FOREFOOT"]], [1])
        self.assertIn("발볼", selected["FOREFOOT"][0].sentence)
        self.assertNotIn("뒤꿈치", selected["FOREFOOT"][0].sentence)
        self.assertNotIn("쿠션", selected["FOREFOOT"][0].sentence)

        self.assertEqual([item.review_id for item in selected["HEEL"]], [1])
        self.assertIn("뒤꿈치", selected["HEEL"][0].sentence)
        self.assertNotIn("발볼", selected["HEEL"][0].sentence)
        self.assertNotIn("쿠션", selected["HEEL"][0].sentence)

        self.assertEqual([item.review_id for item in selected["INSOLE"]], [1])
        self.assertIn("통풍", selected["INSOLE"][0].sentence)
        self.assertNotIn("발볼", selected["INSOLE"][0].sentence)
        self.assertNotIn("뒤꿈치", selected["INSOLE"][0].sentence)

    def test_low_risk_selects_positive_reviews_only(self) -> None:
        for index, reason_type in enumerate(_REASON_TYPES, start=1):
            with self.subTest(reason_type=reason_type):
                positive = _positive(reason_type, index * 10 + 1, 0.8)
                negative = _negative(reason_type, index * 10 + 2, 0.99)

                selected = select_review_evidence(
                    reason_type, "LOW", (negative, positive)
                )

                self.assertEqual([item.review_id for item in selected], [positive.review_id])
                self.assertTrue(all(polarity_score(item.sentence) > 0 for item in selected))

    def test_high_risk_selects_negative_reviews_only(self) -> None:
        for index, reason_type in enumerate(_REASON_TYPES, start=1):
            with self.subTest(reason_type=reason_type):
                positive = _positive(reason_type, index * 10 + 1, 0.99)
                negative = _negative(reason_type, index * 10 + 2, 0.8)

                selected = select_review_evidence(
                    reason_type, "HIGH", (positive, negative)
                )

                self.assertEqual([item.review_id for item in selected], [negative.review_id])
                self.assertTrue(all(polarity_score(item.sentence) < 0 for item in selected))

    def test_medium_risk_represents_both_directions_and_never_exceeds_three(self) -> None:
        candidates = (
            _positive("INSOLE", 101, 0.99),
            _positive("INSOLE", 102, 0.98),
            _negative("INSOLE", 103, 0.97),
            _negative("INSOLE", 104, 0.96),
            _positive("INSOLE", 105, 0.95),
        )

        selected = select_review_evidence("INSOLE", "MEDIUM", candidates)
        polarities = [polarity_score(item.sentence) for item in selected]

        self.assertLessEqual(len(selected), 3)
        self.assertEqual(len({item.review_id for item in selected}), len(selected))
        self.assertTrue(any(value > 0 for value in polarities))
        self.assertTrue(any(value < 0 for value in polarities))

    def test_no_reviews_returns_empty_ids_and_exact_summary_fallback(self) -> None:
        selected = {
            reason_type: select_review_evidence(reason_type, "MEDIUM", ())
            for reason_type in _REASON_TYPES
        }
        summaries = build_personalized_reason_summaries(
            _empty_state(), _areas("MEDIUM"), _shoe(), selected
        )

        for reason_type in _REASON_TYPES:
            with self.subTest(reason_type=reason_type):
                self.assertEqual(selected[reason_type], ())
                self.assertEqual(
                    [item.review_id for item in selected[reason_type]], []
                )
                self.assertEqual(
                    [item.sentence for item in selected[reason_type]], []
                )
                self.assertEqual(summaries[reason_type], _NO_REVIEW_SUMMARY)

    def test_summary_does_not_add_product_judgments_beyond_selected_review(self) -> None:
        selected_forefoot = select_review_evidence(
            "FOREFOOT",
            "LOW",
            (_candidate(201, "발볼이 넉넉해서 편해요."),),
        )
        summaries = build_personalized_reason_summaries(
            _empty_state(),
            _areas("LOW"),
            _shoe(),
            {"FOREFOOT": selected_forefoot, "HEEL": (), "INSOLE": ()},
        )
        summary = summaries["FOREFOOT"]

        self.assertIn("발볼", summary)
        self.assertIn("여유", summary)
        for unsupported_product_judgment in (
            "발볼이 좁",
            "반 사이즈",
            "정사이즈",
            "사이즈 업",
            "쿠션",
            "통풍",
            "가볍",
        ):
            self.assertNotIn(unsupported_product_judgment, summary)

    def test_heel_lock_lacing_is_not_interpreted_as_product_heel_evidence(self) -> None:
        lacing_techniques = (
            _candidate(301, "힐락으로 끈을 묶으니 뒤꿈치가 잘 잡혀요.", 0.99),
            _candidate(302, "heel-lock으로 묶으면 뒤꿈치가 고정돼요.", 0.98),
            _candidate(303, "힐 락 레이싱을 하니 뒤꿈치가 안정적이에요.", 0.97),
        )
        actual_product_experience = _negative(
            "HEEL", 304, 0.8
        )

        selected = select_review_evidence(
            "HEEL", "MEDIUM", (*lacing_techniques, actual_product_experience)
        )
        selected_ids = {item.review_id for item in selected}

        self.assertEqual(selected_ids, {304})
        self.assertNotRegex(
            " ".join(item.sentence for item in selected),
            r"(?i)heel[ -]?lock|힐\s*락|레이싱",
        )

    def test_direct_heel_hold_wording_is_not_rewritten_as_impact_comfort(self) -> None:
        candidates = (
            _candidate(401, "뒤꿈치를 잘 잡아줘서 안정적이에요."),
            _candidate(402, "뒤꿈치 고정감이 좋아요."),
            _candidate(403, "발목을 안정적으로 잡아줘요."),
            _candidate(404, "끈을 조이면 발목이 안정적으로 잡혀요."),
            _candidate(405, "뒤축을 안정적으로 잡아줘요."),
            _candidate(406, "쿠션감이 좋고 오래 걸어도 발 부담이 적어요."),
        )

        selected = select_review_evidence("HEEL", "LOW", candidates)

        self.assertEqual([item.review_id for item in selected], [406])

    def test_design_context_is_not_rewritten_as_fit_evidence(self) -> None:
        self.assertEqual(
            select_review_evidence(
                "FOREFOOT",
                "LOW",
                (_candidate(451, "사이즈는 모르겠지만 디자인이 좋아요."),),
            ),
            (),
        )
        self.assertEqual(
            select_review_evidence(
                "FOREFOOT", "LOW", (_candidate(452, "발볼 디자인이 좋아요."),)
            ),
            (),
        )
        self.assertEqual(
            select_review_evidence(
                "HEEL", "LOW", (_candidate(453, "착화감 디자인이 좋아요."),)
            ),
            (),
        )

    def test_negated_cautions_are_not_misclassified_as_negative(self) -> None:
        forefoot = select_review_evidence(
            "FOREFOOT", "LOW", (_candidate(501, "발볼이 좁지 않고 편해요."),)
        )
        insole = select_review_evidence(
            "INSOLE", "LOW", (_candidate(502, "통풍이 나쁘지 않고 내부가 쾌적해요."),)
        )

        self.assertEqual([item.review_id for item in forefoot], [501])
        self.assertEqual([item.review_id for item in insole], [502])

    def test_medium_keeps_both_directions_even_with_legacy_limit_one(self) -> None:
        selected = select_review_evidence(
            "INSOLE",
            "MEDIUM",
            (_positive("INSOLE", 601), _negative("INSOLE", 602)),
            limit=1,
        )

        self.assertEqual({item.review_id for item in selected}, {601, 602})

    def test_full_review_with_unselected_forefoot_terms_is_not_display_safe(self) -> None:
        cases = (
            (
                "HEEL",
                ReviewCandidate(
                    review_id=701,
                    sentence="쿠션감이 좋아요.",
                    similarity=0.99,
                    source_text="쿠션감이 좋아요. 발등이 눌려 아파요.",
                ),
            ),
            (
                "INSOLE",
                ReviewCandidate(
                    review_id=702,
                    sentence="통풍이 좋아요.",
                    similarity=0.99,
                    source_text="통풍이 좋아요. 앞꿈치가 아파요.",
                ),
            ),
            (
                "HEEL",
                ReviewCandidate(
                    review_id=703,
                    sentence="쿠션감이 좋아요.",
                    similarity=0.99,
                    source_text="쿠션감이 좋아요. 전족부가 아파요.",
                ),
            ),
            (
                "INSOLE",
                ReviewCandidate(
                    review_id=704,
                    sentence="통풍이 좋아요.",
                    similarity=0.99,
                    source_text="통풍이 좋아요. 앞발이 아파요.",
                ),
            ),
        )

        for reason_type, candidate in cases:
            with self.subTest(reason_type=reason_type, source_text=candidate.source_text):
                self.assertEqual(
                    select_review_evidence(reason_type, "LOW", (candidate,)), ()
                )

    def test_bare_insole_thickness_or_internal_design_is_not_fit_evidence(self) -> None:
        selected = select_review_evidence(
            "INSOLE",
            "HIGH",
            (
                _candidate(801, "깔창이 얇아요."),
                _candidate(802, "신발 내부 디자인이 좋아요."),
            ),
        )

        self.assertEqual(selected, ())


if __name__ == "__main__":
    unittest.main()
