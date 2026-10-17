from __future__ import annotations

import unittest

from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.schemas.shoe_server import ServerShoeCharacteristics
from app.services.shoe.shoe_fit_comment_service import (
    ReasonFactsForPrompt,
    ShoeFitCommentError,
    _generated_point_summary_keeps_important_evidence,
    _point_claims_are_traceable,
    prepare_point_summary_evidence,
)
from app.services.shoe.shoe_point_summary import (
    compose_product_point_summary,
    join_point_summary_claims,
    point_evidence_strength,
    prepare_point_evidence_facts,
    render_exact_lab_claim,
    render_product_point_summary,
)


def _empty_reason(reason_type: str) -> ReasonFactsForPrompt:
    return ReasonFactsForPrompt(
        reason_type=reason_type,
        title=f"{reason_type} title",
        risk_level="LOW",
        review_ids=[],
        review_texts=[],
    )


def _reasons(
    *pairs: tuple[str, list[int], list[str]],
) -> list[ReasonFactsForPrompt]:
    supplied = {reason_type: (ids, texts) for reason_type, ids, texts in pairs}
    return [
        ReasonFactsForPrompt(
            reason_type=reason_type,
            title=f"{reason_type} title",
            risk_level="LOW",
            review_ids=supplied.get(reason_type, ([], []))[0],
            review_texts=supplied.get(reason_type, ([], []))[1],
        )
        for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
    ]


def _characteristics(*rows: tuple[str, str]) -> ServerShoeCharacteristics:
    return ServerShoeCharacteristics.model_validate(
        {
            "shoeId": 1,
            "summary": "RunRepeat 특성",
            "characteristics": [
                {
                    "type": characteristic,
                    "level": level,
                    "value": 1,
                    "averageValue": 1,
                    "minValue": 0,
                    "maxValue": 2,
                    "unit": "score",
                    "testedSize": None,
                }
                for characteristic, level in rows
            ],
        }
    )


class PointEvidenceAggregationTests(unittest.TestCase):
    def test_categories_and_review_ids_remain_traceable(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101],
                    ["발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌습니다."],
                ),
                ("HEEL", [201], ["발목을 안정적으로 잡아줍니다."]),
                (
                    "INSOLE",
                    [301],
                    ["바닥이 얇아서 오래 서 있으면 발바닥이 아프고 무게감도 있습니다."],
                ),
            )
        )
        by_id = {fact.evidence_id: fact for fact in facts}
        self.assertEqual(
            by_id["MUSINSA:SIZE_FIT:SIZE_OPTION:FULL_UP:NEUTRAL"].review_ids,
            (101,),
        )
        self.assertIn("MUSINSA:HEEL_FEEL:HEEL_HOLD:STABLE:POSITIVE", by_id)
        self.assertIn(
            "MUSINSA:LONG_WEAR:LONG_WEAR_COMFORT:DISCOMFORT:NEGATIVE",
            by_id,
        )
        self.assertIn("MUSINSA:WEIGHT_FEEL:WEIGHT:HEAVY:NEGATIVE", by_id)

    def test_same_review_across_reason_slots_counts_once(self) -> None:
        body = "발목을 안정적으로 잡아줍니다."
        facts = prepare_point_summary_evidence(
            _reasons(
                ("FOREFOOT", [101], [body]),
                ("HEEL", [101], [body]),
                ("INSOLE", [101], [body]),
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.review_ids, (101,))
        self.assertEqual(stable.mode, "SINGLE")

    def test_duplicate_bodies_do_not_create_false_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [109, 101],
                    ["발목이 안정적이에요.", "\u200b발목이  안정적이에요.\n"],
                )
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.review_ids, (101,))
        self.assertEqual(stable.mode, "SINGLE")

    def test_same_review_id_with_different_body_fails_closed(self) -> None:
        with self.assertRaisesRegex(ShoeFitCommentError, "more than one review body"):
            prepare_point_summary_evidence(
                _reasons(
                    ("FOREFOOT", [101], ["발볼이 편해요."]),
                    ("HEEL", [101], ["발목이 안정적이에요."]),
                )
            )

    def test_distinct_reviews_same_signal_become_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                )
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.mode, "CONSENSUS")
        self.assertEqual(stable.review_ids, (101, 102))

    def test_evidence_strength_uses_independent_support_count(self) -> None:
        for review_count, expected in ((1, "LOW"), (2, "MEDIUM"), (3, "HIGH")):
            ids = list(range(101, 101 + review_count))
            texts = [f"발목을 안정적으로 잡아줘요 {index}" for index in ids]
            facts = prepare_point_summary_evidence(
                _reasons(("HEEL", ids, texts))
            )
            stable = next(fact for fact in facts if fact.signal.value == "STABLE")
            self.assertEqual(point_evidence_strength(stable, facts), expected)

    def test_runrepeat_and_review_width_conflict_is_explicit(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["발볼이 좁아서 불편해요.", "앞볼이 좁고 타이트해요."],
                )
            ),
            _characteristics(("WIDTH_SPACE", "HIGH")),
        )
        width = [
            fact
            for fact in facts
            if fact.signal.subject in {"WIDTH_SPACE", "LAB_WIDTH_SPACE"}
        ]
        self.assertEqual({fact.mode for fact in width}, {"CONFLICT"})

    def test_equal_full_and_half_up_consensus_is_downgraded(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["한 사이즈 업이 잘 맞아요.", "1업했더니 편해요."],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["반 사이즈 업이 잘 맞아요.", "반업했더니 편해요."],
                ),
            )
        )
        concrete = [
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value in {"FULL_UP", "HALF_UP"}
        ]
        self.assertEqual({fact.mode for fact in concrete}, {"MIXED"})
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertNotIn("한 사이즈 업", summary)
        self.assertNotIn("반 사이즈 업", summary)

    def test_opposite_width_signals_are_mixed(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["발볼이 좁아서 불편해요.", "발볼이 여유롭고 편해요."],
                )
            )
        )
        width = [fact for fact in facts if fact.signal.subject == "WIDTH_SPACE"]
        self.assertEqual({fact.mode for fact in width}, {"MIXED"})

    def test_long_wear_is_not_inferred_without_duration_marker(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["바닥이 얇고 발바닥이 아파요."]))
        )
        self.assertFalse(any(fact.signal.category == "LONG_WEAR" for fact in facts))

    def test_soft_cushion_and_thin_sole_are_separate_not_conflicting(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["쿠션은 푹신하지만 바닥은 얇아요."]))
        )
        selected = {
            fact.signal.subject: fact.mode
            for fact in facts
            if fact.signal.subject in {"CUSHION_SOFTNESS", "SOLE_THICKNESS"}
        }
        self.assertEqual(selected, {"CUSHION_SOFTNESS": "SINGLE", "SOLE_THICKNESS": "SINGLE"})

    def test_runrepeat_and_review_same_concept_conflict_is_explicit(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["쿠션이 딱딱하고 푹신하지 않아요."])),
            _characteristics(("CUSHION", "HIGH")),
        )
        cushion = [fact for fact in facts if fact.signal.subject == "CUSHION_SOFTNESS"]
        self.assertTrue(cushion)
        self.assertTrue(all(fact.mode == "CONFLICT" for fact in cushion))


class PointSummaryRendererTests(unittest.TestCase):
    def test_size_down_consensus_is_preserved_exactly(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "사이즈 다운하니 잘 맞아요.",
                        "사이즈다운을 추천합니다.",
                    ],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("사이즈 다운", claims[0].text)
        self.assertNotIn("사이즈 업", claims[0].text)

    def test_repeated_metric_size_step_is_preserved_exactly(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "10mm 사이즈 업하니 잘 맞아요.",
                        "10미리 업하면 잘 맞습니다.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("10mm 사이즈 업", summary)
        self.assertNotIn("반 사이즈 업", summary)
        self.assertNotIn("한 사이즈 업", summary)
        exact_fact = next(
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value == "GENERIC_UP"
            and fact.signal.stance == "POSITIVE"
        )
        generic_claim = "사이즈 선택에서는 사이즈 업을 고려할 만합니다"
        generated = ShoeFitSummary(
            point_summary=f"{generic_claim}.",
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=generic_claim, evidence_ids=[exact_fact.evidence_id]
                )
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _point_claims_are_traceable(generated, shoe_name="테스트", facts=facts)
        )

    def test_different_positive_and_failed_size_consensus_are_both_preserved(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "반업하니 잘 맞아요.",
                        "반 사이즈 업하면 잘 맞아요.",
                    ],
                ),
                (
                    "INSOLE",
                    [201, 202],
                    [
                        "한 사이즈 업해도 여전히 작아요.",
                        "1업했지만 그래도 작게 느껴집니다.",
                    ],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        summary = join_point_summary_claims(claims)
        self.assertIn("반 사이즈 업", summary)
        self.assertIn("한 사이즈 업에서도 여유가 부족", summary)
        half_up = next(
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value == "HALF_UP"
            and fact.signal.stance == "POSITIVE"
        )
        full_up_failed = next(
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value == "FULL_UP"
            and fact.signal.stance == "NEGATIVE"
        )
        rendered_ids = {
            evidence_id for claim in claims for evidence_id in claim.evidence_ids
        }
        self.assertIn(half_up.evidence_id, rendered_ids)
        self.assertIn(full_up_failed.evidence_id, rendered_ids)

    def test_all_runrepeat_characteristics_use_exact_metric_level_copy(self) -> None:
        mappings = {
            "WIDTH_SPACE": {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"},
            "TOEBOX_SPACE": {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"},
            "CUSHION": {"LOW": "FIRM", "MEDIUM": "BALANCED", "HIGH": "SOFT"},
            "HEEL_HOLD": {"LOW": "FLEXIBLE", "MEDIUM": "BALANCED", "HIGH": "FIRM"},
            "BREATHABILITY": {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"},
            "SHOCK_ABSORPTION": {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"},
            "ENERGY_RETURN": {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"},
        }
        for characteristic, values in mappings.items():
            for level, expected_value in values.items():
                with self.subTest(characteristic=characteristic, level=level):
                    facts = prepare_point_summary_evidence(
                        _reasons(), _characteristics((characteristic, level))
                    )
                    fact = next(
                        fact for fact in facts if fact.source == "RUNREPEAT"
                    )
                    self.assertEqual(fact.signal.value, expected_value)
                    canonical = render_exact_lab_claim(fact)
                    self.assertIsNotNone(canonical)
                    claims = render_product_point_summary("테스트", facts)
                    self.assertEqual(claims[0].text, canonical)

    def test_killshot_review_wording_produces_immediate_multi_point_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102, 103],
                    [
                        (
                            "저는 발볼러고 지인은 칼발이라 정사이즈가 맞다고 합니다! "
                            "저는 업해서 샀거든요! 발볼러는 업해서 사시고 "
                            "칼발은 정사이즈로 가세요!"
                        ),
                        "색상이 깔끔하고 좋아요! 발볼 있으면 반업 하세욤",
                        (
                            "너무 예뻐요. 발볼 크면 10정도 업해야 할 것 같아요. "
                            "반업하니까 살짝 작아요."
                        ),
                    ],
                ),
                (
                    "INSOLE",
                    [104],
                    [
                        (
                            "착화감도 생각보다 편하고, 처음엔 살짝 딱딱한 느낌이 "
                            "있지만 금방 길들여져요. 사이즈는 정사이즈 추천!"
                        )
                    ],
                ),
            )
        )

        generic_up = next(
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value == "GENERIC_UP"
            and fact.signal.stance == "POSITIVE"
        )
        self.assertEqual(generic_up.mode, "CONSENSUS")
        self.assertEqual(generic_up.review_ids, (101, 103))
        half_up = [
            fact
            for fact in facts
            if fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value == "HALF_UP"
        ]
        self.assertEqual({fact.mode for fact in half_up}, {"MIXED"})
        self.assertTrue(
            any(fact.signal.subject == "WEAR_COMFORT" for fact in facts)
        )

        summary = join_point_summary_claims(
            render_product_point_summary("나이키 킬샷 2 M", facts)
        )
        self.assertIn("발볼이 넓은 편이라면 사이즈 업을 고려", summary)
        self.assertNotIn("전반적인 착화감", summary)
        self.assertIn("디자인에 대한 만족도가 높은 편", summary)
        self.assertNotIn("반 사이즈 업을 고려", summary)
        self.assertGreaterEqual(summary.count("."), 2)

    def test_bare_up_outside_size_context_is_not_a_size_signal(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "디자인을 스타일 업하기 좋아요.",
                        "업무용 디자인으로 만족합니다.",
                    ],
                )
            )
        )
        self.assertFalse(
            any(fact.signal.subject == "SIZE_OPTION" for fact in facts)
        )

    def test_design_only_llm_copy_cannot_replace_richer_fit_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼러는 업해서 신는 게 좋고 디자인도 좋아요.",
                        "발볼 있으면 사이즈 업해야 편하고 디자인이 좋아요.",
                    ],
                )
            )
        )
        thin = ShoeFitSummary(
            point_summary="디자인에 대한 만족도는 높은 편입니다.",
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text="디자인에 대한 만족도는 높은 편입니다",
                    evidence_ids=[
                        "MUSINSA:DESIGN:APPEARANCE:POSITIVE:POSITIVE"
                    ],
                )
            ],
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )

        self.assertFalse(
            _generated_point_summary_keeps_important_evidence(thin, facts)
        )

    def test_product_copy_uses_consensus_and_excludes_single_evidence(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌어요.",
                        "발볼러는 힘들어서 사이즈를 여유 있게 골라야 합니다.",
                    ],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                ),
                (
                    "INSOLE",
                    [301, 302],
                    [
                        "바닥이 얇아서 오래 서 있으면 발바닥이 아파요.",
                        "쿠션 때문인지 신발에 무게감이 있습니다.",
                    ],
                ),
            ),
            _characteristics(("CUSHION", "HIGH")),
        )
        summary = join_point_summary_claims(
            render_product_point_summary("반스 하프캡 - 블랙", facts)
        )
        self.assertIn("뒤꿈치를 비교적 안정적으로 잡아주는 편", summary)
        self.assertIn("쿠션감은 높은 편", summary)
        self.assertNotIn("한 사이즈 업", summary)
        self.assertNotIn("발바닥 부담", summary)
        self.assertNotIn("무게감", summary)
        for report_style in ("후기를 참고", "확인하는 것이 좋", "한편", "부위별 결과"):
            self.assertNotIn(report_style, summary)

    def test_single_styling_fact_is_not_generalized(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101], ["통큰 바지에 잘 어울려요."]))
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )
        self.assertNotIn("통큰 바지", summary)
        self.assertIn("일관된 근거가 아직 부족", summary)

    def test_explicit_wearing_comfort_without_word_chakwhagam_reaches_summary(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "실제로 신어보니 발이 편하네요.",
                        "신발이 편해서 출퇴근할 때 자주 신어요.",
                    ],
                )
            )
        )

        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )

        self.assertIn("전반적인 착화감은 편하게", summary)

    def test_short_standalone_comfort_predicate_can_join_independent_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "이뻐요 편해요 정사이즈 가세요.",
                        "발이 편해서 자주 신어요.",
                    ],
                )
            )
        )

        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )

        self.assertIn("전반적인 착화감은 편하게", summary)

    def test_spaced_all_day_phrase_is_long_wear_evidence(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "발이 편해서 하루 종일 신어도 피곤하지 않아요.",
                        "하루 종일 신고 있어도 발이 편안합니다.",
                    ],
                )
            )
        )

        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )

        self.assertIn("장시간 착용에서도 비교적 편안", summary)

    def test_breathability_review_consensus_reaches_summary(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "INSOLE",
                    [101, 102],
                    [
                        "통기성이 뛰어나서 출퇴근용으로 좋아요.",
                        "신발 통풍이 잘되어 쾌적합니다.",
                    ],
                )
            )
        )

        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )

        self.assertIn("통풍과 통기성은 좋게", summary)

    def test_single_generic_comfort_review_is_still_not_generalized(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101], ["신어보니 발이 편하네요."]))
        )

        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )

        self.assertNotIn("전반적인 착화감은 편하게", summary)
        self.assertIn("일관된 근거가 아직 부족", summary)

    def test_one_grounded_claim_does_not_append_low_information_boilerplate(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["디자인이 예뻐요.", "디자인이 만족스럽습니다."],
                )
            )
        )

        summary = compose_product_point_summary(
            render_product_point_summary("테스트 신발", facts, max_claims=None)
        )

        self.assertIn("디자인에 대한 만족도가 높은 편", summary)
        self.assertNotIn("그 밖의 착용 특성", summary)
        self.assertNotIn("단정하기 어렵", summary)

    def test_sparse_summary_can_add_clearly_attributed_single_review_details(self) -> None:
        facts = prepare_point_evidence_facts(
            [
                (101, "이뻐요 편해요 정사이즈 가세요 반업은 크네요."),
                (
                    102,
                    "소재가 가볍고 여름에 시원하게 신을 수 있을 것 같아요. "
                    "발볼도 넓게 나온 것 같고 만족합니다.",
                ),
                (103, "발이 편해서 하루 종일 신어도 피곤하지 않네요."),
            ]
        )

        summary = compose_product_point_summary(
            render_product_point_summary(
                "ACG 마운틴 플라이 2 로우 M",
                facts,
                max_claims=None,
                include_attributed_single_evidence=True,
            )
        )

        self.assertIn("전반적인 착화감은 편하게", summary)
        self.assertIn("하루 종일 착용해도 피로가 적었다는 의견", summary)
        self.assertIn("발볼이 넓게 느껴졌다는 의견", summary)
        self.assertIn("무게가 가볍게 느껴졌다는 의견", summary)
        self.assertIn("반복 근거가 아니므로", summary)

    def test_negative_styling_fact_is_never_rendered_as_positive(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101], ["통큰 바지에는 안 어울려요."]))
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )
        self.assertNotIn("자연스럽게 어울", summary)

    def test_full_up_is_never_rendered_as_half_up(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101],
                    ["발볼과 발등이 있는 편이라 1업했고 정사이즈는 발가락이 까졌어요."],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertNotIn("한 사이즈 업", summary)
        self.assertNotIn("반 사이즈 업", summary)

    def test_black_color_copy_requires_review_evidence_and_cites_it(self) -> None:
        facts_without_color = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["디자인이 예뻐요.", "외관이 마음에 듭니다."],
                )
            )
        )
        claims_without_color = render_product_point_summary(
            "테스트 신발 - 블랙", facts_without_color
        )
        self.assertNotIn("검정 색감", join_point_summary_claims(claims_without_color))

        facts_with_color = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102, 103],
                    [
                        "검정 디자인이 예뻐요.",
                        "디자인이 마음에 듭니다.",
                        "검은색 디자인이 예뻐요.",
                    ],
                )
            )
        )
        color_claim = next(
            claim
            for claim in render_product_point_summary(
                "테스트 신발 - 블랙", facts_with_color
            )
            if "검정 색감" in claim.text
        )
        color_evidence_id = next(
            fact.evidence_id
            for fact in facts_with_color
            if any("검은색" in text for text in fact.evidence_texts)
        )
        self.assertIn(color_evidence_id, color_claim.evidence_ids)

    def test_single_size_fact_does_not_invent_a_foot_condition(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("FOREFOOT", [101], ["한 사이즈 업했더니 편하게 맞아요."]))
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertNotIn("한 사이즈 업", summary)
        self.assertNotIn("발볼이나 발등", summary)

    def test_consensus_half_up_outweighs_single_full_up(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102, 103],
                    [
                        "반업했더니 편하게 잘 맞아요.",
                        "반 사이즈 업하니 편안합니다.",
                        "한 사이즈 업했더니 편하게 맞아요.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertIn("반 사이즈 업을 고려", summary)
        self.assertNotIn("한 사이즈 업을 고려", summary)

    def test_true_size_does_not_imply_a_snug_fit_without_exact_fit_evidence(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "정사이즈가 편하게 잘 맞아요.",
                        "정사이즈 선택이 편안합니다.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("정사이즈 선택이 잘 맞는 편", summary)
        self.assertNotIn("전체적으로 딱 맞는 핏", summary)

    def test_exact_fit_consensus_allows_true_size_exact_fit_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "정사이즈가 전체적으로 딱 맞아요.",
                        "정사이즈로 신으니 전체 핏이 딱 맞습니다.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("정사이즈는 전체적으로 딱 맞는 핏", summary)

    def test_failed_half_up_consensus_is_fit_first_not_a_recommendation(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "반업했지만 여전히 작아요. 디자인은 예뻐요.",
                        "반 사이즈 업해도 작게 느껴져요. 디자인이 좋아요.",
                    ],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("반 사이즈 업에서도 여유가 부족하게", claims[0].text)
        self.assertNotIn("고려할 만", claims[0].text)
        self.assertTrue(any("디자인" in claim.text for claim in claims[1:]))

    def test_width_copy_keeps_supported_failed_size_outcome(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼이 좁고 반업했지만 여전히 작아요.",
                        "발볼이 좁아서 불편하고 반 사이즈 업해도 작게 느껴져요.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("발볼은 비교적 슬림하게", summary)
        self.assertIn("반 사이즈 업에서도 여유가 부족하게", summary)

    def test_negative_true_size_consensus_is_not_silently_dropped(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["정사이즈는 좁고 끼어요.", "정사이즈가 작고 답답합니다."],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("정사이즈는 여유가 부족하게", summary)

    def test_three_fit_parts_are_all_rendered_and_traceable(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼이 좁고 반업해도 작으며 정사이즈도 좁아요.",
                        "발볼이 타이트하고 반 사이즈 업도 작고 정사이즈도 끼어요.",
                    ],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("발볼은 비교적 슬림하게", claims[0].text)
        self.assertIn("반 사이즈 업에서도 여유가 부족하게", claims[0].text)
        self.assertIn("정사이즈는 여유가 부족하게", claims[0].text)
        result = ShoeFitSummary(
            point_summary=join_point_summary_claims(claims),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in claims
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(result, shoe_name="테스트", facts=facts)
        )

    def test_single_color_mention_cannot_borrow_design_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["디자인이 예뻐요.", "블랙 색감과 디자인이 만족스러워요."],
                )
            )
        )
        appearance = next(
            fact
            for fact in facts
            if fact.signal.subject == "APPEARANCE"
            and fact.signal.value == "POSITIVE"
        )
        claim = "검정 색감과 디자인에 대한 만족도가 높은 편입니다"
        result = ShoeFitSummary(
            point_summary=f"{claim}.",
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim,
                    evidence_ids=[appearance.evidence_id],
                )
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _point_claims_are_traceable(result, shoe_name="테스트", facts=facts)
        )

    def test_toebox_consensus_stays_toebox_and_passes_grounding(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["앞코가 좁고 발가락이 끼어요.", "앞코 공간이 좁아서 불편해요."],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("앞코 공간은 비교적 좁게", claims[0].text)
        self.assertNotIn("발볼은", claims[0].text)
        result = ShoeFitSummary(
            point_summary=join_point_summary_claims(claims),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in claims
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(result, shoe_name="테스트", facts=facts)
        )

    def test_mixed_weight_is_not_rendered_as_heavy_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "INSOLE",
                    [101, 102],
                    ["신발에 무게감이 있어요.", "신어보니 가볍고 편합니다."],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertNotIn("가볍게", summary)
        self.assertNotIn("다소 무게감이 느껴지는 편", summary)
        self.assertIn("일관된 근거가 아직 부족", summary)

    def test_cushion_weight_wording_is_not_shoe_weight_evidence(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "INSOLE",
                    [101, 102],
                    [
                        "쿠션에서 무게감이 느껴져요.",
                        "깔창에서 다소 무게감이 느껴집니다.",
                    ],
                )
            )
        )
        self.assertFalse(any(fact.signal.subject == "WEIGHT" for fact in facts))

    def test_separate_thin_sole_and_long_wear_facts_do_not_invent_causality(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "INSOLE",
                    [101, 102],
                    [
                        "바닥이 얇아요. 장시간 신으면 발바닥이 피로해요.",
                        "바닥은 얇은 편입니다. 오래 신으면 발바닥이 아파요.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("바닥은 얇게 느껴지는 편", summary)
        self.assertIn("장시간 착용에서는 불편하게", summary)
        self.assertNotIn("바닥이 얇게 느껴지는 편이라 오래", summary)
        sole = next(fact for fact in facts if fact.signal.subject == "SOLE_THICKNESS")
        long_wear = next(
            fact for fact in facts if fact.signal.subject == "LONG_WEAR_COMFORT"
        )
        causal_text = "바닥이 얇아서 장시간 착용하면 발바닥 부담이 느껴질 수 있습니다"
        unsafe = ShoeFitSummary(
            point_summary=f"{causal_text}.",
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=causal_text,
                    evidence_ids=[sole.evidence_id, long_wear.evidence_id],
                )
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=facts)
        )

    def test_important_weight_claim_cannot_be_dropped_by_generated_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["발볼이 좁아 불편해요.", "발볼이 좁고 타이트해요."],
                ),
                (
                    "INSOLE",
                    [201, 202],
                    ["신발이 무거워요.", "착화 시 무게감이 있어요."],
                ),
                (
                    "HEEL",
                    [301, 302],
                    ["디자인이 예뻐요.", "디자인이 만족스러워요."],
                ),
            )
        )
        fallback = render_product_point_summary("테스트", facts)
        kept = [claim for claim in fallback if "무게감" not in claim.text]
        generated = ShoeFitSummary(
            point_summary=join_point_summary_claims(kept),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in kept
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _generated_point_summary_keeps_important_evidence(generated, facts)
        )

    def test_all_fit_claims_stay_before_design_claims(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["반업하니 잘 맞아요.", "반 사이즈 업이 편해요."],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["디자인이 예뻐요.", "디자인 만족도가 높아요."],
                ),
            ),
            _characteristics(("WIDTH_SPACE", "LOW"), ("TOEBOX_SPACE", "HIGH")),
        )
        fallback = render_product_point_summary("테스트", facts)
        self.assertGreaterEqual(len(fallback), 4)
        reordered = [fallback[0], fallback[3], fallback[1], fallback[2]]
        generated = ShoeFitSummary(
            point_summary=join_point_summary_claims(reordered),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in reordered
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _generated_point_summary_keeps_important_evidence(generated, facts)
        )

        combined_text = "디자인에 대한 만족도가 높으며, 발볼 공간은 낮은 편입니다"
        combined = ShoePointSummaryClaim(
            text=combined_text,
            evidence_ids=[
                *fallback[3].evidence_ids,
                *fallback[0].evidence_ids,
            ],
        )
        inside_reordered = [combined, fallback[1], fallback[2]]
        generated_inside = ShoeFitSummary(
            point_summary=join_point_summary_claims(inside_reordered),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in inside_reordered
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertFalse(
            _generated_point_summary_keeps_important_evidence(
                generated_inside, facts
            )
        )

    def test_non_fit_clause_cannot_prefix_the_first_fit_claim(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "외관이 예쁘고 발볼은 좁아요.",
                        "외관이 예뻐서 좋고 발볼이 타이트해요.",
                    ],
                ),
                (
                    "HEEL",
                    [201, 202],
                    [
                        "발목을 안정적으로 잡아줘요.",
                        "뒤꿈치를 안정적으로 받쳐줍니다.",
                    ],
                ),
            )
        )
        width = next(
            fact
            for fact in facts
            if fact.signal.subject == "WIDTH_SPACE" and fact.signal.value == "TIGHT"
        )
        appearance = next(
            fact
            for fact in facts
            if fact.signal.subject == "APPEARANCE"
            and fact.signal.value == "POSITIVE"
        )
        heel = next(
            fact
            for fact in facts
            if fact.signal.subject == "HEEL_HOLD" and fact.signal.value == "STABLE"
        )
        first = "외관이 예쁘고 발볼은 좁게 맞는 편입니다"
        second = "발목은 안정적으로 잡아주는 편입니다"
        generated = ShoeFitSummary(
            point_summary=f"{first}. {second}.",
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=first,
                    evidence_ids=[appearance.evidence_id, width.evidence_id],
                ),
                ShoePointSummaryClaim(text=second, evidence_ids=[heel.evidence_id]),
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(generated, shoe_name="테스트", facts=facts)
        )
        self.assertFalse(
            _generated_point_summary_keeps_important_evidence(generated, facts)
        )

    def test_two_specific_outfit_consensus_supports_broad_styling_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102, 103],
                    [
                        "청바지에 잘 어울려요.",
                        "데님과 슬랙스 코디에 잘 어울립니다.",
                        "슬랙스 코디가 잘 어울립니다.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("청바지와 슬랙스", summary)
        self.assertIn("다양한 스타일", summary)

    def test_general_styling_consensus_is_preserved(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "어디에 신어도 깔끔하게 잘 어울려요.",
                        "여기저기 코디하기 좋아요.",
                    ],
                )
            )
        )
        general = next(
            fact
            for fact in facts
            if fact.signal.subject == "OUTFIT" and fact.signal.value == "GENERAL"
        )
        self.assertEqual(point_evidence_strength(general, facts), "MEDIUM")
        self.assertIn(
            "다양한 스타일과 잘 어울리는",
            join_point_summary_claims(render_product_point_summary("테스트", facts)),
        )

    def test_weight_and_styling_fallback_copy_self_grounds(self) -> None:
        scenarios = (
            (["신발이 무거워요.", "착화하면 무거워서 부담돼요."], "무게감"),
            (["신발이 가벼워요.", "착화하면 가벼워서 편해요."], "가볍게"),
            (["청바지에 잘 어울려요.", "데님에 잘 어울립니다."], "청바지와 잘 어울"),
            (["어디든 잘 어울려요.", "어디에나 잘 어울립니다."], "다양한 스타일과 잘 어울"),
            (["데일리 코디에 잘 어울려요.", "데일리로 매치하기 좋아요."], "데일리로 활용"),
        )
        for reviews, expected in scenarios:
            with self.subTest(expected=expected):
                facts = prepare_point_summary_evidence(
                    _reasons(("HEEL", [101, 102], reviews))
                )
                claims = render_product_point_summary("테스트", facts)
                self.assertIn(expected, join_point_summary_claims(claims))
                result = ShoeFitSummary(
                    point_summary=join_point_summary_claims(claims),
                    point_summary_claims=[
                        ShoePointSummaryClaim(
                            text=claim.text,
                            evidence_ids=list(claim.evidence_ids),
                        )
                        for claim in claims
                    ],
                    forefoot_summary="발볼",
                    heel_summary="뒤꿈치",
                    insole_summary="깔창",
                    forefoot_review_ids=[],
                    heel_review_ids=[],
                    insole_review_ids=[],
                )
                self.assertTrue(
                    _point_claims_are_traceable(
                        result, shoe_name="테스트", facts=facts
                    )
                )

    def test_neutral_or_conflicting_design_does_not_become_satisfaction(self) -> None:
        neutral_facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["디자인은 무난합니다.", "디자인을 살펴봤습니다."],
                )
            )
        )
        mentioned = next(
            fact
            for fact in neutral_facts
            if fact.signal.subject == "APPEARANCE"
            and fact.signal.value == "MENTIONED"
        )
        self.assertEqual(point_evidence_strength(mentioned, neutral_facts), "LOW")
        self.assertNotIn(
            "만족도가 높은",
            join_point_summary_claims(
                render_product_point_summary("테스트", neutral_facts)
            ),
        )

        conflicting = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [201, 202, 203],
                    [
                        "디자인이 예뻐요.",
                        "디자인이 만족스러워요.",
                        "디자인은 별로예요.",
                    ],
                )
            )
        )
        appearance = [
            fact for fact in conflicting if fact.signal.subject == "APPEARANCE"
        ]
        self.assertEqual({fact.mode for fact in appearance}, {"MIXED"})

    def test_runrepeat_width_is_the_first_fit_claim_without_review_fit(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101, 102], ["디자인이 예뻐요.", "디자인이 좋아요."])),
            _characteristics(("WIDTH_SPACE", "LOW"), ("TOEBOX_SPACE", "MEDIUM")),
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("발볼 공간은 낮은 편", claims[0].text)
        self.assertIn("앞코 공간은 보통 수준", claims[1].text)

        summary = ShoeFitSummary(
            point_summary=join_point_summary_claims(claims),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text,
                    evidence_ids=list(claim.evidence_ids),
                )
                for claim in claims
            ],
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(summary, shoe_name="테스트", facts=facts)
        )

    def test_lab_width_precedes_review_size_when_review_width_is_missing(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["반업했더니 잘 맞아요.", "반 사이즈 업이 편해요."],
                )
            ),
            _characteristics(("WIDTH_SPACE", "LOW"), ("TOEBOX_SPACE", "HIGH")),
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("발볼 공간은 낮은 편", claims[0].text)
        self.assertIn("앞코 공간은 높은 편", claims[1].text)
        self.assertIn("반 사이즈 업", claims[2].text)

    def test_negative_heel_and_wear_consensus_are_product_copy(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    [
                        "뒤꿈치가 헐렁하고 마찰이 느껴져요.",
                        "뒤축이 들뜨고 뒤꿈치가 쓸려요.",
                    ],
                ),
                (
                    "INSOLE",
                    [201, 202],
                    ["착화감이 불편해요.", "착화감이 불편하게 느껴져요."],
                ),
            )
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트", facts)
        )
        self.assertIn("뒤꿈치는 다소 헐렁하게", summary)
        self.assertIn("마찰이 느껴질 수", summary)
        self.assertIn("전반적인 착화감은 다소 불편하게", summary)
        claims = render_product_point_summary("테스트", facts)
        result = ShoeFitSummary(
            point_summary=join_point_summary_claims(claims),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in claims
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(result, shoe_name="테스트", facts=facts)
        )

    def test_heel_comfort_fallback_self_grounds(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["뒤꿈치가 편해요.", "뒤축이 편합니다."],
                )
            )
        )
        claims = render_product_point_summary("테스트", facts)
        self.assertIn("뒤꿈치 주변은 비교적 편안하게", claims[0].text)
        result = ShoeFitSummary(
            point_summary=join_point_summary_claims(claims),
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text, evidence_ids=list(claim.evidence_ids)
                )
                for claim in claims
            ],
            forefoot_summary="발볼",
            heel_summary="뒤꿈치",
            insole_summary="깔창",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )
        self.assertTrue(
            _point_claims_are_traceable(result, shoe_name="테스트", facts=facts)
        )

    def test_long_wear_profile_and_overall_fit_fallbacks_self_ground(self) -> None:
        scenarios = (
            (
                [
                    "장시간 걸어도 발바닥이 편합니다.",
                    "오래 걸어도 발바닥은 편해요.",
                ],
                "장시간 착용에서도 비교적 편안하게",
            ),
            (
                [
                    "굽은 낮은 로우 프로파일입니다.",
                    "로우 프로파일이 특징이에요.",
                ],
                "로우 프로파일이 특징",
            ),
            (
                ["전체적으로 딱 맞아요.", "전체 핏이 딱 맞습니다."],
                "전체적으로 딱 맞는 핏",
            ),
        )
        for reviews, expected in scenarios:
            with self.subTest(expected=expected):
                facts = prepare_point_summary_evidence(
                    _reasons(("INSOLE", [101, 102], reviews))
                )
                claims = render_product_point_summary("테스트", facts)
                self.assertIn(expected, join_point_summary_claims(claims))
                result = ShoeFitSummary(
                    point_summary=join_point_summary_claims(claims),
                    point_summary_claims=[
                        ShoePointSummaryClaim(
                            text=claim.text,
                            evidence_ids=list(claim.evidence_ids),
                        )
                        for claim in claims
                    ],
                    forefoot_summary="발볼",
                    heel_summary="뒤꿈치",
                    insole_summary="깔창",
                    forefoot_review_ids=[],
                    heel_review_ids=[],
                    insole_review_ids=[],
                )
                self.assertTrue(
                    _point_claims_are_traceable(
                        result, shoe_name="테스트", facts=facts
                    )
                )
                self.assertTrue(
                    _generated_point_summary_keeps_important_evidence(
                        result, facts
                    )
                )

    def test_deterministic_product_claims_pass_the_same_grounding_gate(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌어요.",
                        "발볼러는 힘들어요.",
                    ],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                ),
                (
                    "INSOLE",
                    [301],
                    ["바닥이 얇아서 오래 서 있으면 발바닥이 아파요."],
                ),
            ),
            _characteristics(("CUSHION", "HIGH")),
        )
        claims = render_product_point_summary("테스트 신발", facts)
        point_summary = join_point_summary_claims(claims)
        result = ShoeFitSummary(
            point_summary=point_summary,
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text,
                    evidence_ids=list(claim.evidence_ids),
                )
                for claim in claims
            ],
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )

        self.assertTrue(
            _point_claims_are_traceable(
                result,
                shoe_name="테스트 신발",
                facts=facts,
            )
        )


if __name__ == "__main__":
    unittest.main()
