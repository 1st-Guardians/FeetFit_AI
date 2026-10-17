from __future__ import annotations

import unittest

from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.services.shoe.shoe_feature_rules import PointSignal
from app.services.shoe.shoe_fit_comment_service import (
    _contains_mechanical_area_result_summary,
    _point_claims_are_traceable,
)
from app.services.shoe.shoe_point_summary import (
    PointEvidenceFact,
    render_exact_lab_claim,
)


def _fact(
    evidence_id: str,
    *,
    category: str,
    subject: str,
    value: str,
    stance: str,
    source: str = "MUSINSA",
    mode: str = "SINGLE",
    review_ids: tuple[int, ...] = (101,),
    evidence_text: str,
) -> PointEvidenceFact:
    return PointEvidenceFact(
        evidence_id=evidence_id,
        signal=PointSignal(category, subject, value, stance),
        source=source,
        mode=mode,
        review_ids=review_ids if source == "MUSINSA" else (),
        evidence_texts=(evidence_text,),
    )


def _summary(
    *claims: tuple[str, tuple[str, ...]],
    point_summary: str | None = None,
) -> ShoeFitSummary:
    claim_models = [
        ShoePointSummaryClaim(text=text, evidence_ids=list(evidence_ids))
        for text, evidence_ids in claims
    ]
    if point_summary is None:
        point_summary = " ".join(
            f"{claim.text.rstrip('.')}." for claim in claim_models
        )
    return ShoeFitSummary(
        point_summary=point_summary,
        point_summary_claims=claim_models,
        forefoot_summary="발볼",
        heel_summary="뒤꿈치",
        insole_summary="깔창",
        forefoot_review_ids=[],
        heel_review_ids=[],
        insole_review_ids=[],
    )


class PointClaimGroundingAdversarialTests(unittest.TestCase):
    def test_assertion_without_any_evidence_is_rejected(self) -> None:
        summary = _summary(point_summary="이 신발은 내구성이 뛰어납니다.")

        self.assertFalse(
            _point_claims_are_traceable(summary, shoe_name="테스트", facts=[])
        )

    def test_single_fact_is_excluded_even_when_hedged(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            evidence_text="발볼이 있어 1업했습니다.",
        )
        unsafe = _summary(
            ("한 착화 사례에서는 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
            ("한 사이즈 업이 잘 맞습니다", (full_up.evidence_id,)),
        )
        safe = _summary(
            ("한 착화 사례에서는 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[full_up])
        )
        self.assertFalse(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[full_up])
        )

    def test_single_fact_cannot_use_universal_quantifiers(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            evidence_text="1업했습니다.",
        )
        summary = _summary(
            ("누구나 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
        )

        self.assertFalse(
            _point_claims_are_traceable(summary, shoe_name="테스트", facts=[full_up])
        )

    def test_mixed_fact_is_excluded_even_with_variation_scope(self) -> None:
        tight = _fact(
            "MUSINSA:WIDTH:TIGHT",
            category="WIDTH_FIT",
            subject="WIDTH_SPACE",
            value="TIGHT",
            stance="NEGATIVE",
            mode="MIXED",
            evidence_text="발볼이 좁게 느껴집니다.",
        )
        unsafe = _summary(("발볼은 좁은 편입니다", (tight.evidence_id,)))
        safe = _summary(
            ("발볼 착화감에는 개인차가 있어 좁게 느껴질 수 있습니다", (tight.evidence_id,))
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[tight])
        )
        self.assertFalse(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[tight])
        )

    def test_lab_fact_rejects_reversal_exaggeration_and_unrelated_property(self) -> None:
        cushion = _fact(
            "RUNREPEAT:CUSHION:HIGH",
            category="CUSHION_FEEL",
            subject="CUSHION_SOFTNESS",
            value="SOFT",
            stance="OBJECTIVE",
            source="RUNREPEAT",
            mode="LAB",
            review_ids=(),
            evidence_text="RunRepeat 비교 특성에서 쿠션감은 높은 편입니다.",
        )
        accurate = _summary((render_exact_lab_claim(cushion), (cushion.evidence_id,)))
        reversed_claim = _summary(
            ("RunRepeat 실측에서 쿠션감은 낮은 편입니다", (cushion.evidence_id,))
        )
        exaggerated = _summary(
            ("쿠션은 구름 위를 걷듯 최고 수준입니다", (cushion.evidence_id,))
        )
        unrelated = _summary(
            ("이 신발은 내구성이 뛰어납니다", (cushion.evidence_id,))
        )

        self.assertTrue(
            _point_claims_are_traceable(accurate, shoe_name="테스트", facts=[cushion])
        )
        negated = _summary(
            ("RunRepeat 실측에서 쿠션감은 높은 편이 아닙니다", (cushion.evidence_id,))
        )
        unsupported_effect = _summary(
            ("쿠션감은 높은 편이라 관절을 보호합니다", (cushion.evidence_id,))
        )
        for unsafe in (
            reversed_claim,
            exaggerated,
            unrelated,
            negated,
            unsupported_effect,
        ):
            self.assertFalse(
                _point_claims_are_traceable(
                    unsafe, shoe_name="테스트", facts=[cushion]
                )
            )

    def test_full_up_may_not_be_generalized_to_unspecified_size_up(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="1업했더니 편했습니다.",
        )
        unsafe = _summary(
            ("일부 착화에서는 사이즈 업을 고려할 만합니다", (full_up.evidence_id,))
        )
        safe = _summary(
            ("한 사이즈 업을 고려할 만합니다", (full_up.evidence_id,))
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[full_up])
        )
        self.assertTrue(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[full_up])
        )

    def test_numeric_size_relation_must_preserve_source_order_and_direction(self) -> None:
        generic_up = _fact(
            "MUSINSA:SIZE:GENERIC_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="GENERIC_UP",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="평소 250을 신다가 260으로 사이즈 업했습니다.",
        )
        correct = _summary(
            ("일부 착화에서는 250에서 260으로 사이즈 업했습니다", (generic_up.evidence_id,))
        )
        reversed_claim = _summary(
            ("일부 착화에서는 260에서 250으로 사이즈 다운했습니다", (generic_up.evidence_id,))
        )

        self.assertTrue(
            _point_claims_are_traceable(correct, shoe_name="테스트", facts=[generic_up])
        )
        self.assertFalse(
            _point_claims_are_traceable(
                reversed_claim, shoe_name="테스트", facts=[generic_up]
            )
        )

    def test_distributed_area_list_and_report_style_variants_are_rejected(self) -> None:
        distributed = (
            "발볼 적합도는 주의입니다. "
            "뒤꿈치 적합도는 좋습니다. "
            "깔창 적합도는 보통입니다."
        )
        parallel = "발볼은 좁습니다. 뒤꿈치는 안정적입니다. 깔창은 단단합니다."

        self.assertTrue(_contains_mechanical_area_result_summary(distributed))
        self.assertTrue(_contains_mechanical_area_result_summary(parallel))
        self.assertTrue(
            _contains_mechanical_area_result_summary(
                "구매평에 따르면 쿠션이 좋고 착화평도 우수합니다."
            )
        )
        for report_style in (
            "후기에서는 발볼이 좁다고 설명합니다.",
            "발볼이 좁다는 의견이 있습니다.",
            "반면 일부 사용자는 편하다고 말합니다.",
        ):
            self.assertTrue(_contains_mechanical_area_result_summary(report_style))

    def test_negative_size_option_cannot_be_rewritten_as_recommendation(self) -> None:
        failed_half_up = _fact(
            "MUSINSA:SIZE:HALF_UP:NEGATIVE",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="HALF_UP",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="반업했지만 여전히 작았습니다.",
        )
        recommendation = _summary(
            (
                "반 사이즈 업을 고려할 만합니다",
                (failed_half_up.evidence_id,),
            )
        )
        faithful = _summary(
            (
                "반 사이즈 업도 여전히 작게 느껴지는 편입니다",
                (failed_half_up.evidence_id,),
            )
        )

        self.assertFalse(
            _point_claims_are_traceable(
                recommendation, shoe_name="테스트", facts=[failed_half_up]
            )
        )
        self.assertTrue(
            _point_claims_are_traceable(
                faithful, shoe_name="테스트", facts=[failed_half_up]
            )
        )

    def test_long_wear_discomfort_cannot_be_reversed_to_comfort(self) -> None:
        discomfort = _fact(
            "MUSINSA:LONG_WEAR:DISCOMFORT",
            category="LONG_WEAR",
            subject="LONG_WEAR_COMFORT",
            value="DISCOMFORT",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="장시간 신으면 발바닥이 아파요.",
        )
        reversed_claim = _summary(
            (
                "장시간 착용에서도 편안하게 느껴지는 편입니다",
                (discomfort.evidence_id,),
            )
        )
        faithful = _summary(
            (
                "장시간 착용에서는 발바닥 부담이 느껴질 수 있습니다",
                (discomfort.evidence_id,),
            )
        )
        self.assertFalse(
            _point_claims_are_traceable(
                reversed_claim, shoe_name="테스트", facts=[discomfort]
            )
        )
        self.assertTrue(
            _point_claims_are_traceable(
                faithful, shoe_name="테스트", facts=[discomfort]
            )
        )

    def test_wear_discomfort_cannot_be_reversed_to_comfort(self) -> None:
        discomfort = _fact(
            "MUSINSA:OTHER:WEAR_DISCOMFORT",
            category="OTHER",
            subject="WEAR_COMFORT",
            value="DISCOMFORT",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="착화감이 불편해요.",
        )
        reversed_claim = _summary(
            (
                "전반적인 착화감은 편안하게 느껴지는 편입니다",
                (discomfort.evidence_id,),
            )
        )
        faithful = _summary(
            (
                "전반적인 착화감은 불편하게 느껴지는 편입니다",
                (discomfort.evidence_id,),
            )
        )
        self.assertFalse(
            _point_claims_are_traceable(
                reversed_claim, shoe_name="테스트", facts=[discomfort]
            )
        )
        self.assertTrue(
            _point_claims_are_traceable(
                faithful, shoe_name="테스트", facts=[discomfort]
            )
        )

    def test_positive_product_facts_reject_inflected_output_reversals(self) -> None:
        cases = (
            (
                _fact(
                    "MUSINSA:DESIGN:POSITIVE",
                    category="DESIGN",
                    subject="APPEARANCE",
                    value="POSITIVE",
                    stance="POSITIVE",
                    mode="CONSENSUS",
                    review_ids=(101, 102),
                    evidence_text="디자인 만족도가 높아요.",
                ),
                "디자인 만족도가 낮은 편입니다",
            ),
            (
                _fact(
                    "MUSINSA:HEEL:STABLE",
                    category="HEEL_FEEL",
                    subject="HEEL_HOLD",
                    value="STABLE",
                    stance="POSITIVE",
                    mode="CONSENSUS",
                    review_ids=(101, 102),
                    evidence_text="발목을 안정적으로 잡아줘요.",
                ),
                "발목을 안정적으로 잡지 못하는 편입니다",
            ),
            (
                _fact(
                    "MUSINSA:CUSHION:SOFT",
                    category="CUSHION_FEEL",
                    subject="CUSHION_SOFTNESS",
                    value="SOFT",
                    stance="POSITIVE",
                    mode="CONSENSUS",
                    review_ids=(101, 102),
                    evidence_text="쿠션이 부드러워요.",
                ),
                "쿠션은 부드럽지 못한 편입니다",
            ),
            (
                _fact(
                    "MUSINSA:OUTFIT:JEANS",
                    category="STYLING",
                    subject="OUTFIT",
                    value="JEANS",
                    stance="POSITIVE",
                    mode="CONSENSUS",
                    review_ids=(101, 102),
                    evidence_text="청바지에 잘 어울려요.",
                ),
                "청바지에 활용하기 어려운 편입니다",
            ),
            (
                _fact(
                    "MUSINSA:SIZE:TRUE",
                    category="SIZE_FIT",
                    subject="SIZE_OPTION",
                    value="TRUE_SIZE",
                    stance="POSITIVE",
                    mode="CONSENSUS",
                    review_ids=(101, 102),
                    evidence_text="정사이즈가 잘 맞아요.",
                ),
                "정사이즈를 추천하지 않습니다",
            ),
        )
        for fact, reversed_text in cases:
            with self.subTest(reversed_text=reversed_text):
                result = _summary((reversed_text, (fact.evidence_id,)))
                self.assertFalse(
                    _point_claims_are_traceable(
                        result, shoe_name="테스트", facts=[fact]
                    )
                )

    def test_cited_heel_problem_cannot_add_uncited_opposite_benefit(self) -> None:
        loose = _fact(
            "MUSINSA:HEEL:LOOSE",
            category="HEEL_FEEL",
            subject="HEEL_HOLD",
            value="LOOSE",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="뒤꿈치가 헐렁해요.",
        )
        rubbing = _fact(
            "MUSINSA:HEEL:RUBBING",
            category="HEEL_FEEL",
            subject="HEEL_COMFORT",
            value="RUBBING",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(201, 202),
            evidence_text="뒤꿈치가 쓸려요.",
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "뒤꿈치는 헐렁하지만 발을 잘 잡아주는 편입니다",
                        (loose.evidence_id,),
                    )
                ),
                shoe_name="테스트",
                facts=[loose],
            )
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "뒤꿈치는 쓸리지만 전반적으로 편안한 편입니다",
                        (rubbing.evidence_id,),
                    )
                ),
                shoe_name="테스트",
                facts=[rubbing],
            )
        )

    def test_soft_cushion_citation_cannot_add_uncited_firmness(self) -> None:
        soft = _fact(
            "MUSINSA:CUSHION:SOFT",
            category="CUSHION_FEEL",
            subject="CUSHION_SOFTNESS",
            value="SOFT",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="쿠션이 부드러워요.",
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "쿠션은 부드럽지만 단단한 편입니다",
                        (soft.evidence_id,),
                    )
                ),
                shoe_name="테스트",
                facts=[soft],
            )
        )

    def test_all_lab_cushion_levels_self_ground_in_typed_renderer(self) -> None:
        value_by_level = {"LOW": "FIRM", "MEDIUM": "BALANCED", "HIGH": "SOFT"}
        for level, value in value_by_level.items():
            with self.subTest(level=level):
                fact = _fact(
                    f"RUNREPEAT:CUSHION:{level}",
                    category="CUSHION_FEEL",
                    subject="CUSHION_SOFTNESS",
                    value=value,
                    stance="OBJECTIVE",
                    source="RUNREPEAT",
                    mode="LAB",
                    review_ids=(),
                    evidence_text=f"RunRepeat 비교 특성에서 쿠션감은 {level}입니다.",
                )
                self.assertTrue(
                    _point_claims_are_traceable(
                        _summary((render_exact_lab_claim(fact), (fact.evidence_id,))),
                        shoe_name="테스트",
                        facts=[fact],
                    )
                )

    def test_universal_sizing_language_is_rejected(self) -> None:
        half_up = _fact(
            "MUSINSA:SIZE:HALF_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="HALF_UP",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="반업하니 잘 맞아요.",
        )
        for text in (
            "많은 구매자가 반 사이즈 업을 추천합니다",
            "모든 구매자에게 반 사이즈 업이 잘 맞습니다",
            "구매자 모두에게 반 사이즈 업이 잘 맞습니다",
            "다수의 구매자에게 반 사이즈 업이 잘 맞습니다",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((text, (half_up.evidence_id,))),
                        shoe_name="테스트",
                        facts=[half_up],
                    )
                )

    def test_positive_size_fact_rejects_avoidance_advice(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="한 사이즈 업하니 잘 맞아요.",
        )
        for text in (
            "한 사이즈 업은 피해야 합니다",
            "한 사이즈 업은 권장 대상이 아닙니다",
            "한 사이즈 업을 선택하면 안 됩니다",
            "한 사이즈 업은 삼가는 편이 좋습니다",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((text, (full_up.evidence_id,))),
                        shoe_name="테스트",
                        facts=[full_up],
                    )
                )

    def test_numeric_size_advice_cannot_change_units(self) -> None:
        generic_up = _fact(
            "MUSINSA:SIZE:GENERIC_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="GENERIC_UP",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="10mm 사이즈 업하니 잘 맞아요.",
        )
        for text in (
            "10cm 사이즈 업을 고려할 만합니다",
            "10센티 사이즈 업을 고려할 만합니다",
            "10센티미터 사이즈 업을 고려할 만합니다",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((text, (generic_up.evidence_id,))),
                        shoe_name="테스트",
                        facts=[generic_up],
                    )
                )

    def test_one_color_mention_cannot_borrow_generic_design_consensus(self) -> None:
        appearance = PointEvidenceFact(
            evidence_id="MUSINSA:DESIGN:APPEARANCE:POSITIVE:POSITIVE",
            signal=PointSignal("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE"),
            source="MUSINSA",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_texts=("디자인이 예뻐요.", "노란색 색감이 예뻐요."),
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "노란색 색감과 디자인 만족도가 높은 편입니다",
                        (appearance.evidence_id,),
                    )
                ),
                shoe_name="테스트",
                facts=[appearance],
            )
        )

    def test_one_outfit_consensus_cannot_be_broadened_to_many_styles(self) -> None:
        jeans = _fact(
            "MUSINSA:STYLING:JEANS",
            category="STYLING",
            subject="OUTFIT",
            value="JEANS",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="청바지에 잘 어울려요.",
        )
        for text in (
            "청바지를 비롯해 여러 스타일에 잘 어울리는 편입니다",
            "청바지를 포함한 모든 스타일에 잘 어울립니다",
            "청바지부터 어떤 옷에도 잘 어울립니다",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((text, (jeans.evidence_id,))),
                        shoe_name="테스트",
                        facts=[jeans],
                    )
                )

    def test_separate_foot_conditions_cannot_be_rewritten_as_both(self) -> None:
        condition = PointEvidenceFact(
            evidence_id="MUSINSA:WIDTH:FOOT_CONDITION",
            signal=PointSignal(
                "WIDTH_FIT",
                "FOOT_CONDITION",
                "WIDE_OR_HIGH_INSTEP",
                "NEUTRAL",
            ),
            source="MUSINSA",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_texts=(
                "발볼이 넓은 편입니다.",
                "발등이 높은 편입니다.",
            ),
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "발볼과 발등이 모두 있는 편입니다",
                        (condition.evidence_id,),
                    )
                ),
                shoe_name="테스트",
                facts=[condition],
            )
        )

    def test_predicates_cannot_be_swapped_between_product_attributes(self) -> None:
        width = _fact(
            "MUSINSA:WIDTH:TIGHT",
            category="WIDTH_FIT",
            subject="WIDTH_SPACE",
            value="TIGHT",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="발볼이 좁아요.",
        )
        heel = _fact(
            "MUSINSA:HEEL:STABLE",
            category="HEEL_FEEL",
            subject="HEEL_HOLD",
            value="STABLE",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(201, 202),
            evidence_text="발목을 안정적으로 잡아줘요.",
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "발볼은 안정적으로 잡아주고 발목은 좁은 편입니다",
                        (width.evidence_id, heel.evidence_id),
                    )
                ),
                shoe_name="테스트",
                facts=[width, heel],
            )
        )
        self.assertTrue(
            _point_claims_are_traceable(
                _summary(
                    (
                        "발볼은 좁게 느껴지는 편이고 발목은 안정적으로 잡아주는 편입니다",
                        (width.evidence_id, heel.evidence_id),
                    )
                ),
                shoe_name="테스트",
                facts=[width, heel],
            )
        )

        cushion = _fact(
            "MUSINSA:CUSHION:SOFT",
            category="CUSHION_FEEL",
            subject="CUSHION_SOFTNESS",
            value="SOFT",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(301, 302),
            evidence_text="쿠션이 부드러워요.",
        )
        weight = _fact(
            "MUSINSA:WEIGHT:HEAVY",
            category="WEIGHT_FEEL",
            subject="WEIGHT",
            value="HEAVY",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(401, 402),
            evidence_text="신발 무게가 무거워요.",
        )
        self.assertFalse(
            _point_claims_are_traceable(
                _summary(
                    (
                        "쿠션은 무겁게 느껴지고 신발 무게는 부드럽게 느껴지는 편입니다",
                        (cushion.evidence_id, weight.evidence_id),
                    )
                ),
                shoe_name="테스트",
                facts=[cushion, weight],
            )
        )

    def test_heel_hold_and_comfort_predicates_cannot_be_swapped(self) -> None:
        hold = _fact(
            "MUSINSA:HEEL:HOLD",
            category="HEEL_FEEL",
            subject="HEEL_HOLD",
            value="STABLE",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="발목을 안정적으로 잡아줘요.",
        )
        rubbing = _fact(
            "MUSINSA:HEEL:RUBBING:LOCAL",
            category="HEEL_FEEL",
            subject="HEEL_COMFORT",
            value="RUBBING",
            stance="NEGATIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="뒤꿈치가 쓸려요.",
        )
        correct = _summary(
            (
                "발목은 안정적으로 잡아주고 뒤꿈치 주변은 마찰이 느껴지는 편입니다",
                (hold.evidence_id, rubbing.evidence_id),
            )
        )
        swapped = _summary(
            (
                "발목은 마찰이 느껴지고 뒤꿈치는 안정적으로 잡아주는 편입니다",
                (hold.evidence_id, rubbing.evidence_id),
            )
        )
        self.assertTrue(
            _point_claims_are_traceable(correct, shoe_name="테스트", facts=[hold, rubbing])
        )
        self.assertFalse(
            _point_claims_are_traceable(swapped, shoe_name="테스트", facts=[hold, rubbing])
        )

    def test_runrepeat_claim_cannot_append_an_inferred_benefit(self) -> None:
        examples = (
            ("BREATHABILITY", "HIGH", "통기성은 높은 편입니다", "통기성은 높은 편이라 더운 날에도 쾌적합니다"),
            ("SHOCK_ABSORPTION", "HIGH", "충격 완화 수준은 높은 편입니다", "충격 완화 수준은 높은 편이라 무릎 부담을 덜어줍니다"),
            ("ENERGY_RETURN", "HIGH", "반발력은 높은 편입니다", "반발력은 높은 편이라 추진력을 높여줍니다"),
        )
        for subject, value, evidence_text, unsafe_text in examples:
            with self.subTest(subject=subject):
                fact = _fact(
                    f"RUNREPEAT:{subject}:HIGH",
                    category="OTHER" if subject != "SHOCK_ABSORPTION" else "CUSHION_FEEL",
                    subject=subject,
                    value=value,
                    stance="OBJECTIVE",
                    source="RUNREPEAT",
                    mode="LAB",
                    review_ids=(),
                    evidence_text=evidence_text,
                )
                self.assertTrue(
                    _point_claims_are_traceable(
                        _summary((render_exact_lab_claim(fact), (fact.evidence_id,))),
                        shoe_name="테스트",
                        facts=[fact],
                    )
                )
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((unsafe_text, (fact.evidence_id,))),
                        shoe_name="테스트",
                        facts=[fact],
                    )
                )

    def test_typed_fact_cannot_hide_an_uncited_appended_clause(self) -> None:
        comfortable = _fact(
            "MUSINSA:HEEL:COMFORTABLE",
            category="HEEL_FEEL",
            subject="HEEL_COMFORT",
            value="COMFORTABLE",
            stance="POSITIVE",
            mode="CONSENSUS",
            review_ids=(101, 102),
            evidence_text="뒤꿈치 주변이 편해요.",
        )
        for text in (
            "뒤꿈치 주변은 편하면서 끈이 자주 풀립니다",
            "뒤꿈치 주변은 편하나 봉제가 약합니다",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    _point_claims_are_traceable(
                        _summary((text, (comfortable.evidence_id,))),
                        shoe_name="테스트",
                        facts=[comfortable],
                    )
                )


if __name__ == "__main__":
    unittest.main()
