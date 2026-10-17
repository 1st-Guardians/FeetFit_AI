from __future__ import annotations

import unittest

from app.services.shoe.shoe_feature_rules import classify_point_evidence


def _signals(text: str) -> set[tuple[str, str, str, str]]:
    return {
        (signal.category, signal.subject, signal.value, signal.stance)
        for signal in classify_point_evidence(text)
    }


class PointEvidenceAdversarialClassificationTests(unittest.TestCase):
    def test_full_up_with_explicit_discomfort_is_not_positive(self) -> None:
        signals = _signals("1업했지만 작고 불편해요.")

        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "FULL_UP", "POSITIVE"), signals)
        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "FULL_UP", "NEGATIVE"), signals)

    def test_negated_heaviness_is_not_heavy(self) -> None:
        signals = _signals("생각보다 무겁지 않아요.")

        self.assertFalse(
            any(subject == "WEIGHT" and value == "HEAVY" for _, subject, value, _ in signals)
        )

    def test_common_weight_inflections_are_classified(self) -> None:
        self.assertIn(
            ("WEIGHT_FEEL", "WEIGHT", "HEAVY", "NEGATIVE"),
            _signals("신발이 무거워요."),
        )
        self.assertIn(
            ("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE"),
            _signals("가벼운 신발입니다."),
        )

    def test_cushion_weight_wording_is_not_physical_weight(self) -> None:
        signals = _signals("쿠션에서 무게감이 느껴집니다.")
        self.assertFalse(
            any(subject == "WEIGHT" for _, subject, _, _ in signals)
        )

    def test_abstract_design_weight_is_not_physical_weight(self) -> None:
        for text in (
            "디자인에 무게감이 있어요.",
            "색감에 무게감이 있습니다.",
            "실루엣에서 무게감이 느껴져요.",
            "디자인이 가벼운 느낌이에요.",
            "색감이 가볍게 느껴집니다.",
            "실루엣이 가벼워 보여요.",
            "쿠션이 가볍게 느껴집니다.",
            "무거운 디자인이라 별로예요.",
            "가벼운 색감이 마음에 들어요.",
            "가벼운 실루엣이 예뻐요.",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    any(subject == "WEIGHT" for _, subject, _, _ in _signals(text))
                )

    def test_negated_long_wear_pain_is_comfortable_not_discomfort(self) -> None:
        signals = _signals("오래 신어도 안 아프고 편해요.")

        self.assertNotIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"),
            signals,
        )
        self.assertIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            signals,
        )

    def test_negated_outfit_pairing_is_not_positive(self) -> None:
        signals = _signals("통큰 바지에 안 어울려요.")

        self.assertNotIn(("STYLING", "OUTFIT", "WIDE_PANTS", "POSITIVE"), signals)
        self.assertIn(("STYLING", "OUTFIT", "WIDE_PANTS", "NEGATIVE"), signals)

    def test_wide_wearer_is_not_roomy_shoe(self) -> None:
        signals = _signals("발볼이 넓은 사람이라 한 사이즈 업했어요.")

        self.assertNotIn(("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"), signals)
        self.assertIn(
            ("WIDTH_FIT", "FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP", "NEUTRAL"),
            signals,
        )

    def test_bare_wide_fit_phrase_describes_the_shoe_not_the_wearer(self) -> None:
        signals = _signals("발볼이 넓은 편이고 반 사이즈 업을 추천해요.")
        self.assertIn(
            ("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"), signals
        )
        self.assertNotIn(
            ("WIDTH_FIT", "FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP", "NEUTRAL"),
            signals,
        )

    def test_conditional_wide_foot_phrase_describes_the_wearer(self) -> None:
        for text in (
            "발볼이 넓으면 반업을 추천해요.",
            "발볼 넓다면 반 사이즈 업하세요.",
        ):
            with self.subTest(text=text):
                signals = _signals(text)
                self.assertIn(
                    (
                        "WIDTH_FIT",
                        "FOOT_CONDITION",
                        "WIDE_OR_HIGH_INSTEP",
                        "NEUTRAL",
                    ),
                    signals,
                )
                self.assertNotIn(
                    ("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"), signals
                )

    def test_size_options_keep_their_own_polarity_in_one_sentence(self) -> None:
        signals = _signals("반업은 편하고 정사이즈는 좁아요.")

        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "HALF_UP", "POSITIVE"), signals)
        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "TRUE_SIZE", "NEGATIVE"), signals)
        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "HALF_UP", "NEGATIVE"), signals)
        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "TRUE_SIZE", "POSITIVE"), signals)

    def test_negated_product_properties_do_not_become_positive_facts(self) -> None:
        cases = (
            (
                "발볼이 좁지 않아요.",
                ("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE"),
            ),
            (
                "발목을 안정적으로 잡아주지 않아요.",
                ("HEEL_FEEL", "HEEL_HOLD", "STABLE", "POSITIVE"),
            ),
            (
                "쿠션이 부드럽지 않아요.",
                ("CUSHION_FEEL", "CUSHION_SOFTNESS", "SOFT", "POSITIVE"),
            ),
            (
                "바닥이 얇지 않아요.",
                ("CUSHION_FEEL", "SOLE_THICKNESS", "THIN", "NEGATIVE"),
            ),
        )
        for text, forbidden in cases:
            with self.subTest(text=text):
                self.assertNotIn(forbidden, _signals(text))

    def test_uncomfortable_wear_is_not_misread_from_pyeonhae_substring(self) -> None:
        signals = _signals("착화감이 불편해요.")
        self.assertIn(
            ("OTHER", "WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"), signals
        )
        self.assertNotIn(
            ("OTHER", "WEAR_COMFORT", "COMFORTABLE", "POSITIVE"), signals
        )

    def test_other_positive_attribute_does_not_bleed_into_design_or_styling(self) -> None:
        design = _signals("디자인은 별로지만 착화감은 편해요.")
        self.assertNotIn(
            ("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE"), design
        )
        self.assertIn(
            ("DESIGN", "APPEARANCE", "NEGATIVE", "NEGATIVE"), design
        )

        styling = _signals("청바지를 입고 신었는데 착화감이 좋아요.")
        self.assertNotIn(
            ("STYLING", "OUTFIT", "JEANS", "POSITIVE"), styling
        )

    def test_toebox_evidence_is_not_relabelled_as_width(self) -> None:
        signals = _signals("앞코가 좁고 발가락이 끼어요.")
        self.assertIn(
            ("WIDTH_FIT", "TOEBOX_SPACE", "TIGHT", "NEGATIVE"), signals
        )
        self.assertNotIn(
            ("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE"), signals
        )

    def test_attribute_sentiments_stay_in_their_local_subject_span(self) -> None:
        width_and_wear = _signals("발볼은 좁지만 착화감은 좋아요.")
        self.assertIn(
            ("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE"),
            width_and_wear,
        )
        self.assertIn(
            ("OTHER", "WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            width_and_wear,
        )
        self.assertNotIn(
            ("OTHER", "WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"),
            width_and_wear,
        )

        width_and_toebox = _signals("발볼은 넓고 앞코는 좁아요.")
        self.assertIn(
            ("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"),
            width_and_toebox,
        )
        self.assertIn(
            ("WIDTH_FIT", "TOEBOX_SPACE", "TIGHT", "NEGATIVE"),
            width_and_toebox,
        )
        self.assertNotIn(
            ("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE"),
            width_and_toebox,
        )
        self.assertNotIn(
            ("WIDTH_FIT", "TOEBOX_SPACE", "ROOMY", "POSITIVE"),
            width_and_toebox,
        )

        heel = _signals("발볼은 좁지만 발목은 편해요.")
        self.assertIn(
            ("HEEL_FEEL", "HEEL_COMFORT", "COMFORTABLE", "POSITIVE"), heel
        )

    def test_inflected_negations_are_not_positive_evidence(self) -> None:
        cases = (
            (
                "디자인이 좋지 않아요.",
                ("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE"),
            ),
            (
                "청바지에 잘 어울리지 않아요.",
                ("STYLING", "OUTFIT", "JEANS", "POSITIVE"),
            ),
            (
                "전체 핏이 딱 맞는 편은 아니에요.",
                ("SIZE_FIT", "OVERALL_FIT", "EXACT", "POSITIVE"),
            ),
            (
                "신발이 가벼운 편은 아니에요.",
                ("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE"),
            ),
            (
                "바닥이 얇은 편은 아니에요.",
                ("CUSHION_FEEL", "SOLE_THICKNESS", "THIN", "NEGATIVE"),
            ),
        )
        for text, forbidden in cases:
            with self.subTest(text=text):
                self.assertNotIn(forbidden, _signals(text))

    def test_indirect_negations_do_not_become_positive_evidence(self) -> None:
        cases = (
            (
                "뒤꿈치가 안정적으로 잡아주는 느낌은 아니에요.",
                ("HEEL_FEEL", "HEEL_HOLD", "STABLE", "POSITIVE"),
            ),
            (
                "쿠션이 푹신하다고 느끼지는 않았어요.",
                ("CUSHION_FEEL", "CUSHION_SOFTNESS", "SOFT", "POSITIVE"),
            ),
            (
                "착화감이 편하다는 느낌은 없어요.",
                ("OTHER", "WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            ),
            (
                "디자인이 예쁘다는 생각은 들지 않았어요.",
                ("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE"),
            ),
            (
                "청바지에 잘 어울린다고 느끼지는 않았어요.",
                ("STYLING", "OUTFIT", "JEANS", "POSITIVE"),
            ),
            (
                "착화 시 가볍다는 느낌은 없어요.",
                ("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE"),
            ),
            (
                "발목을 안정적으로 잡아주는 편은 아닙니다.",
                ("HEEL_FEEL", "HEEL_HOLD", "STABLE", "POSITIVE"),
            ),
            (
                "쿠션이 푹신한 편은 아닙니다.",
                ("CUSHION_FEEL", "CUSHION_SOFTNESS", "SOFT", "POSITIVE"),
            ),
            (
                "청바지에 잘 어울리는 편은 아닙니다.",
                ("STYLING", "OUTFIT", "JEANS", "POSITIVE"),
            ),
            (
                "가벼운 편은 아닙니다.",
                ("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE"),
            ),
        )
        for text, forbidden in cases:
            with self.subTest(text=text):
                self.assertNotIn(forbidden, _signals(text))

    def test_non_wear_duration_does_not_become_long_wear_comfort(self) -> None:
        for text in (
            "디자인은 오래 봐도 좋아요.",
            "오래 고민했지만 디자인은 좋아요.",
            "배송은 오래 걸렸지만 신발은 좋아요.",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    any(
                        subject == "LONG_WEAR_COMFORT"
                        for _, subject, _, _ in _signals(text)
                    )
                )

        self.assertFalse(
            any(
                subject == "LONG_WEAR_COMFORT"
                for _, subject, _, _ in _signals(
                    "오래 신지 않았지만 발볼은 편해요."
                )
            )
        )

    def test_unrelated_positive_subject_does_not_validate_size_selection(self) -> None:
        for text in (
            "반업했는데 디자인은 좋아요.",
            "한 사이즈 업했는데 색감은 좋아요.",
            "반업 추천은 별로지만 디자인은 좋아요.",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    any(
                        subject == "SIZE_OPTION" and stance == "POSITIVE"
                        for _, subject, _, stance in _signals(text)
                    )
                )

    def test_plain_size_selection_without_fit_outcome_is_neutral(self) -> None:
        self.assertIn(
            ("SIZE_FIT", "SIZE_OPTION", "HALF_UP", "NEUTRAL"),
            _signals("반업했는데 잘 모르겠어요."),
        )

    def test_repeated_subject_mentions_preserve_both_directions(self) -> None:
        design = _signals("디자인은 예쁘지만 착화감은 불편하고 색감은 별로예요.")
        self.assertIn(("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE"), design)
        self.assertIn(("DESIGN", "APPEARANCE", "NEGATIVE", "NEGATIVE"), design)

        width = _signals(
            "발볼은 좁지만 착화감은 편하고 실제 발볼은 여유 있어요."
        )
        self.assertIn(("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE"), width)
        self.assertIn(("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"), width)

    def test_unrelated_exact_wording_does_not_become_overall_fit(self) -> None:
        for text in (
            "디자인이 제 취향에 딱 맞아요. 사이즈는 별로예요.",
            "색상이 딱 맞는 취향이고 사이즈는 보통이에요.",
        ):
            with self.subTest(text=text):
                self.assertNotIn(
                    ("SIZE_FIT", "OVERALL_FIT", "EXACT", "POSITIVE"),
                    _signals(text),
                )

    def test_negated_softness_does_not_imply_firm_cushion(self) -> None:
        for text in (
            "쿠션이 푹신하지 않지만 딱딱하지도 않아요.",
            "쿠션이 부드럽지 않지만 단단한 편도 아닙니다.",
            "쿠션은 푹신하지도 딱딱하지도 않은 보통입니다.",
        ):
            with self.subTest(text=text):
                signals = _signals(text)
                self.assertFalse(
                    any(subject == "CUSHION_SOFTNESS" for _, subject, _, _ in signals)
                )

    def test_long_wear_polarity_stays_inside_duration_subject(self) -> None:
        comfortable = _signals("오래 신어도 편하지만 발볼은 좁아 불편해요.")
        self.assertIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            comfortable,
        )
        self.assertNotIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"),
            comfortable,
        )

        also_comfortable = _signals(
            "장시간 착용해도 편하고 뒤꿈치는 쓸려서 아파요."
        )
        self.assertIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            also_comfortable,
        )
        self.assertNotIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"),
            also_comfortable,
        )

    def test_comfort_or_rubbing_alone_does_not_invent_space_direction(self) -> None:
        for text in (
            "발볼이 편해요.",
            "발볼도 편합니다.",
            "발가락이 까졌어요.",
            "앞코 때문에 발가락이 까져요.",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    any(
                        subject in {"WIDTH_SPACE", "TOEBOX_SPACE"}
                        for _, subject, _, _ in _signals(text)
                    )
                )

    def test_negated_silhouette_and_profile_are_not_positive_features(self) -> None:
        for text, subject, value in (
            ("실루엣이 날렵하지 않아요.", "SILHOUETTE", "SLIM"),
            ("실루엣은 날렵한 편이 아닙니다.", "SILHOUETTE", "SLIM"),
            ("디자인이 로우 프로파일은 아닙니다.", "PROFILE", "LOW"),
            ("실루엣은 로우 프로파일이 아니에요.", "PROFILE", "LOW"),
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    any(
                        signal_subject == subject and signal_value == value
                        for _, signal_subject, signal_value, _ in _signals(text)
                    )
                )


if __name__ == "__main__":
    unittest.main()
