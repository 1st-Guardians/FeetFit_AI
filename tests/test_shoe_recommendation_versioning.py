from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.shoe.recommendation_policy_version import (
    DEFAULT_RECOMMENDATION_POLICY_VERSION,
    RecommendationPolicyVersion,
)
from app.services.shoe.shoe_fit_policy import (
    POLICY_VERSION,
    RECOMMENDATION_POLICY_VERSION,
    overall_fit_score,
    score_shoe_fit,
)
from app.services.shoe.shoe_recommendation import (
    ShoeFacts,
    ShoeRecommendationBatch,
    compute_shoe_recommendations,
    compute_shoe_recommendations_v1,
)
from app.services.shoe.shoe_recommendation_comparison import (
    compare_recommendation_batches,
    compare_recommendations,
)
from app.services.shoe.shoe_recommendation_v2 import (
    ShoeRecommendationV2Batch,
    ShoeRecommendationV2NotImplementedError,
    V2ShoeRecommendation,
    compute_shoe_recommendations_v2,
)
from tests.test_shoe_recommendation import _context


class ShoeRecommendationVersioningTests(unittest.TestCase):
    def test_v1_golden_scores_are_unchanged(self) -> None:
        context = _context()
        areas = score_shoe_fit(context.shoes[0], context.foot_state)

        self.assertEqual(
            [(area.reason_type, area.score, area.risk_level) for area in areas],
            [
                ("FOREFOOT", 89.43, "LOW"),
                ("HEEL", 85.62, "LOW"),
                ("INSOLE", 81.33, "LOW"),
            ],
        )
        self.assertEqual(overall_fit_score(areas), 85.86)

    def test_historical_fit_revision_is_explicitly_product_v1(self) -> None:
        self.assertEqual(POLICY_VERSION, "phase-d-v2-domain-map")
        self.assertIs(RECOMMENDATION_POLICY_VERSION, RecommendationPolicyVersion.V1)

    def test_backward_compatible_entry_point_delegates_to_v1(self) -> None:
        sentinel = ShoeRecommendationBatch(user_id=7, measurement_session_id=30, items=[])
        with patch(
            "app.services.shoe.shoe_recommendation.compute_shoe_recommendations_v1",
            return_value=sentinel,
        ) as v1:
            result = compute_shoe_recommendations(_context())

        self.assertIs(result, sentinel)
        v1.assert_called_once()
        self.assertIs(
            result.recommendation_policy_version,
            RecommendationPolicyVersion.V1,
        )
        self.assertIs(
            DEFAULT_RECOMMENDATION_POLICY_VERSION,
            RecommendationPolicyVersion.V1,
        )

    def test_v2_entry_point_fails_without_running_v1_or_making_scores(self) -> None:
        with self.assertRaises(ShoeRecommendationV2NotImplementedError):
            compute_shoe_recommendations_v2(_context())

    def test_comparison_uses_score_desc_and_shoe_id_as_tie_breaker(self) -> None:
        v1 = ShoeRecommendationBatch(
            user_id=7,
            measurement_session_id=30,
            items=[
                ShoeFacts(2, 80.0, []),
                ShoeFacts(1, 80.0, []),
                ShoeFacts(3, 70.0, []),
            ],
        )
        v2 = ShoeRecommendationV2Batch(
            user_id=7,
            measurement_session_id=30,
            items=(
                V2ShoeRecommendation(shoe_id=1, fit_score_v2=75.0),
                V2ShoeRecommendation(shoe_id=2, fit_score_v2=90.0),
                V2ShoeRecommendation(shoe_id=3, fit_score_v2=None),
            ),
        )

        compared = compare_recommendation_batches(v1, v2)
        by_shoe = {item.shoe_id: item for item in compared.items}

        self.assertEqual((by_shoe[1].v1_rank, by_shoe[2].v1_rank), (1, 2))
        self.assertEqual((by_shoe[1].v2_rank, by_shoe[2].v2_rank), (2, 1))
        self.assertEqual(by_shoe[1].score_delta, -5.0)
        self.assertEqual(by_shoe[1].rank_delta, 1)
        self.assertIsNone(by_shoe[3].v2_fit_score)

    def test_comparison_rejects_cross_session_data(self) -> None:
        v1 = ShoeRecommendationBatch(user_id=7, measurement_session_id=30, items=[])
        v2 = ShoeRecommendationV2Batch(user_id=7, measurement_session_id=31, items=())

        with self.assertRaisesRegex(ValueError, "same measurement session"):
            compare_recommendation_batches(v1, v2)

    def test_comparison_reports_v2_unavailable_without_fake_values(self) -> None:
        v1 = ShoeRecommendationBatch(
            user_id=7,
            measurement_session_id=30,
            items=[ShoeFacts(101, 85.86, [])],
        )

        compared = compare_recommendations(_context(), v1_runner=lambda _context: v1)

        self.assertFalse(compared.v2_available)
        self.assertIn("V2 is not implemented", compared.v2_unavailable_reason)
        self.assertEqual(compared.items[0].v1_fit_score, 85.86)
        self.assertIsNone(compared.items[0].v2_fit_score)

    def test_v1_and_v2_result_types_cannot_be_mislabeled(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use recommendation policy version V2"):
            V2ShoeRecommendation(
                shoe_id=101,
                recommendation_policy_version=RecommendationPolicyVersion.V1,
            )


if __name__ == "__main__":
    unittest.main()
