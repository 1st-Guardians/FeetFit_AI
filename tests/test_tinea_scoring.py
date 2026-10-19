from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from app.services import tinea_analysis as skin
from app.services.tinea_policy import (
    aggregate_tinea_safety_scores,
    lesion_safety_score_from_area,
)


class LesionSafetyScoreTests(unittest.TestCase):
    def test_absence_and_small_detected_regions_have_distinct_scores(self):
        self.assertEqual(lesion_safety_score_from_area(0.0), 100.0)
        self.assertEqual(lesion_safety_score_from_area(0.0001), 64.95)
        self.assertEqual(lesion_safety_score_from_area(0.01), 60.61)

    def test_wider_regions_reduce_scores_monotonically(self):
        scores = [lesion_safety_score_from_area(ratio) for ratio in (0.01, 0.05, 0.1, 0.2, 1.0)]
        self.assertEqual(scores[:4], [60.61, 45.8, 32.28, 16.03])
        self.assertTrue(all(left > right for left, right in zip(scores, scores[1:])))
        self.assertTrue(all(0 <= score <= 65 for score in scores))

    def test_invalid_area_ratios_are_rejected(self):
        for ratio in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                lesion_safety_score_from_area(ratio)

    def test_unilateral_detection_is_not_diluted_by_clear_opposite_foot(self):
        for scores in ((65, 100), (100, 65), (45.8, 100), (100, 32.28)):
            with self.subTest(scores=scores):
                self.assertEqual(aggregate_tinea_safety_scores(scores), math.floor(min(scores) + 0.5))
        self.assertEqual(aggregate_tinea_safety_scores((100, 100)), 100)
        self.assertEqual(aggregate_tinea_safety_scores((64.5, 100)), 65)

    def test_invalid_aggregate_scores_are_rejected(self):
        for scores in ((), (-1, 100), (101,), (math.nan,), (math.inf,)):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                aggregate_tinea_safety_scores(scores)


class FungalScoringGateTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("min_suspicion_area", 18), ("score_hard_weight", 0.7),
                            ("score_sensitivity", 7.0), ("tinea_inflammation_enabled", True)):
            option = patch.object(skin.settings, name, value)
            option.start()
            self.addCleanup(option.stop)

    def analyze_candidate(self, width):
        probabilities = np.zeros((3, 100, 100), dtype=np.float32)
        probabilities[1, 10:10 + width, 10:10 + width] = 1.0
        return skin.build_authoritative_analysis_masks(
            probs=probabilities,
            fungal_mask=probabilities[1] >= 0.78,
            inflammation_mask=np.zeros((100, 100), dtype=bool),
            foot_mask=np.ones((100, 100), dtype=np.uint8),
            foot_found=True, fungal_threshold=0.78, inflammation_threshold=0.9,
        )

    def test_new_score_does_not_turn_previously_rejected_candidates_into_detections(self):
        fungal, _, metrics, gate = self.analyze_candidate(5)
        self.assertFalse(fungal.any())
        self.assertFalse(gate["fungal_signal_detected"])
        self.assertGreaterEqual(gate["fungal_candidate_detection_score"], 95)
        self.assertLessEqual(gate["fungal_candidate_safety_score"], 65)
        self.assertEqual(metrics["fungal_score"], 100)

    def test_accepted_fungal_region_starts_in_sixties_and_decreases_with_area(self):
        small, _, small_metrics, small_gate = self.analyze_candidate(10)
        large, _, large_metrics, large_gate = self.analyze_candidate(30)
        self.assertTrue(small_gate["fungal_signal_detected"])
        self.assertTrue(large_gate["fungal_signal_detected"])
        self.assertEqual(int(small.sum()), 100)
        self.assertEqual(int(large.sum()), 900)
        self.assertEqual(small_metrics["fungal_score"], 60.61)
        self.assertLess(large_metrics["fungal_score"], small_metrics["fungal_score"])
        self.assertEqual(small_metrics["inflammation_score"], 100)


if __name__ == "__main__":
    unittest.main()
