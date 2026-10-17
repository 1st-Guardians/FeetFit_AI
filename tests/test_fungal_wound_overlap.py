from io import BytesIO
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from app.services import tinea_analysis as skin
from app.services import wound_analysis as wound
from app.services import report_text_generation as report


class FungalWoundOverlapTests(unittest.TestCase):
    def test_partial_overlap_uses_confidence_with_fungal_ties(self):
        fungal = np.zeros((7, 8), dtype=bool)
        fungal[1:4, 1:5] = True
        injury = np.zeros_like(fungal)
        injury[2:5, 3:7] = True
        fungal_prob = np.full(fungal.shape, 0.96, dtype=np.float32)
        wound_prob = np.full(fungal.shape, 0.94, dtype=np.float32)
        wound_prob[3, 3] = 0.98
        wound_prob[2, 4] = 0.96
        saved = [array.copy() for array in (fungal, injury, fungal_prob, wound_prob)]

        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            fungal, injury, fungal_prob, wound_prob, np.ones_like(fungal),
        )

        self.assertFalse((resolved_f & resolved_w).any())
        self.assertTrue(resolved_w[3, 3])
        self.assertFalse(resolved_f[3, 3])
        self.assertTrue(resolved_f[2, 4])
        self.assertFalse(resolved_w[2, 4])
        np.testing.assert_array_equal(resolved_f | resolved_w, fungal | injury)
        self.assertEqual(metrics["overlap_fungal_winner_pixels"], 3)
        self.assertEqual(metrics["overlap_wound_winner_pixels"], 1)
        for original, snapshot in zip((fungal, injury, fungal_prob, wound_prob), saved):
            np.testing.assert_array_equal(original, snapshot)

    def test_completely_contained_stronger_wound_is_fungal(self):
        fungal = np.zeros((20, 20), dtype=bool)
        fungal[2:18, 2:18] = True
        injury = np.zeros_like(fungal)
        injury[6:14, 6:14] = True
        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            fungal, injury, np.full(fungal.shape, 0.85),
            np.full(fungal.shape, 0.99), np.ones_like(fungal),
        )
        np.testing.assert_array_equal(resolved_f, fungal)
        self.assertFalse(resolved_w.any())
        self.assertEqual(metrics["contained_wound_pixels_reclassified_as_fungal"], 64)

    def test_identical_region_uses_confidence_instead_of_containment(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:17, 3:17] = True
        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            mask, mask, np.full(mask.shape, 0.95),
            np.full(mask.shape, 0.99), np.ones_like(mask),
        )
        self.assertFalse(resolved_f.any())
        np.testing.assert_array_equal(resolved_w, mask)
        self.assertEqual(metrics["contained_wound_component_count"], 0)

    def test_competition_cannot_leave_an_isolated_wound_inside_fungal(self):
        fungal = np.zeros((7, 8), dtype=bool)
        fungal[1:4, 1:5] = True
        injury = np.zeros_like(fungal)
        injury[2:5, 3:7] = True
        fungal_prob = np.full(fungal.shape, 0.96, dtype=np.float32)
        wound_prob = np.full(fungal.shape, 0.94, dtype=np.float32)
        wound_prob[2, 3] = 0.99
        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            fungal, injury, fungal_prob, wound_prob, np.ones_like(fungal),
        )
        self.assertTrue(resolved_f[2, 3])
        self.assertFalse(resolved_w[2, 3])
        self.assertTrue(resolved_w[4, 3])
        self.assertEqual(metrics["contained_wound_pixels_reclassified_as_fungal"], 1)

    def test_fungal_ring_contains_wound_but_other_hole_pixels_stay_empty(self):
        fungal = np.zeros((30, 30), dtype=bool)
        fungal[2:20, 2:20] = True
        fungal[5:17, 5:17] = False
        injury = np.zeros_like(fungal)
        injury[8:12, 8:12] = True
        injury[23:26, 23:26] = True
        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            fungal, injury, fungal.astype(float) * 0.95,
            injury.astype(float) * 0.99, np.ones_like(fungal),
        )
        self.assertTrue(resolved_f[8:12, 8:12].all())
        self.assertFalse(resolved_w[8:12, 8:12].any())
        self.assertFalse(resolved_f[5, 5])
        self.assertTrue(resolved_w[23:26, 23:26].all())
        np.testing.assert_array_equal(resolved_f | resolved_w, fungal | injury)
        self.assertEqual(metrics["contained_wound_component_count"], 1)
        self.assertEqual(metrics["contained_wound_pixels_reclassified_as_fungal"], 16)

    def test_independent_single_pixel_wound_remains_and_background_is_excluded(self):
        fungal = np.zeros((10, 10), dtype=bool)
        fungal[1:4, 1:4] = True
        injury = np.zeros_like(fungal)
        injury[6, 6] = True
        injury[9, 9] = True
        foot = np.ones_like(fungal)
        foot[9, 9] = False
        resolved_f, resolved_w, metrics = skin.resolve_fungal_wound_overlap(
            fungal, injury, fungal.astype(float), injury.astype(float), foot,
        )
        np.testing.assert_array_equal(resolved_f, fungal)
        self.assertEqual(int(resolved_w.sum()), 1)
        self.assertTrue(resolved_w[6, 6])
        self.assertEqual(metrics["contained_wound_component_count"], 0)

    def test_wound_marker_cannot_paint_inside_a_fungal_enclosure(self):
        fungal = np.zeros((100, 100), dtype=bool)
        fungal[10:50, 10:50] = True
        fungal[20:40, 20:40] = False
        injury = np.zeros_like(fungal)
        injury[55:65, 40:50] = True
        probs = np.zeros((3, 100, 100), dtype=np.float32)
        marker = {"center": (30, 50), "radius": 30, "layer": 0,
                  "confidence": 0.99, "threshold": 0.9,
                  "color": skin.MAP_INFLAMMATION_COLOR}
        with patch.object(skin, "circles_and_dots_from_overlay_mask",
                          side_effect=[([], []), ([marker], [])]):
            image, circles, _ = skin.render_suspicion_map(
                (100, 100), np.ones_like(fungal), probs, fungal, injury, 0.78, 0.9,
            )
        self.assertEqual(circles, 1)
        np.testing.assert_array_equal(image[45, 35], [255, 255, 255])
        np.testing.assert_array_equal(image[30, 30], [255, 255, 255])
        self.assertFalse(np.array_equal(image[58, 45], [255, 255, 255]))


class FungalWoundScoringIntegrationTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("tinea_inflammation_enabled", True),
                            ("wound_segmentation_enabled", True),
                            ("score_hard_weight", 0.7), ("score_sensitivity", 7.0),
                            ("min_suspicion_area", 18), ("photo_cutout_background", False)):
            option = patch.object(skin.settings, name, value)
            option.start()
            self.addCleanup(option.stop)
        self.foot = np.ones((100, 100), dtype=np.uint8)

    def analyze(self, fungal_probability, wound_probability):
        photo = Image.new("RGB", (100, 100), (100, 80, 60))
        data = BytesIO()
        photo.save(data, format="PNG")
        probabilities = np.zeros((3, 100, 100), dtype=np.float32)
        probabilities[1] = fungal_probability
        probabilities[0] = 1 - fungal_probability
        prediction = wound.WoundPrediction(
            wound_probability, wound_probability >= 0.9, wound_probability,
            {"wound_threshold": 0.9, "wound_model": "wound_segformer_b0"},
        )
        analyzer = object.__new__(skin.TineaAnalyzer)
        analyzer.tinea_model, analyzer.sam_input_size, analyzer.device = object(), 16, "cpu"
        analyzer.fungal_threshold, analyzer.inflammation_threshold = 0.78, 0.9
        with patch.object(skin, "predict_tinea_probs_with_sliding_window", return_value=(probabilities, {})), \
             patch.object(skin, "get_wound_analyzer", return_value=Mock(analyze=Mock(return_value=prediction))):
            result = analyzer.analyze(data.getvalue(), "dorsal.png", "left",
                                      foot_mask=self.foot, foot_surface="dorsal")
        injury = cv2.imdecode(np.frombuffer(result.wound_mask_png, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        return result, injury

    def test_partial_overlap_updates_both_scores_and_exported_masks(self):
        fungal = np.zeros(self.foot.shape, dtype=np.float32)
        fungal[10:60, 10:50] = 0.96
        injury = np.zeros_like(fungal)
        injury[30:70, 30:70] = 0.98
        injury[30:40, 30:50] = 0.94
        result, wound_mask = self.analyze(fungal, injury)

        expected_f = fungal >= 0.78
        expected_f[40:60, 30:50] = False
        expected_w = injury >= 0.9
        expected_w[30:40, 30:50] = False
        np.testing.assert_array_equal(result.authoritative_fungal_mask, expected_f)
        np.testing.assert_array_equal(wound_mask, expected_w)
        self.assertFalse((expected_f & wound_mask).any())
        self.assertEqual(result.metrics["fungal_pixels"], 1600)
        self.assertEqual(result.metrics["inflammation_pixels"], 1400)
        self.assertTrue(result.metrics["wound_detected"])
        # Confidence belongs only to the winning class, so neither score
        # includes the other's overlapping pixels.
        self.assertAlmostEqual(result.metrics["fungal_confidence_weighted_ratio"], 0.16 * (0.96 - 0.78) / 0.22, places=6)
        self.assertAlmostEqual(result.metrics["wound_confidence_weighted_ratio"], 0.14 * (0.98 - 0.9) / 0.1, places=6)
        self.assertFalse(result.authoritative_fungal_probability[~expected_f].any())
        self.assertEqual(sum(region["area_pixels"] for region in result.metrics["inflammation_regions"]), 1400)
        overlay = cv2.imdecode(np.frombuffer(result.photo_overlay_png, np.uint8), cv2.IMREAD_COLOR)
        # Display turns clockwise; native masks and confidence retain the
        # original coordinates used above for scoring.
        # Dorsal overlays retain the native orientation.
        self.assertGreater(int(overlay[35, 35, 0]), 60)  # fungal blue
        self.assertLess(int(overlay[35, 35, 2]), 100)
        self.assertGreater(int(overlay[50, 40, 2]), 100)  # wound red

    def test_fungal_loses_all_overlap_without_residual_fungal_score_penalty(self):
        fungal = np.zeros(self.foot.shape, dtype=np.float32)
        fungal[20:50, 20:50] = 0.97
        injury = np.zeros_like(fungal)
        injury[10:60, 10:60] = 0.99
        result, mask = self.analyze(fungal, injury)
        self.assertFalse(result.authoritative_fungal_mask.any())
        self.assertEqual(result.fungal_safety_score, 100)
        self.assertEqual(result.metrics["fungal_confidence_weighted_ratio"], 0)
        self.assertEqual(result.metrics["fungal_burden"], 0)
        self.assertFalse(result.metrics["fungal_signal_detected"])
        self.assertEqual(result.metrics["fungal_regions"], [])
        self.assertEqual(int(mask.sum()), 2500)

    def test_enclosed_wound_changes_only_fungal_score_and_report_facts(self):
        fungal = np.zeros(self.foot.shape, dtype=np.float32)
        fungal[10:70, 10:70] = 0.95
        fungal[30:50, 30:50] = 0
        injury = np.zeros_like(fungal)
        injury[40:45, 40:45] = 0.99
        result, mask = self.analyze(fungal, injury)
        self.assertFalse(mask.any())
        self.assertEqual(result.skin_reaction_safety_score, 100)
        self.assertEqual(result.metrics["wound_confidence_weighted_ratio"], 0)
        self.assertFalse(result.metrics["wound_detected"])
        self.assertEqual(result.metrics["inflammation_regions"], [])
        self.assertEqual(result.metrics["max_inflammation_prob"], 0)
        self.assertEqual(result.metrics["fungal_pixels"], 3225)
        self.assertTrue(result.authoritative_fungal_mask[40:45, 40:45].all())
        self.assertFalse(result.authoritative_fungal_mask[35, 35])
        payload = report._side_payload(result)
        self.assertFalse(payload["wound_detected"])
        self.assertEqual(payload["inflammation_regions"], [])
        self.assertEqual(payload["contained_wound_pixels_reclassified_as_fungal"], 25)


if __name__ == "__main__":
    unittest.main()
