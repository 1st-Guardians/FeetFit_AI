from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from app.services import wound_analysis as wound
from app.services import tinea_analysis as skin
from app.services import report_text_generation as report


class WoundInferenceTests(unittest.TestCase):
    def analyzer(self, maps):
        analyzer = object.__new__(wound.WoundAnalyzer)
        analyzer._inference_lock = threading.Lock()
        predictions = [(p, wound.image_tensor(Image.new("RGB", (p.shape[1], p.shape[0])), 352)[1]) for p in maps]
        analyzer._predict_probability = Mock(side_effect=predictions)
        analyzer.checkpoint_path = Path("weights/wound_segformer_experiment_c_best.pt")
        analyzer.checkpoint_sha256 = "test"
        analyzer.checkpoint_epoch = 15
        analyzer.device = "cpu"
        analyzer.input_size = 352
        analyzer.threshold = 0.5
        analyzer.redness_reduction = 0.2
        analyzer.weight_keys_converted = True
        return analyzer

    def test_single_pass_empty_mask_keeps_native_low_probabilities(self):
        probability = np.full((80, 120), 0.499, dtype=np.float32)
        a = self.analyzer([probability])
        result = a.analyze(Image.new("RGB", (120, 80)))
        self.assertEqual(a._predict_probability.call_count, 1)
        self.assertFalse(result.mask.any())
        np.testing.assert_array_equal(result.probability, probability)
        np.testing.assert_array_equal(result.coarse_probability, probability)
        self.assertEqual(result.mask.shape, (80, 120))
        self.assertIsNone(result.metrics["wound_roi_xyxy"])
        self.assertEqual(result.metrics["wound_inference_passes"], 1)
        self.assertEqual(result.metrics["wound_status"], "no_detection")

    def test_checkpoint_threshold_is_inclusive_and_no_roi_is_reinferred(self):
        probability = np.zeros((80, 120), dtype=np.float32)
        probability[30:40, 50:70] = 0.5
        probability[4:8, 7:12] = 0.499
        a = self.analyzer([probability])
        photo = Image.fromarray(np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3))
        with patch.object(wound, "reduce_redness", wraps=wound.reduce_redness) as correct:
            result = a.analyze(photo)
        self.assertEqual(a._predict_probability.call_count, 1)
        correct.assert_called_once()
        self.assertEqual(correct.call_args.args[1:], ("scale", 0.2))
        np.testing.assert_array_equal(np.asarray(a._predict_probability.call_args.args[0]), wound.reduce_redness(np.asarray(photo), "scale", .2))
        np.testing.assert_array_equal(np.asarray(photo), np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3))
        np.testing.assert_array_equal(result.mask, probability >= 0.5)
        self.assertEqual(int(result.mask.sum()), 200)
        self.assertEqual(result.metrics["wound_model"], "wound_segformer_b0")
        self.assertEqual(result.metrics["wound_threshold_source"], "checkpoint")
        self.assertEqual(result.metrics["wound_redness_reduction"], 0.2)
        self.assertEqual(result.metrics["wound_red_chroma_gain"], 0.8)

    def test_zero_redness_reduction_preserves_original_input_exactly(self):
        a = self.analyzer([np.zeros((80, 120), np.float32)])
        a.redness_reduction = 0
        photo = Image.new("RGB", (120, 80), (163, 109, 100))
        result = a.analyze(photo)
        np.testing.assert_array_equal(np.asarray(a._predict_probability.call_args.args[0]), np.asarray(photo))
        self.assertFalse(result.metrics["wound_color_preprocessing_applied"])

    def test_exif_rotation_defines_native_mask_coordinates(self):
        photo = Image.new("RGB", (120, 80))
        photo.getexif()[274] = 6
        a = self.analyzer([np.zeros((120, 80), dtype=np.float32)])
        result = a.analyze(photo)
        self.assertEqual(a._predict_probability.call_args.args[0].size, (80, 120))
        self.assertEqual(result.mask.shape, (120, 80))

    def test_invalid_probability_shape_nan_and_range_fail_closed(self):
        for probability in (np.zeros((79, 120), np.float32), np.full((80, 120), np.nan),
                            np.full((80, 120), -0.1), np.full((80, 120), 1.1)):
            with self.subTest(value=probability.flat[0], shape=probability.shape):
                with self.assertRaises(wound.WoundAnalysisError):
                    self.analyzer([probability]).analyze(Image.new("RGB", (120, 80)))

    def test_checkpoint_outside_project_is_rejected_before_loading(self):
        outside = wound.PROJECT_ROOT.parent / "external_wound_checkpoint.pt"
        with patch.object(wound.weights, "wound_segmentation", outside), patch.object(wound, "load_checkpoint") as load:
            with self.assertRaisesRegex(wound.WoundAnalysisError, "inside this project"):
                wound.WoundAnalyzer()
        load.assert_not_called()

    def test_cached_model_is_loaded_once(self):
        with patch.object(wound, "_analyzer", None), patch.object(wound, "WoundAnalyzer") as load:
            self.assertIs(wound.get_wound_analyzer(), wound.get_wound_analyzer())
            load.assert_called_once_with()

    def test_logit_restoration_precedes_sigmoid(self):
        a = object.__new__(wound.WoundAnalyzer)
        a.input_size, a.device = 32, torch.device("cpu")
        logits = torch.tensor([[[[-4., 4.]]]])
        a.model = Mock(return_value=SimpleNamespace(logits=logits))
        photo = Image.new("RGB", (5, 1), (80, 80, 80))
        probability, geometry = a._predict_probability(photo)
        wrong = wound.native_logits(logits.sigmoid(), [geometry], 32)[0][0].numpy()
        self.assertEqual(probability.shape, (1, 5))
        self.assertAlmostEqual(float(probability[0, 2]), 0.5, places=6)
        self.assertGreater(float(np.max(np.abs(probability - wrong))), 0.01)
        tensor = a.model.call_args.kwargs["pixel_values"]
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))
        self.assertEqual((geometry.original_width, geometry.original_height), (5, 1))


class WoundSkinIntegrationTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("tinea_inflammation_enabled", True), ("wound_segmentation_enabled", True), ("score_hard_weight", 0.7), ("score_sensitivity", 7.0), ("photo_cutout_background", False)):
            p = patch.object(skin.settings, name, value)
            p.start()
            self.addCleanup(p.stop)

    def prediction(self, probability):
        return wound.WoundPrediction(probability, probability >= 0.5, probability, {"wound_threshold": 0.5, "wound_model": "wound_segformer_b0"})

    def test_score_uses_final_inside_foot_area_and_keeps_confidence_as_metadata(self):
        foot = np.zeros((100, 100), dtype=np.uint8)
        foot[:50] = 255
        probability = np.full(foot.shape, 0.49, dtype=np.float32)
        probability[:10, :10] = 0.75  # 100/5000 = 2%, excess=(0.75-0.5)/0.5=0.5
        probability[70:90, :20] = 1.0  # outside foot, no penalty
        mask, metrics = skin.build_wound_skin_metrics(self.prediction(probability), foot, True)
        self.assertEqual(int(mask.sum()), 100)
        self.assertAlmostEqual(metrics["wound_area_ratio"], 0.02)
        self.assertAlmostEqual(metrics["wound_confidence_weighted_ratio"], 0.01)
        self.assertAlmostEqual(metrics["inflammation_burden"], 0.017)
        self.assertEqual(metrics["wound_safety_score"], 56.51)

    def test_tiny_final_wound_is_retained_and_starts_in_sixties(self):
        prob = np.zeros((100, 100), dtype=np.float32)
        prob[10, 10] = 0.99
        mask, metrics = skin.build_wound_skin_metrics(self.prediction(prob), np.ones_like(prob), True)
        self.assertTrue(mask[10, 10])
        self.assertTrue(metrics["wound_detected"])
        self.assertEqual(metrics["wound_safety_score"], 64.95)
        regions = skin.summarize_suspicion_regions(mask, prob, np.ones_like(prob), 0.9, "left", min_area_pixels=1)
        self.assertEqual(len(regions), 1)

    def test_larger_skin_regions_lower_score_independently_of_prediction_strength(self):
        small_probability = np.zeros((100, 100), dtype=np.float32)
        small_probability[:10, :10] = 0.99
        large_probability = np.zeros_like(small_probability)
        large_probability[:30, :30] = 0.51
        _, small = skin.build_wound_skin_metrics(
            self.prediction(small_probability), np.ones_like(small_probability), True,
        )
        _, large = skin.build_wound_skin_metrics(
            self.prediction(large_probability), np.ones_like(large_probability), True,
        )
        self.assertEqual(small["wound_safety_score"], 60.61)
        self.assertLess(large["wound_safety_score"], small["wound_safety_score"])

    def test_empty_mask_has_zero_penalty(self):
        p = np.full((20, 20), 0.49, dtype=np.float32)
        mask, metrics = skin.build_wound_skin_metrics(self.prediction(p), np.ones_like(p), True)
        self.assertFalse(mask.any())
        self.assertEqual(metrics["wound_safety_score"], 100)
        self.assertEqual(metrics["inflammation_burden"], 0)

    def test_report_fallback_preserves_tiny_wound_even_when_average_is_100(self):
        facts = {"tinea_pedis": {
            "final_scores": {"skin_reaction_safety_score": 100},
            "dorsal": {"left": {"skin_assessment_source": "wound_segformer_b0", "wound_detected": True}},
        }}
        text, signal = report._fallback_photo_skin_observation(facts, "skin_reaction_safety_score", "피부 자극")
        self.assertTrue(signal)
        self.assertIn("상처", text)
        self.assertNotIn("관찰되지", text)

    def test_dorsal_score_and_overlay_use_segformer_instead_of_legacy_skin_channel(self):
        photo = Image.new("RGB", (100, 100), (100, 80, 60))
        old_probs = np.zeros((3, 100, 100), dtype=np.float32)
        old_probs[0] = 1
        old_probs[2, 20:60, 20:60] = 1
        p = np.zeros((100, 100), dtype=np.float32)
        p[10:30, 10:30] = 0.97
        a = object.__new__(skin.TineaAnalyzer)
        a.tinea_model, a.sam_input_size, a.device = object(), 16, "cpu"
        a.fungal_threshold, a.inflammation_threshold = 0.78, 0.9
        with patch.object(skin, "load_image", return_value=(photo, np.array(photo)[..., ::-1])), patch.object(skin, "predict_tinea_probs_with_sliding_window", return_value=(old_probs, {})), patch.object(skin, "get_wound_analyzer", return_value=Mock(analyze=Mock(return_value=self.prediction(p)))):
            result = a.analyze(b"test", "dorsal.png", "left", foot_mask=np.ones((100, 100), dtype=np.uint8))
        self.assertLess(result.skin_reaction_safety_score, 100)
        self.assertEqual(result.metrics["inflammation_pixels"], 400)
        mask = cv2.imdecode(np.frombuffer(result.wound_mask_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(mask > 0, p >= 0.5)
        self.assertEqual(result.metrics["inflammation_analysis_mode"], "wound_segformer_single_pass")
        np.testing.assert_array_equal(old_probs[2, 20:60, 20:60], 1)

    def test_plantar_helper_rejects_before_loading_any_skin_models(self):
        with patch.object(skin, "get_tinea_analyzer") as fungal, patch.object(skin, "get_wound_analyzer") as injury:
            with self.assertRaises(skin.AnalysisError):
                skin.build_plantar_skin_result(b"unused", "plantar.jpg", "left", foot_mask=np.ones((2, 2)))
        fungal.assert_not_called()
        injury.assert_not_called()

    def test_fungal_only_threshold_ignores_larger_legacy_skin_probability(self):
        probs = np.zeros((3, 1, 3), dtype=np.float32)
        probs[1, 0] = [0.779, 0.78, 0.97]
        probs[2, 0] = [1.0, 1.0, 0.99]
        fungal, old_skin = skin.threshold_predictions(probs, 0.78, 0.9, fungal_only=True)
        np.testing.assert_array_equal(fungal, [[False, True, True]])
        self.assertFalse(old_skin.any())

    def test_public_skin_entrypoint_rejects_plantar_before_model_loading(self):
        with patch.object(skin, "get_tinea_analyzer") as fungal, patch.object(skin, "get_wound_analyzer") as injury:
            with self.assertRaises(skin.AnalysisError):
                skin.analyze_foot_image(b"unused", "plantar.png", "left", foot_surface="plantar")
        fungal.assert_not_called()
        injury.assert_not_called()


if __name__ == "__main__":
    unittest.main()
