from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
import sys
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from app.services import tinea_analysis as skin
from app.tinea import segmentation as fungal


class TinyNetwork(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.weight.expand(1, 1, 2, 2))


def stub_segmenter(probability=0.6):
    return SimpleNamespace(
        input_size=384, threshold=0.30, threshold_selection="validation_mean_dice",
        checkpoint_path=Path("tinea_segformer_b2_teacher_student_best.pt"),
        source_checkpoint_sha256="test-source", weight_keys_converted=True,
        predict_probability=Mock(side_effect=lambda image: np.full(
            (image.height, image.width), probability, dtype=np.float32)),
    )


class FungalPreprocessingTests(unittest.TestCase):
    def test_exact_rgb_square_imagenet_normalization(self):
        pixels = np.arange(9 * 17 * 3, dtype=np.uint8).reshape(9, 17, 3)
        image = Image.fromarray(pixels)
        actual = fungal.image_tensor(image, 384, "cpu")
        resized = image.resize((384, 384), Image.Resampling.BILINEAR)
        expected = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].float() / 255
        expected = (expected - torch.tensor(fungal.MEAN)[None, :, None, None]) / torch.tensor(fungal.STD)[None, :, None, None]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(np.array(image), pixels)

    def test_binary_sigmoid_and_training_interpolation_order(self):
        model = object.__new__(fungal.FungalSegmenter)
        model.device, model.input_size = "cpu", 384
        model._inference_lock = threading.Lock()
        logits = torch.tensor([[[[-4.0, 2.0]]]])
        model.network = Mock(return_value=SimpleNamespace(logits=logits))
        photo = Image.new("RGB", (7, 3), (80, 90, 100))
        actual = model.predict_probability(photo)
        square = F.interpolate(logits, size=(384, 384), mode="bilinear", align_corners=False)
        expected = F.interpolate(square.sigmoid(), size=(3, 7), mode="bilinear", align_corners=False)[0, 0].numpy()
        wrong = F.interpolate(square, size=(3, 7), mode="bilinear", align_corners=False).sigmoid()[0, 0].numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(np.abs(actual - wrong).max(), 1e-7)
        self.assertGreater(actual.max() - actual.min(), 0.5)
        model.network.assert_called_once()


class FungalCheckpointTests(unittest.TestCase):
    def checkpoint(self):
        return {
            "architecture": fungal.ARCHITECTURE,
            "model": TinyNetwork().state_dict(),
            "model_config": {"num_labels": 1, "num_channels": 3, "hidden_sizes": [64, 128, 320, 512],
                             "depths": [3, 4, 6, 3], "decoder_hidden_size": 768},
            "input_size": 384, "threshold": 0.30,
            "normalization": {"mean": fungal.MEAN, "std": fungal.STD},
        }

    def load(self, checkpoint):
        from transformers import SegformerConfig

        path = Path("student.pt")
        with patch.object(fungal.torch, "load", return_value=checkpoint) as load, \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(
                 SegformerConfig=SegformerConfig, SegformerForSemanticSegmentation=TinyNetwork)}):
            model = fungal.FungalSegmenter(path, "cpu")
        load.assert_called_once_with(path, map_location="cpu", weights_only=True)
        return model

    def test_loads_complete_student_and_checkpoint_threshold(self):
        model = self.load(self.checkpoint())
        self.assertEqual(model.threshold, 0.30)
        self.assertEqual(model.input_size, 384)
        self.assertFalse(model.network.training)
        self.assertFalse(model.weight_keys_converted)

    def test_rejects_incompatible_checkpoint_and_invalid_threshold(self):
        for key, value in (("threshold", float("nan")), ("threshold", 0), ("threshold", 1),
                           ("architecture", "sam"), ("input_size", 1024), ("input_size", 384.0),
                           ("normalization", {"mean": [0, 0, 0], "std": [1, 1, 1]})):
            with self.subTest(key=key, value=value):
                checkpoint = self.checkpoint()
                checkpoint[key] = value
                with self.assertRaises(ValueError):
                    self.load(checkpoint)
        checkpoint = self.checkpoint()
        checkpoint["model_config"]["num_labels"] = 3
        with self.assertRaises(ValueError):
            self.load(checkpoint)

    def test_parameter_conversion_preserves_tensors_and_requires_all_keys(self):
        tensor = torch.randn(2, 3)
        old = "segformer.encoder.block.0.0.attention.self.query.weight"
        new = "segformer.stages.0.blocks.0.attention.q_proj.weight"
        converted, changed = fungal.compatible_state_dict({old: tensor}, [new])
        self.assertTrue(changed)
        self.assertIs(converted[new], tensor)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            fungal.compatible_state_dict({old: tensor}, [new, "missing.weight"])

    def test_legacy_threshold_and_missing_sam_do_not_override_student(self):
        model = stub_segmenter()
        with patch.object(skin.settings, "fungal_threshold", 0.78), \
             patch.object(skin.settings, "sam_source_dir", Path("missing-sam")), \
             patch.object(skin.weights, "sam_checkpoint", Path("missing-sam.pth")), \
             patch.object(skin, "FungalSegmenter", return_value=model) as load:
            actual, size, threshold, _ = skin.load_tinea_model(Path(__file__), "cpu")
        load.assert_called_once_with(Path(__file__).resolve(), "cpu")
        self.assertIs(actual, model)
        self.assertEqual((size, threshold), (384, 0.30))

    def test_outside_project_checkpoint_is_rejected_before_loading(self):
        with patch.object(skin, "FungalSegmenter") as load:
            with self.assertRaisesRegex(skin.AnalysisError, "inside this project"):
                skin.load_tinea_model(skin.PROJECT_ROOT.parent / "external.pt", "cpu")
        load.assert_not_called()


class FungalServiceTests(unittest.TestCase):
    def test_one_foot_crop_restored_to_native_canvas_with_zero_wound_channel(self):
        foot = np.zeros((80, 120), np.uint8)
        foot[10:70, 30:90] = 1
        foot[20:25, 40:45] = 0
        photo = Image.new("RGB", (120, 80), (80, 90, 100))
        model = stub_segmenter()
        with patch.object(skin.settings, "tinea_sliding_window_enabled", True), \
             patch.object(skin.settings, "tinea_preprocess_enhance_enabled", True):
            probs, metrics = skin.predict_tinea_probs_with_sliding_window(model, photo, 384, "cpu", foot)
        model.predict_probability.assert_called_once()
        self.assertEqual(model.predict_probability.call_args.args[0].size, (60, 60))
        self.assertEqual(probs.shape, (3, 80, 120))
        np.testing.assert_allclose(probs[1][foot > 0], 0.6)
        self.assertFalse(probs[1][foot == 0].any())
        self.assertFalse(probs[2].any())
        np.testing.assert_allclose(probs.sum(axis=0), 1.0)
        self.assertEqual(metrics["fungal_input_roi_xyxy"], [30, 10, 90, 70])
        self.assertFalse(metrics["tinea_sliding_window_applied"])
        self.assertFalse(metrics["tinea_preprocess_enhance_enabled"])
        self.assertEqual(metrics["fungal_threshold_source"], "checkpoint")

    def test_threshold_is_inclusive_for_fungal_only(self):
        p = np.array([[0.2999, 0.30, 0.3001]], dtype=np.float32)
        mask, inflammation = skin.threshold_predictions(np.stack((1 - p, p, p * 0)), 0.30, 0.9, fungal_only=True)
        np.testing.assert_array_equal(mask, [[False, True, True]])
        self.assertFalse(inflammation.any())

    def test_analysis_keeps_blue_overlay_and_student_metadata(self):
        analyzer = object.__new__(skin.TineaAnalyzer)
        analyzer.tinea_model = stub_segmenter()
        analyzer.sam_input_size, analyzer.device = 384, "cpu"
        analyzer.fungal_threshold, analyzer.inflammation_threshold = 0.30, 0.90
        photo = Image.new("RGB", (120, 80), (80, 90, 100))
        data = BytesIO()
        photo.save(data, format="PNG")
        foot = np.zeros((80, 120), np.uint8)
        foot[10:70, 30:90] = 1
        with patch.object(skin.settings, "tinea_inflammation_enabled", False), \
             patch.object(skin.settings, "photo_cutout_background", False):
            result = analyzer.analyze(data.getvalue(), "foot.png", "left", foot_mask=foot)
        self.assertEqual(result.metrics["fungal_model"], fungal.ARCHITECTURE)
        self.assertEqual(result.metrics["fungal_threshold"], 0.30)
        self.assertEqual(result.authoritative_fungal_mask.shape, foot.shape)
        self.assertEqual(result.metrics["fungal_pixels"], int(foot.sum()))
        overlay = cv2.imdecode(np.frombuffer(result.photo_overlay_png, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(overlay[30, 50, 0]), 100)
        self.assertLess(int(overlay[30, 50, 2]), 80)
        self.assertEqual(result.metrics["fungal_active_classes"], ["fungal"])
        self.assertFalse(result.metrics["sam_inflammation_class_used"])


if __name__ == "__main__":
    unittest.main()
