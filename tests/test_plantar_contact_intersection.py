from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from app.services import plantar_full_sam_analysis as service
from tests.test_plantar_full_sam_analysis import _prepared_canvas


class PlantarContactIntersectionTests(unittest.TestCase):
    def test_both_feet_preserve_original_sole_rgb_and_keep_source_unchanged(self):
        image = np.array([[[0, 40, 80], [130, 170, 255], [255, 0, 200], [60, 90, 120]]], dtype=np.uint8)
        original = image.copy()
        sole = np.array([[True, True, True, False]])
        for side in ("left", "right"):
            with self.subTest(side=side):
                actual, audit = service._prepare_contact_model_input(image, sole, side)
                np.testing.assert_array_equal(actual[sole], original[sole])
                self.assertTrue(np.all(actual[~sole] == 255))
                self.assertEqual(audit["brightness_gain"], 1.0)
                self.assertEqual(audit["contrast_factor"], 1.0)
                self.assertFalse(np.shares_memory(actual, image))
                np.testing.assert_array_equal(image, original)

    def test_restored_flip_intersection_precedes_noninteger_display_resize(self):
        prepared = _prepared_canvas()
        source = np.arange(12 * 6 * 3, dtype=np.uint8).reshape(12, 6, 3)
        crop_sole = np.ones((12, 6), dtype=bool)
        crop_sole[0, 0] = False
        prepared = replace(
            prepared, contact_model_input_rgb=source, contact_model_sole=crop_sole,
            sole_canvas=np.ones((15, 9), dtype=np.uint8) * 255,
        )
        direct = np.zeros((12, 6), dtype=np.float32)
        restored_flip = direct.copy()
        direct[2, 1], restored_flip[2, 1] = .9, .1  # Mean TTA would wrongly retain this.
        direct[3, 4], restored_flip[3, 4] = .5, .8  # Inclusive threshold and asymmetric position.
        direct[5, 2], restored_flip[5, 2] = .3, .99
        direct[0, 0] = restored_flip[0, 0] = .99  # Never outside the sole.
        segmenter = Mock(device=torch.device("cpu"))
        segmenter.predict.side_effect = [
            SimpleNamespace(original_probabilities=(torch.from_numpy(direct),)),
            SimpleNamespace(original_probabilities=(torch.from_numpy(restored_flip[:, ::-1].copy()),)),
        ]
        with (
            patch.object(service, "get_plantar_contact_segmenter", return_value=(segmenter, {})),
            patch.object(service.settings, "plantar_contact_threshold", .5),
            patch.object(service, "_canvas_placement", return_value=(0, 0, 9, 15, 0, 0)),
        ):
            result = service.LocalPlantarFullSamAnalyzer._predict_pressed_contact(prepared)
        white_input = source.copy()
        white_input[~crop_sole] = 255
        self.assertEqual(segmenter.predict.call_count, 2)
        np.testing.assert_array_equal(segmenter.predict.call_args_list[0].args[0][0], white_input)
        np.testing.assert_array_equal(segmenter.predict.call_args_list[1].args[0][0], white_input[:, ::-1])
        expected_crop = np.zeros((12, 6), dtype=np.uint8)
        expected_crop[3, 4] = 1
        expected = cv2.resize(expected_crop, (9, 15), interpolation=cv2.INTER_NEAREST) > 0
        np.testing.assert_array_equal(result.mask, expected)
        np.testing.assert_array_equal(result.mask, result.probability >= .5)
        self.assertEqual(result.audit["tta"]["intersectionContactPixels"], 1)
        self.assertEqual(result.audit["tta"]["aggregation"], "binary_intersection")
        np.testing.assert_array_equal(prepared.contact_model_input_rgb, source)

    def test_empty_intersection_is_valid_without_single_view_fallback(self):
        segmenter = Mock(device=torch.device("cpu"))
        segmenter.predict.side_effect = [
            SimpleNamespace(original_probabilities=(torch.ones((12, 6)),)),
            SimpleNamespace(original_probabilities=(torch.zeros((12, 6)),)),
        ]
        with (
            patch.object(service, "get_plantar_contact_segmenter", return_value=(segmenter, {})),
            patch.object(service, "_canvas_placement", return_value=(0, 0, 6, 12, 2, 2)),
        ):
            result = service.LocalPlantarFullSamAnalyzer._predict_pressed_contact(_prepared_canvas())
        self.assertFalse(result.mask.any())
        self.assertFalse(result.probability.any())

    def test_invalid_second_inference_fails_instead_of_using_only_first(self):
        for second in (
            SimpleNamespace(original_probabilities=(torch.full((12, 6), float("nan")),)),
            SimpleNamespace(original_probabilities=(torch.zeros((11, 6)),)),
            RuntimeError("second inference failed"),
        ):
            with self.subTest(second=type(second).__name__):
                segmenter = Mock(device=torch.device("cpu"))
                segmenter.predict.side_effect = [
                    SimpleNamespace(original_probabilities=(torch.ones((12, 6)),)), second,
                ]
                with (
                    patch.object(service, "get_plantar_contact_segmenter", return_value=(segmenter, {})),
                    self.assertRaises(service.PlantarFullSamAnalysisError),
                ):
                    service.LocalPlantarFullSamAnalyzer._predict_pressed_contact(_prepared_canvas())


if __name__ == "__main__":
    unittest.main()
