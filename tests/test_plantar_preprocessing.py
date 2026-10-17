from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

import numpy as np

from app.plantar_full_sam.foot_roi import FootRoiResult
from app.plantar_full_sam.preprocessing import (
    MAX_PREPROCESSING_ATTEMPTS,
    iter_plantar_foot_candidates,
    iter_plantar_foot_inputs,
    prepare_plantar_foot_input,
)


def _roi(image: np.ndarray, *, found: bool, confidence: float = 0.0) -> FootRoiResult:
    height, width = image.shape[:2]
    mask = np.full((height, width), int(found), dtype=np.uint8)
    return FootRoiResult(
        found=found,
        mask=mask,
        bbox=(0, 0, width, height),
        crop_image=image,
        crop_mask=mask,
        confidence=confidence,
        instance_count=int(found),
        reason="" if found else "yolo_foot_not_detected",
    )


class PlantarPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.array(
            [
                [[1, 3, 5], [7, 9, 11], [90, 130, 200]],
                [[13, 15, 17], [19, 21, 23], [150, 190, 250]],
            ],
            dtype=np.uint8,
        )

    def test_rotates_once_and_stops_after_original_detection_succeeds(self) -> None:
        original = self.image.copy()
        seen = []
        rois = []

        def extract(image: np.ndarray) -> FootRoiResult:
            seen.append(image)
            self.assertEqual(image.shape, (3, 2, 3))
            roi = _roi(image, found=True, confidence=0.42)
            rois.append(roi)
            return roi

        result = prepare_plantar_foot_input(self.image, extract=extract)

        np.testing.assert_array_equal(result.image_rgb, np.rot90(original, -1))
        self.assertIs(result.image_rgb, seen[0])
        self.assertIs(result.roi, rois[0])
        self.assertFalse(np.shares_memory(result.image_rgb, self.image))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.audit["attempt_count"], 1)
        self.assertEqual(result.audit["selected"]["source"], "preprocessed")
        self.assertEqual(result.rotation_degrees_clockwise, 90)
        self.assertEqual(result.brightness_gain, 1.0)

    def test_original_full_foot_is_not_replaced_by_confident_bright_partial_mask(self) -> None:
        rotated = np.rot90(self.image, -1).copy()
        complete = _roi(rotated, found=True, confidence=0.66)
        partial = _roi(rotated, found=True, confidence=0.79)
        partial.mask[:2] = 0
        extract = Mock(side_effect=[complete, partial])

        result = prepare_plantar_foot_input(self.image, extract=extract)

        self.assertIs(result.roi, complete)
        self.assertTrue(np.all(result.roi.mask))
        self.assertEqual(result.brightness_gain, 1.0)
        self.assertEqual(result.audit["attempt_count"], 1)
        extract.assert_called_once()

    def test_no_foot_returns_rotated_failure_and_serializable_complete_audit(self) -> None:
        rois = []

        def extract(image: np.ndarray) -> FootRoiResult:
            roi = _roi(image, found=False)
            rois.append(roi)
            return roi

        result = prepare_plantar_foot_input(self.image, extract=extract)

        self.assertFalse(result.roi.found)
        np.testing.assert_array_equal(result.image_rgb, np.rot90(self.image, -1))
        self.assertIs(result.roi, rois[0])
        self.assertEqual(result.rotation_degrees_clockwise, 90)
        self.assertEqual(result.brightness_gain, 1.0)
        self.assertEqual(result.audit["attempt_count"], MAX_PREPROCESSING_ATTEMPTS)
        self.assertEqual(
            [(row["rotation_degrees_clockwise"], row["brightness_gain"])
             for row in result.audit["attempts"]],
            [(90, 1.0)],
        )
        self.assertEqual(
            [row["method"] for row in result.audit["attempts"]],
            ["original"],
        )
        self.assertEqual(
            result.audit["policy"],
            "fixed_quarter_turn_original_pixels_v4",
        )
        self.assertEqual(len(rois), 1)
        self.assertEqual(result.audit["selected"]["candidate_index"], 0)
        self.assertEqual(json.loads(json.dumps(result.audit)), result.audit)

    def test_candidate_iterator_exposes_exact_pixels_without_detector(self) -> None:
        original = self.image.copy()

        candidates = list(iter_plantar_foot_inputs(self.image))

        self.assertEqual([(rotation, gain) for rotation, gain, _ in candidates],
                         [(90, 1.0)])
        expected_rotated = np.array(
            [[[13, 15, 17], [1, 3, 5]],
             [[19, 21, 23], [7, 9, 11]],
             [[150, 190, 250], [90, 130, 200]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(candidates[0][2], expected_rotated)
        np.testing.assert_array_equal(self.image, original)
        for _, _, pixels in candidates:
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertFalse(np.shares_memory(pixels, self.image))

    def test_candidate_iterator_rejects_malformed_inputs_when_consumed(self) -> None:
        invalid_inputs = (
            (None, TypeError),
            ([[1, 2, 3]], TypeError),
            (np.zeros((2, 3), dtype=np.uint8), ValueError),
            (np.zeros((2, 3, 4), dtype=np.uint8), ValueError),
            (np.zeros((2, 3, 3), dtype=np.float32), ValueError),
            (np.zeros((0, 3, 3), dtype=np.uint8), ValueError),
        )
        for invalid, expected_error in invalid_inputs:
            with self.subTest(input_type=type(invalid), error=expected_error):
                with self.assertRaises(expected_error):
                    list(iter_plantar_foot_inputs(invalid))

    def test_detector_error_propagates_without_retry(self) -> None:
        failure = RuntimeError("model unavailable")
        extract = Mock(side_effect=failure)
        with self.assertRaises(RuntimeError) as caught:
            prepare_plantar_foot_input(self.image, extract=extract)
        self.assertIs(caught.exception, failure)
        extract.assert_called_once()

    def test_dark_and_saturated_pixels_are_preserved_without_correction(self) -> None:
        image = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        candidates = list(iter_plantar_foot_candidates(image))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].method, "original")
        np.testing.assert_array_equal(candidates[0].image_rgb[:, 0, 0], np.arange(256))

    def test_invalid_rgb_input_is_rejected_before_detector_call(self) -> None:
        for image in (
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 3, 4), dtype=np.uint8),
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((0, 3, 3), dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                extract = Mock()
                with self.assertRaises(ValueError):
                    prepare_plantar_foot_input(image, extract=extract)
                extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
