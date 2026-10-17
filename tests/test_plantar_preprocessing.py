from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

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

        with patch("app.plantar_full_sam.preprocessing._gamma_rgb") as gamma:
            result = prepare_plantar_foot_input(self.image, extract=extract)
        gamma.assert_not_called()

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

    def test_brightness_selection_keeps_rgb_and_roi_in_same_rotated_coordinates(self) -> None:
        seen = []
        rois = []
        original = self.image.copy()

        def extract(image: np.ndarray) -> FootRoiResult:
            seen.append(image)
            roi = _roi(image, found=len(seen) == 2, confidence=0.32)
            rois.append(roi)
            return roi

        result = prepare_plantar_foot_input(self.image, extract=extract)

        # The lower-left input pixel moves to the upper-left after clockwise 90.
        expected = np.array(
            [
                [[19, 22, 25], [1, 4, 7]],
                [[28, 31, 34], [10, 13, 16]],
                [[225, 255, 255], [135, 195, 255]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(result.image_rgb, expected)
        np.testing.assert_array_equal(self.image, original)
        self.assertIs(result.roi, rois[1])
        self.assertIs(result.image_rgb, seen[1])
        self.assertEqual(result.roi.mask.shape, result.image_rgb.shape[:2])
        self.assertEqual(result.rotation_degrees_clockwise, 90)
        self.assertEqual(result.brightness_gain, 1.5)
        self.assertEqual(len(seen), 2)
        self.assertEqual(result.audit["original_shape"], [2, 3, 3])
        self.assertEqual(result.audit["coordinate_space"], "prepared_image_rgb")

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

    def test_second_brightness_fallback_runs_after_two_failed_candidates(self) -> None:
        scores = iter([None, None, 0.40])

        def extract(image: np.ndarray) -> FootRoiResult:
            score = next(scores)
            return _roi(image, found=score is not None, confidence=score or 0.0)

        result = prepare_plantar_foot_input(self.image, extract=extract)

        self.assertEqual(result.rotation_degrees_clockwise, 90)
        self.assertEqual(result.brightness_gain, 2.0)
        self.assertEqual(result.audit["attempt_count"], 3)
        self.assertEqual(result.audit["selected"]["candidate_index"], 2)

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
             for row in result.audit["attempts"][:3]],
            [(90, 1.0), (90, 1.5), (90, 2.0)],
        )
        self.assertEqual(
            [row["method"] for row in result.audit["attempts"]],
            ["original", "gain_1.5", "gain_2", "gamma_0.65", "gamma_0.45",
             "clahe_luminance", "gamma_0.65_clahe_luminance", "percentile_luminance_stretch"],
        )
        self.assertEqual(
            result.audit["policy"],
            "fixed_quarter_turn_then_photometric_fallback_v3",
        )
        self.assertEqual(result.audit["selected"]["candidate_index"], 0)
        self.assertEqual(json.loads(json.dumps(result.audit)), result.audit)

    def test_every_brightness_candidate_uses_original_not_previous_enhancement(self) -> None:
        seen = []

        def extract(image: np.ndarray) -> FootRoiResult:
            seen.append(image.copy())
            return _roi(image, found=False)

        prepare_plantar_foot_input(self.image, extract=extract)

        # 90 (source top-right red) must be 180 at 2x, not 255 after 1.5x then 2x.
        self.assertEqual(len(seen), MAX_PREPROCESSING_ATTEMPTS)
        self.assertEqual(int(seen[0][2, 1, 0]), 90)
        self.assertEqual(int(seen[1][2, 1, 0]), 135)
        self.assertEqual(int(seen[2][2, 1, 0]), 180)
        for image in seen:
            self.assertEqual(image.shape, (3, 2, 3))
        np.testing.assert_array_equal(seen[0], np.rot90(self.image, -1))

    def test_candidate_iterator_exposes_exact_pixels_without_detector(self) -> None:
        original = self.image.copy()

        candidates = list(iter_plantar_foot_inputs(self.image))

        self.assertEqual([(rotation, gain) for rotation, gain, _ in candidates[:3]],
                         [(90, 1.0), (90, 1.5), (90, 2.0)])
        expected_rotated = np.array(
            [[[13, 15, 17], [1, 3, 5]],
             [[19, 21, 23], [7, 9, 11]],
             [[150, 190, 250], [90, 130, 200]]],
            dtype=np.uint8,
        )
        expected_brightest = np.array(
            [[[26, 30, 34], [2, 6, 10]],
             [[38, 42, 46], [14, 18, 22]],
             [[255, 255, 255], [180, 255, 255]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(candidates[0][2], expected_rotated)
        np.testing.assert_array_equal(candidates[2][2], expected_brightest)
        self.assertEqual(int(candidates[1][2][2, 1, 0]), 135)
        self.assertEqual(int(candidates[1][2][0, 0, 0]), 19)
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

    def test_detector_error_propagates_from_any_brightness_without_fallback(self) -> None:
        for successful_calls in (0, 2, 5, 7):
            with self.subTest(successful_calls=successful_calls):
                failure = RuntimeError("model unavailable")
                extract = Mock(
                    side_effect=[_roi(self.image, found=False)] * successful_calls + [failure]
                )
                with self.assertRaises(RuntimeError) as caught:
                    prepare_plantar_foot_input(self.image, extract=extract)
                self.assertIs(caught.exception, failure)
                self.assertEqual(extract.call_count, successful_calls + 1)

    def test_gamma_recovers_after_gain_failures_and_stops_before_clahe(self) -> None:
        image = np.repeat(np.array([[0, 64, 255]], dtype=np.uint8)[..., None], 3, axis=2)
        seen = []

        def extract(candidate):
            seen.append(candidate.copy())
            return _roi(candidate, found=len(seen) == 4, confidence=.8)

        with patch("app.plantar_full_sam.preprocessing._clahe_rgb") as clahe:
            result = prepare_plantar_foot_input(image, extract=extract)
        clahe.assert_not_called()
        self.assertEqual(len(seen), 4)
        # Gamma lifts shadows while keeping both endpoints, unlike gain clipping.
        np.testing.assert_array_equal(result.image_rgb[:, 0, 0], [0, 104, 255])
        self.assertEqual(result.audit["selected"]["method"], "gamma_0.65")
        self.assertEqual(result.audit["selected"]["parameters"], {"gamma_exponent": .65})
        self.assertEqual(result.brightness_gain, 1.0)

    def test_late_clahe_candidate_and_roi_are_handed_off_together(self) -> None:
        seen = []
        rois = []

        def extract(candidate):
            seen.append(candidate)
            roi = _roi(candidate, found=len(seen) == 6, confidence=.71)
            rois.append(roi)
            return roi

        result = prepare_plantar_foot_input(self.image, extract=extract)
        self.assertEqual(len(seen), 6)
        self.assertIs(result.image_rgb, seen[-1])
        self.assertIs(result.roi, rois[-1])
        self.assertEqual(result.audit["selected"]["method"], "clahe_luminance")
        self.assertEqual(result.audit["selected"]["parameters"]["clahe_clip_limit"], 2.0)

    def test_luminance_stretch_skips_almost_uniform_sensor_noise(self) -> None:
        image = np.full((64, 96, 3), 80, dtype=np.uint8)
        image[::2] = 81
        methods = [candidate.method for candidate in iter_plantar_foot_candidates(image)]
        self.assertNotIn("percentile_luminance_stretch", methods)
        self.assertLessEqual(len(methods), MAX_PREPROCESSING_ATTEMPTS)

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
