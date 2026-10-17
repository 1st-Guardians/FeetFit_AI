from __future__ import annotations

import unittest

from app.services.pressure_signal_quality import classify_weak_foot_signal


class PressureSignalQualityTests(unittest.TestCase):
    def classify(self, left, right, minimum=5.0):
        return classify_weak_foot_signal(
            left, right, minimum_foot_load_percent=minimum,
            sensor_peak_percent=5.0,
        )

    def test_nonzero_residuals_on_either_foot_are_detected_without_reversing_sides(self):
        weak = [0.01] * 12
        loaded = [1000.0] * 12
        for side, left, right in (("left", weak, loaded), ("right", loaded, weak)):
            with self.subTest(side=side):
                result = self.classify(left, right)
                self.assertEqual(result["lowPressureSide"], side)
                self.assertEqual(result[side + "EffectiveSensorCount"], 0)
                self.assertEqual(result["qualityStatus"], "single-side-low-load")

    def test_one_sensor_at_the_effective_threshold_prevents_all_weak_classification(self):
        loaded = [1000.0] * 12
        self.assertIsNone(self.classify([50.0] + [0.01] * 11, loaded))
        below = self.classify([49.99] + [0.01] * 11, loaded)
        self.assertEqual(below["leftEffectiveSensorCount"], 0)

    def test_five_percent_load_boundary_and_balanced_weak_frames_are_not_rejected(self):
        self.assertIsNone(self.classify([1.0] * 12, [228.0] + [0.0] * 11))
        self.assertIsNone(self.classify([0.01] * 12, [0.01] * 12))

    def test_original_real_frame_remains_valid_and_gate_can_be_disabled(self):
        left = [0, 0, 0, 2752.1001, 0, 6268.5, 11832, 6945, 1617.9, 458.8, 2453.3, 17295.4004]
        right = [16760.3008, 54.7, 329.6, 3363.5, 1951.3, 5612.5, 1256.6, 0, 0, 0, 0, 0]
        self.assertIsNone(self.classify(left, right))
        self.assertIsNone(self.classify([0.01] * 12, [1000.0] * 12, minimum=0))


if __name__ == "__main__":
    unittest.main()
