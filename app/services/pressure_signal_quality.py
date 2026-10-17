"""Conservative detection of a weak foot signal above a nonzero noise floor."""
from __future__ import annotations

import math
from typing import Sequence


def classify_weak_foot_signal(
    left: Sequence[float], right: Sequence[float], *,
    minimum_foot_load_percent: float, sensor_peak_percent: float,
) -> dict[str, float | int | str] | None:
    """Check validated frames; a weak share alone cannot establish absent load.

    Retry only when both the relative foot load and effective sensor support
    are insufficient. Absolute sensor-unit calibration is not assumed.
    """
    if minimum_foot_load_percent <= 0:
        return None
    left_total, right_total = math.fsum(left), math.fsum(right)
    pair_total = math.fsum((left_total, right_total))
    peak = max(max(left), max(right))
    if pair_total <= 0 or peak <= 0:
        # The existing no-load validation owns zero-signal frames.
        return None
    sensor_threshold = peak * (sensor_peak_percent / 100.0)
    left_count = sum(value >= sensor_threshold for value in left)
    right_count = sum(value >= sensor_threshold for value in right)
    left_percent = left_total / pair_total * 100.0
    right_percent = right_total / pair_total * 100.0
    for side, percent, count in (("left", left_percent, left_count), ("right", right_percent, right_count)):
        if percent < minimum_foot_load_percent and count == 0:
            return {
                "qualityStatus": "single-side-low-load",
                "lowPressureSide": side,
                "leftPressurePercent": left_percent,
                "rightPressurePercent": right_percent,
                "leftEffectiveSensorCount": left_count,
                "rightEffectiveSensorCount": right_count,
                "minimumFootLoadPercent": minimum_foot_load_percent,
                "minimumEffectiveSensorCount": 1,
                "effectiveSensorThreshold": sensor_threshold,
                "effectiveSensorPeakPercent": sensor_peak_percent,
            }
    return None
