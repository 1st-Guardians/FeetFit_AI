from __future__ import annotations

import math
from collections.abc import Iterable


# A rounded safety score at or above this value is reported as having no
# meaningful visible signal. Keep the decision in one place so report text and
# rendered segmentation cannot disagree about borderline model activations.
TINEA_SIGNAL_SAFETY_SCORE_CUTOFF = 95
FUNGAL_PRESENT_SAFETY_SCORE = 94.0
FUNGAL_AREA_SCORE_SENSITIVITY = 4.0
LESION_PRESENT_SAFETY_SCORE = 65.0
LESION_AREA_SCORE_SENSITIVITY = 7.0


def _area_safety_score(area_ratio: float, baseline: float, sensitivity: float) -> float:
    ratio = float(area_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("Lesion area ratio must be finite and between 0 and 1.")
    if ratio == 0.0:
        return 100.0
    return round(
        baseline * math.exp(-sensitivity * ratio),
        2,
    )


def fungal_safety_score_from_area(area_ratio: float) -> float:
    """Score final fungal coverage gently while retaining a visible signal below 95."""

    return _area_safety_score(
        area_ratio, FUNGAL_PRESENT_SAFETY_SCORE, FUNGAL_AREA_SCORE_SENSITIVITY,
    )


def lesion_safety_score_from_area(area_ratio: float) -> float:
    """Keep the existing 65-point baseline for wound/skin-reaction coverage."""

    return _area_safety_score(
        area_ratio, LESION_PRESENT_SAFETY_SCORE, LESION_AREA_SCORE_SENSITIVITY,
    )


def aggregate_tinea_safety_scores(scores: Iterable[int | float]) -> int:
    """Return the lowest side's safety score so unilateral lesions stay visible.

    Safety scores are constrained to ``0..100``.  The backend report contract
    stores an integer, so the minimum is rounded once after aggregation using the
    user-facing ``0.5 rounds up`` rule.  This deliberately does not use
    Python's ties-to-even :func:`round` behavior.
    """

    values = tuple(float(score) for score in scores)
    if not values:
        raise ValueError("At least one tinea safety score is required.")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 100.0
        for value in values
    ):
        raise ValueError("Tinea safety scores must be finite values between 0 and 100.")

    return int(math.floor(min(values) + 0.5))


def safety_score_indicates_tinea_signal(score: int | float) -> bool:
    """Return the categorical signal decision used by every tinea consumer."""

    return int(round(float(score))) < TINEA_SIGNAL_SAFETY_SCORE_CUTOFF
