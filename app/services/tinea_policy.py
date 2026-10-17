from __future__ import annotations

import math
from collections.abc import Iterable


# A rounded safety score at or above this value is reported as having no
# meaningful visible signal. Keep the decision in one place so report text and
# rendered segmentation cannot disagree about borderline model activations.
TINEA_SIGNAL_SAFETY_SCORE_CUTOFF = 95


def aggregate_tinea_safety_scores(scores: Iterable[int | float]) -> int:
    """Return the integer API score for the arithmetic mean of surface scores.

    Safety scores are constrained to ``0..100``.  The backend report contract
    stores an integer, so the mean is rounded once after aggregation using the
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

    mean_score = math.fsum(values) / len(values)
    return int(math.floor(mean_score + 0.5))


def safety_score_indicates_tinea_signal(score: int | float) -> bool:
    """Return the categorical signal decision used by every tinea consumer."""

    return int(round(float(score))) < TINEA_SIGNAL_SAFETY_SCORE_CUTOFF
