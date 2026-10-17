from __future__ import annotations

from enum import StrEnum


class RecommendationPolicyVersion(StrEnum):
    """Product-level recommendation policy version.

    This is deliberately separate from the historical fit-policy revision
    string (``phase-d-v2-domain-map``).  That revision name predates product
    V2 and continues to identify the unchanged V1 implementation.
    """

    V1 = "V1"
    V2 = "V2"


DEFAULT_RECOMMENDATION_POLICY_VERSION = RecommendationPolicyVersion.V1


__all__ = [
    "DEFAULT_RECOMMENDATION_POLICY_VERSION",
    "RecommendationPolicyVersion",
]
