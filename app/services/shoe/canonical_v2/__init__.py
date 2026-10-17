"""Isolated Canonical Shoe V2 staging and identity-matching package.

Nothing in this package is imported by the V1 recommendation runtime.
"""

from .matching import CanonicalShoeMatcher
from .models import (
    CanonicalShoeV2,
    MatchDecision,
    MatchStatus,
    RetailProductSource,
    RetailProductV2,
)

__all__ = [
    "CanonicalShoeMatcher",
    "CanonicalShoeV2",
    "MatchDecision",
    "MatchStatus",
    "RetailProductSource",
    "RetailProductV2",
]
