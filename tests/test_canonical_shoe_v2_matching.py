from __future__ import annotations

import pytest

from app.services.shoe.canonical_v2.matching import CanonicalShoeMatcher
from app.services.shoe.canonical_v2.models import (
    CanonicalShoeV2, MatchStatus, RetailProductSource, RetailProductV2,
)


def canonical(name="Nike Pegasus 41", skus=("FD2722-001",), brand="Nike"):
    return CanonicalShoeV2("RR:nike-pegasus-41", brand, name, skus, "nike-pegasus-41", "https://runrepeat.com/nike-pegasus-41")


def retail(name="Pegasus 41", skus=("FD2722001",), brand="나이키", product_id="1", provenance="STRUCTURED"):
    return RetailProductV2(RetailProductSource.MUSINSA, product_id, name, brand, f"https://example.test/{product_id}", skus, provenance)


@pytest.fixture
def matcher():
    return CanonicalShoeMatcher()


def test_exact_sku_matches(matcher):
    decision = matcher.evaluate(canonical(), retail())
    assert (decision.status, decision.confidence) == (MatchStatus.MATCHED, 1.0)
    assert "SKU_EXACT" in decision.reason_codes


def test_different_brand_rejected_even_if_sku_matches(matcher):
    assert matcher.evaluate(canonical(), retail(brand="Adidas")).status == MatchStatus.REJECTED


def test_different_generation_rejected(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(name="Pegasus 40", skus=()))
    assert decision.status == MatchStatus.REJECTED
    assert "PROTECTED_VARIANT_CONFLICT" in decision.reason_codes


@pytest.mark.parametrize("left,right", [
    ("Nike Pegasus 41 Men", "Nike Pegasus 41 Women"),
    ("Nike Pegasus 41", "Nike Pegasus 41 Wide"),
    ("Nike Pegasus 41 GTX", "Nike Pegasus 41"),
])
def test_protected_or_incomplete_variant_is_never_auto_matched(matcher, left, right):
    decision = matcher.evaluate(canonical(left, ()), retail(right, (), provenance="NONE"))
    assert decision.status in {MatchStatus.REJECTED, MatchStatus.REVIEW_REQUIRED}


def test_color_only_difference_stays_same_canonical(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(name="Nike Pegasus 41 Blue", skus=(), provenance="NONE"))
    assert decision.status == MatchStatus.MATCHED


def test_model_alias_korean_english(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(name="나이키 페가수스 41", skus=(), provenance="NONE"))
    assert decision.status == MatchStatus.MATCHED


def test_sku_missing_but_exact_identity_can_match(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(skus=(), provenance="NONE"))
    assert decision.status == MatchStatus.MATCHED
    assert decision.confidence == 0.86


def test_sku_conflict_rejected(matcher):
    decision = matcher.evaluate(canonical(), retail(skus=("FD9999-999",)))
    assert decision.status == MatchStatus.REJECTED
    assert "SKU_CONFLICT" in decision.reason_codes


def test_empty_product_code_is_not_treated_as_sku(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(skus=("",), provenance="STRUCTURED"))
    assert decision.status == MatchStatus.MATCHED
    assert "SKU_EXACT" not in decision.reason_codes


def test_marketing_phrase_does_not_change_identity(matcher):
    decision = matcher.evaluate(canonical(skus=()), retail(name="[29CM단독] 공식 Nike Pegasus 41 무료배송", skus=(), provenance="NONE"))
    assert decision.status == MatchStatus.MATCHED


def test_duplicate_exact_sku_requires_review(matcher):
    decision = matcher.select(canonical(), [retail(product_id="1"), retail(product_id="2")])
    assert decision.status == MatchStatus.REVIEW_REQUIRED
    assert "SKU_COLLISION" in decision.reason_codes


def test_no_search_candidate_is_not_found(matcher):
    assert matcher.select(canonical(), []).status == MatchStatus.NOT_FOUND


def test_collaboration_and_kids_are_separate(matcher):
    collab = matcher.evaluate(canonical("Nike Pegasus 41", ()), retail(name="Nike Pegasus 41 x Artist", skus=()))
    kids = matcher.evaluate(canonical("Nike Pegasus 41 Men", ()), retail(name="Nike Pegasus 41 Kids", skus=()))
    assert collab.status == MatchStatus.REJECTED
    assert kids.status == MatchStatus.REJECTED
