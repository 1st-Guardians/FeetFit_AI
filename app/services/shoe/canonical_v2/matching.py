from __future__ import annotations

from .models import CanonicalShoeV2, MatchDecision, MatchStatus, RetailProductV2, VariantProfile
from .normalization import normalize_identity


PROTECTED_VARIANTS = ("generation", "gender", "width", "waterproof", "terrain", "age_group", "collaboration", "construction")


def _variant_conflicts(left: VariantProfile, right: VariantProfile) -> tuple[str, ...]:
    conflicts: list[str] = []
    for key in PROTECTED_VARIANTS:
        a, b = getattr(left, key), getattr(right, key)
        if key == "gender" and {a, b} == {"UNISEX", "MEN"}:
            continue
        if key == "gender" and {a, b} == {"UNISEX", "WOMEN"}:
            continue
        if a and b and a != b:
            conflicts.append(key)
    return tuple(conflicts)


def _one_sided_variants(left: VariantProfile, right: VariantProfile) -> tuple[str, ...]:
    result = []
    for key in PROTECTED_VARIANTS:
        a, b = getattr(left, key), getattr(right, key)
        if bool(a) != bool(b):
            result.append(key)
    return tuple(result)


def _token_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a and b else 0.0


class CanonicalShoeMatcher:
    """Conservative identity matcher; confidence is evidence-tiered, not ML."""

    def evaluate(self, canonical: CanonicalShoeV2, product: RetailProductV2) -> MatchDecision:
        left = normalize_identity(canonical.brand, canonical.canonical_model_name, canonical.manufacturer_skus, "RUNREPEAT_EXPLICIT")
        right = normalize_identity(product.source_brand, product.source_product_name, product.source_skus, product.sku_provenance)
        evidence = {
            "canonicalBrand": left.brand, "retailBrand": right.brand,
            "canonicalSkus": left.skus, "retailSkus": right.skus,
            "modelSimilarity": round(_token_similarity(left.model_tokens, right.model_tokens), 4),
        }
        if not left.brand or not right.brand or left.brand != right.brand:
            return MatchDecision(MatchStatus.REJECTED, 0.0, ("BRAND_MISMATCH",), "정규화 브랜드가 일치하지 않음", product, evidence)
        conflicts = _variant_conflicts(left.variant, right.variant)
        if conflicts:
            evidence["variantConflicts"] = conflicts
            return MatchDecision(MatchStatus.REJECTED, 0.0, ("PROTECTED_VARIANT_CONFLICT",), "세대/성별/폭/방수/용도 등 보호 variant가 충돌함", product, evidence)
        sku_overlap = sorted(set(left.skus) & set(right.skus))
        if sku_overlap:
            evidence["matchedSkus"] = sku_overlap
            one_sided = _one_sided_variants(left.variant, right.variant)
            if one_sided:
                evidence["unverifiedVariants"] = one_sided
                return MatchDecision(MatchStatus.REVIEW_REQUIRED, 0.90, ("BRAND_EXACT", "SKU_EXACT", "VARIANT_INCOMPLETE"), "SKU는 일치하지만 한쪽 variant 정보만 있어 자동 확정하지 않음", product, evidence)
            confidence = 1.0 if product.sku_provenance in {"STRUCTURED", "OFFICIAL"} else 0.96
            return MatchDecision(MatchStatus.MATCHED, confidence, ("BRAND_EXACT", "SKU_EXACT"), "브랜드와 제조사 SKU가 정확히 일치함", product, evidence)
        if left.skus and right.skus:
            return MatchDecision(MatchStatus.REJECTED, 0.05, ("SKU_CONFLICT",), "양쪽에 SKU가 있으나 교집합이 없음", product, evidence)
        similarity = evidence["modelSimilarity"]
        one_sided = _one_sided_variants(left.variant, right.variant)
        if left.model_name == right.model_name and left.model_name:
            if one_sided:
                evidence["unverifiedVariants"] = one_sided
                return MatchDecision(MatchStatus.REVIEW_REQUIRED, 0.74, ("MODEL_EXACT", "VARIANT_INCOMPLETE"), "모델명은 같지만 한쪽 variant 정보만 있어 자동 확정할 수 없음", product, evidence)
            return MatchDecision(MatchStatus.MATCHED, 0.86, ("BRAND_EXACT", "MODEL_EXACT", "VARIANT_COMPATIBLE"), "SKU는 없지만 브랜드·정규화 모델명·variant가 정확히 일치함", product, evidence)
        if similarity >= 0.75:
            return MatchDecision(MatchStatus.REVIEW_REQUIRED, round(0.60 + 0.20 * similarity, 2), ("MODEL_NEAR_EXACT", "SKU_MISSING"), "모델명이 강하게 유사하지만 SKU exact 증거가 없음", product, evidence)
        return MatchDecision(MatchStatus.REJECTED, round(0.35 * similarity, 2), ("INSUFFICIENT_IDENTITY_EVIDENCE",), "동일 제품을 확정할 식별 증거가 부족함", product, evidence)

    def select(self, canonical: CanonicalShoeV2, candidates: list[RetailProductV2]) -> MatchDecision:
        if not candidates:
            return MatchDecision(MatchStatus.NOT_FOUND, None, ("NO_SEARCH_CANDIDATE",), "검색 artifact에 후보가 없음")
        decisions = [self.evaluate(canonical, item) for item in candidates]
        matched = [d for d in decisions if d.status == MatchStatus.MATCHED]
        if len(matched) > 1:
            ids = [d.product.source_product_id for d in matched if d.product]
            return MatchDecision(MatchStatus.REVIEW_REQUIRED, max(d.confidence or 0 for d in matched), ("MULTIPLE_MATCHES", "SKU_COLLISION"), "동일 강도 증거를 가진 복수 상품이 존재하여 수동 검수가 필요함", evidence={"candidateProductIds": ids})
        if matched:
            return matched[0]
        reviews = [d for d in decisions if d.status == MatchStatus.REVIEW_REQUIRED]
        if reviews:
            return max(reviews, key=lambda d: d.confidence or 0)
        return max(decisions, key=lambda d: d.confidence or 0)
