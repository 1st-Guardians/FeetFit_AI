from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RetailProductSource(StrEnum):
    MUSINSA = "MUSINSA"
    ABC_MART = "ABC_MART"
    TWENTY_NINE_CM = "29CM"


class MatchStatus(StrEnum):
    MATCHED = "MATCHED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class VariantProfile:
    generation: str | None = None
    gender: str | None = None
    width: str | None = None
    waterproof: str | None = None
    terrain: str | None = None
    age_group: str | None = None
    collaboration: str | None = None
    construction: tuple[str, ...] = ()
    color: str | None = None


@dataclass(frozen=True)
class NormalizedIdentity:
    brand: str
    model_name: str
    model_tokens: tuple[str, ...]
    skus: tuple[str, ...]
    sku_provenance: str
    variant: VariantProfile


@dataclass(frozen=True)
class CanonicalShoeV2:
    canonical_shoe_id: str
    brand: str
    canonical_model_name: str
    manufacturer_skus: tuple[str, ...]
    runrepeat_product_id: str
    runrepeat_url: str
    category: str | None = None
    target_gender: str | None = None
    variant: VariantProfile = field(default_factory=VariantProfile)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class RetailProductV2:
    source: RetailProductSource
    source_product_id: str
    source_product_name: str
    source_brand: str
    product_url: str
    source_skus: tuple[str, ...] = ()
    sku_provenance: str = "NONE"
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchDecision:
    status: MatchStatus
    confidence: float | None
    reason_codes: tuple[str, ...]
    reason: str
    product: RetailProductV2 | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        if self.product is not None:
            value["product"]["source"] = self.product.source.value
        return value
