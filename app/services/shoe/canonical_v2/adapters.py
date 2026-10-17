from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path

from .models import CanonicalShoeV2, RetailProductSource, RetailProductV2
from .normalization import extract_title_skus, normalize_brand, normalize_identity, normalize_sku


class RetailProductSourceAdapter(ABC):
    source: RetailProductSource

    @abstractmethod
    def search_product(self, canonical: CanonicalShoeV2) -> list[RetailProductV2]: ...

    @abstractmethod
    def fetch_product_metadata(self, source_product_id: str) -> RetailProductV2 | None: ...

    def extract_sku(self, raw: dict[str, object]) -> tuple[str, ...]:
        return extract_title_skus(str(raw.get("product_name") or raw.get("sourceProductName") or ""))

    def normalize_product(self, product: RetailProductV2):
        return normalize_identity(product.source_brand, product.source_product_name, product.source_skus, product.sku_provenance)


class ArtifactCatalogAdapter(RetailProductSourceAdapter):
    """Read-only adapter over dated evidence artifacts; performs no live crawl."""

    def __init__(self, source: RetailProductSource, products: list[RetailProductV2]):
        self.source = source
        self._products = products
        self._by_id = {p.source_product_id: p for p in products}
        self._identity = {p.source_product_id: self.normalize_product(p) for p in products}

    def search_product(self, canonical: CanonicalShoeV2) -> list[RetailProductV2]:
        target = normalize_identity(canonical.brand, canonical.canonical_model_name, canonical.manufacturer_skus, "RUNREPEAT_EXPLICIT")
        results: list[tuple[int, RetailProductV2]] = []
        target_skus = set(target.skus)
        target_tokens = set(target.model_tokens)
        for product in self._products:
            identity = self._identity[product.source_product_id]
            if identity.brand != target.brand:
                continue
            sku_hit = bool(target_skus & set(identity.skus))
            overlap = len(target_tokens & set(identity.model_tokens))
            if sku_hit or overlap >= 1:
                results.append((100 if sku_hit else overlap, product))
        results.sort(key=lambda row: (-row[0], row[1].source_product_id))
        return [row[1] for row in results[:25]]

    def fetch_product_metadata(self, source_product_id: str) -> RetailProductV2 | None:
        return self._by_id.get(source_product_id)


class MusinsaArtifactAdapter(ArtifactCatalogAdapter):
    def __init__(self, candidates_path: Path):
        raw = json.loads(candidates_path.read_text(encoding="utf-8"))
        products = []
        for item in raw.get("candidates", []):
            name = item.get("englishShoeName") or item.get("koreanShoeName") or ""
            sku = normalize_sku(item.get("styleNo"))
            products.append(RetailProductV2(
                source=RetailProductSource.MUSINSA,
                source_product_id=str(item.get("goodsNo") or ""),
                source_product_name=name,
                source_brand=item.get("englishBrandName") or item.get("koreanBrandName") or "",
                product_url=item.get("musinsaUrl") or "",
                source_skus=(sku,) if sku else (),
                sku_provenance="STRUCTURED" if sku else "NONE",
                category=item.get("categoryName"),
                metadata={"koreanName": item.get("koreanShoeName"), "identityStatus": item.get("identityStatus")},
            ))
        super().__init__(RetailProductSource.MUSINSA, products)


class InventoryArtifactAdapter(ArtifactCatalogAdapter):
    def __init__(self, inventory_path: Path, source: RetailProductSource, sku_evidence_path: Path | None = None):
        site = "ABC" if source == RetailProductSource.ABC_MART else "29CM"
        evidence_skus: dict[str, set[str]] = {}
        if sku_evidence_path and sku_evidence_path.is_file():
            id_column = "abc_product_ids" if source == RetailProductSource.ABC_MART else "29cm_product_ids"
            with sku_evidence_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    sku = normalize_sku(row.get("normalized_sku"))
                    for source_id in (row.get(id_column) or "").split("|"):
                        source_id = source_id.strip()
                        if source_id and sku:
                            evidence_skus.setdefault(source_id, set()).add(sku)
        products = []
        with inventory_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("site") != site:
                    continue
                official_skus = tuple(sorted(evidence_skus.get(row.get("source_id", ""), set())))
                title_skus = official_skus or extract_title_skus(row.get("product_name"))
                provenance = "OFFICIAL" if official_skus and source == RetailProductSource.ABC_MART else ("TITLE_EXTRACTED" if title_skus else "NONE")
                products.append(RetailProductV2(
                    source=source,
                    source_product_id=row.get("source_id", ""),
                    source_product_name=row.get("product_name", ""),
                    source_brand=row.get("brand", ""),
                    product_url=row.get("source_url", ""),
                    source_skus=title_skus,
                    sku_provenance=provenance,
                    metadata={"familyId": row.get("family_id"), "artifactNormalizedModel": row.get("normalized_model")},
                ))
        super().__init__(source, products)


class AbcMartArtifactAdapter(InventoryArtifactAdapter):
    def __init__(self, inventory_path: Path, sku_evidence_path: Path | None = None):
        super().__init__(inventory_path, RetailProductSource.ABC_MART, sku_evidence_path)


class TwentyNineCmArtifactAdapter(InventoryArtifactAdapter):
    def __init__(self, inventory_path: Path, sku_evidence_path: Path | None = None):
        super().__init__(inventory_path, RetailProductSource.TWENTY_NINE_CM, sku_evidence_path)
