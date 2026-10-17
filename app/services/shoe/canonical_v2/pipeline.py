from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .adapters import RetailProductSourceAdapter
from .matching import CanonicalShoeMatcher
from .models import CanonicalShoeV2, MatchDecision, MatchStatus, RetailProductSource
from .normalization import extract_variant


def _runrepeat_id(url: str) -> str:
    slug = urlparse(url).path.strip("/").split("/")[-1]
    return slug or hashlib.sha256(url.encode()).hexdigest()[:16]


def load_runrepeat_pages(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload.get("pages", []) if row.get("crawlStatus") == "SUCCEEDED"]


def stratified_sample(rows: list[dict], size: int) -> list[dict]:
    """Deterministic brand round-robin; keeps minority brands represented."""
    groups: dict[str, deque] = defaultdict(deque)
    for row in sorted(rows, key=lambda x: (str(x.get("brandName", "")).casefold(), str(x.get("shoeName", "")).casefold())):
        groups[str(row.get("brandName") or "UNKNOWN")].append(row)
    selected: list[dict] = []
    brands = sorted(groups)
    while len(selected) < min(size, len(rows)):
        progressed = False
        for brand in brands:
            if groups[brand] and len(selected) < size:
                selected.append(groups[brand].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def to_canonical(row: dict) -> CanonicalShoeV2:
    url = row.get("sourceUrl") or row.get("canonicalUrl") or ""
    rr_id = _runrepeat_id(url)
    cohort = next((m.get("comparisonCohort") for m in row.get("rawMetrics", []) if m.get("comparisonCohort")), None)
    return CanonicalShoeV2(
        canonical_shoe_id=f"RR:{rr_id}", brand=row.get("brandName") or "",
        canonical_model_name=row.get("shoeName") or "",
        manufacturer_skus=tuple(row.get("explicitSkus") or ()),
        runrepeat_product_id=rr_id, runrepeat_url=url, category=cohort,
        variant=extract_variant(row.get("shoeName")),
        created_at=row.get("capturedAt"), updated_at=row.get("capturedAt"),
    )


def _decision_row(canonical: CanonicalShoeV2, source: RetailProductSource, decision: MatchDecision) -> dict:
    product = decision.product
    return {
        "runRepeatId": canonical.runrepeat_product_id,
        "runRepeatModel": canonical.canonical_model_name,
        "runRepeatSku": "|".join(canonical.manufacturer_skus),
        "source": source.value,
        "found": decision.status != MatchStatus.NOT_FOUND,
        "sourceProductId": product.source_product_id if product else None,
        "sourceProductName": product.source_product_name if product else None,
        "sourceSku": "|".join(product.source_skus) if product else None,
        "matchStatus": decision.status.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "reasonCodes": list(decision.reason_codes),
        "productUrl": product.product_url if product else None,
        "evidence": decision.evidence,
    }


def run_dry_run(rows: list[dict], adapters: list[RetailProductSourceAdapter], sample_size: int = 75) -> dict:
    matcher = CanonicalShoeMatcher()
    results = []
    flat_rows = []
    for raw in stratified_sample(rows, sample_size):
        canonical = to_canonical(raw)
        source_results = {}
        for adapter in adapters:
            decision = matcher.select(canonical, adapter.search_product(canonical))
            result = _decision_row(canonical, adapter.source, decision)
            source_results[adapter.source.value] = result
            flat_rows.append(result)
        statuses = [row["matchStatus"] for row in source_results.values()]
        final = "MATCHED" if "MATCHED" in statuses else ("REVIEW_REQUIRED" if "REVIEW_REQUIRED" in statuses else ("REJECTED" if "REJECTED" in statuses else "NOT_FOUND"))
        results.append({
            "canonical": asdict(canonical), "sources": source_results,
            "canonicalMatchStatus": final,
            "selectedCanonicalIdentifiers": {"canonicalShoeId": canonical.canonical_shoe_id, "runRepeatProductId": canonical.runrepeat_product_id},
            "manualReviewRequired": final == "REVIEW_REQUIRED",
        })
    status_counts = Counter(row["matchStatus"] for row in flat_rows)
    canonical_status_counts = Counter(item["canonicalMatchStatus"] for item in results)
    source_counts = {source.value: Counter(row["matchStatus"] for row in flat_rows if row["source"] == source.value) for source in RetailProductSource}
    return {
        "format": "feetfit-canonical-shoe-v2-phase-2a-dry-run", "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "policy": {"falsePositivePriority": True, "confidenceType": "RULE_BASED_EVIDENCE_TIER", "groundTruthAccuracyMeasured": False},
        "summary": {
            "runRepeatSampleCount": len(results), "sourceDecisionCount": len(flat_rows),
            "statusCounts": dict(status_counts),
            "canonicalStatusCounts": dict(canonical_status_counts),
            "sourceStatusCounts": {key: dict(value) for key, value in source_counts.items()},
            "matchedAtLeastOneSource": sum(any(x["matchStatus"] == "MATCHED" for x in item["sources"].values()) for item in results),
            "manualReviewCanonicalCount": sum(item["manualReviewRequired"] for item in results),
        },
        "results": results, "auditRows": flat_rows,
    }


def write_artifacts(payload: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["runRepeatId", "runRepeatModel", "runRepeatSku", "source", "found", "sourceProductId", "sourceProductName", "sourceSku", "matchStatus", "confidence", "reason", "reasonCodes", "productUrl"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(payload["auditRows"], key=lambda x: (x["matchStatus"] != "REVIEW_REQUIRED", x["runRepeatModel"], x["source"])):
            output = dict(row)
            output["reasonCodes"] = "|".join(output["reasonCodes"])
            writer.writerow(output)
