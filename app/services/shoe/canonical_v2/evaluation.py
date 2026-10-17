from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .normalization import extract_variant


ALLOWED_HUMAN_LABELS = frozenset({"", "TRUE_MATCH", "FALSE_MATCH", "AMBIGUOUS"})
PREDICTED_STATUSES = ("MATCHED", "REVIEW_REQUIRED", "REJECTED", "NOT_FOUND")
HUMAN_LABELS = ("TRUE_MATCH", "FALSE_MATCH", "AMBIGUOUS")
MANDATORY_SOURCE_CASES = (
    ("asics-gel-1130", "MUSINSA"),
    ("nike-acg-mountain-fly-2-low", "MUSINSA"),
    ("adidas-adizero-boston-13", "MUSINSA"),
    ("puma-deviate-nitro-4", "29CM"),
    ("reebok-club-c-85-vintage", "MUSINSA"),
    ("nike-air-force-1-07", "29CM"),
)

GROUND_TRUTH_COLUMNS = (
    "sampleId",
    "runRepeatId", "runRepeatBrand", "runRepeatModel", "runRepeatSku",
    "runRepeatUrl", "runRepeatCategory", "runRepeatGender", "runRepeatVariant",
    "source", "sourceProductId", "sourceBrand", "sourceProductName",
    "sourceSku", "productUrl", "sourceCategory", "sourceGender", "sourceVariant",
    "predictedStatus", "matchConfidence", "reasonCodes", "matchReason",
    "humanLabel", "humanReason", "reviewerNote",
)


class GroundTruthValidationError(ValueError):
    pass


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_urls(source: str, candidate_ids: Iterable[str]) -> str:
    templates = {
        "MUSINSA": "https://www.musinsa.com/products/{id}",
        "ABC_MART": "https://abcmart.a-rt.com/product?prdtNo={id}",
        "29CM": "https://product.29cm.co.kr/catalog/{id}",
    }
    template = templates.get(source)
    return "|".join(template.format(id=value) for value in candidate_ids) if template else ""


def _review_priority(row: dict[str, Any]) -> int:
    codes = set(row.get("reasonCodes") or ())
    if codes & {"SKU_COLLISION", "MULTIPLE_MATCHES"}:
        return 0
    if "VARIANT_INCOMPLETE" in codes:
        variants = set((row.get("evidence") or {}).get("unverifiedVariants") or ())
        if "gender" in variants:
            return 1
        if "construction" in variants:
            return 2
        return 3
    if "MODEL_NEAR_EXACT" in codes:
        return 4
    return 5


def _confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence")
    return float(value) if value is not None else -1.0


def _select_one(rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("matchStatus") == status]
    if not candidates:
        raise GroundTruthValidationError(f"canonical status {status} has no matching source decision")
    if status == "REVIEW_REQUIRED":
        return min(candidates, key=lambda row: (_review_priority(row), -_confidence(row), row.get("source", "")))
    return max(candidates, key=lambda row: (_confidence(row), row.get("source", "")))


def _rejected_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    codes = row.get("reasonCodes") or ["UNKNOWN"]
    variants = (row.get("evidence") or {}).get("variantConflicts") or ["NONE"]
    return str(codes[0]), str(variants[0]), str(row.get("source") or "UNKNOWN")


def _select_rejected(results: list[dict[str, Any]], count: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for result in results:
        if result.get("canonicalMatchStatus") != "REJECTED":
            continue
        rows = list((result.get("sources") or {}).values())
        rejected = [row for row in rows if row.get("matchStatus") == "REJECTED"]
        if rejected:
            # Prefer explicit protected conflicts, then insufficient evidence,
            # then ordinary SKU conflicts for a more diagnostic review set.
            priority = {"PROTECTED_VARIANT_CONFLICT": 0, "INSUFFICIENT_IDENTITY_EVIDENCE": 1, "SKU_CONFLICT": 2}
            chosen = min(rejected, key=lambda row: (priority.get((row.get("reasonCodes") or [""])[0], 9), -_confidence(row), row.get("source", "")))
            candidates.append((result, chosen))

    groups: dict[tuple[str, str, str], deque[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(deque)
    for item in sorted(candidates, key=lambda pair: pair[0]["canonical"]["runrepeat_product_id"]):
        groups[_rejected_signature(item[1])].append(item)
    ordered_keys = sorted(groups, key=lambda key: ({"PROTECTED_VARIANT_CONFLICT": 0, "INSUFFICIENT_IDENTITY_EVIDENCE": 1, "SKU_CONFLICT": 2}.get(key[0], 9), key))
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while len(selected) < min(count, len(candidates)):
        progressed = False
        for key in ordered_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def _template_row(result: dict[str, Any], decision: dict[str, Any]) -> dict[str, str]:
    canonical = result["canonical"]
    evidence = decision.get("evidence") or {}
    candidate_ids = [str(value) for value in evidence.get("candidateProductIds") or ()]
    source = str(decision.get("source") or "")
    product_id = str(decision.get("sourceProductId") or "")
    product_url = str(decision.get("productUrl") or "")
    product_name = str(decision.get("sourceProductName") or "")
    if candidate_ids and not product_id:
        product_id = "|".join(candidate_ids)
        product_url = _candidate_urls(source, candidate_ids)
        product_name = f"[MULTIPLE CANDIDATES: {len(candidate_ids)}]"
    source_brand = str(evidence.get("retailBrand") or "")
    source_variant = extract_variant(product_name if not candidate_ids else "")
    runrepeat_variant = canonical.get("variant") or {}
    return {
        "sampleId": "",
        "runRepeatId": str(canonical.get("runrepeat_product_id") or ""),
        "runRepeatBrand": str(canonical.get("brand") or ""),
        "runRepeatModel": str(canonical.get("canonical_model_name") or ""),
        "runRepeatSku": "|".join(canonical.get("manufacturer_skus") or ()),
        "runRepeatUrl": str(canonical.get("runrepeat_url") or ""),
        "runRepeatCategory": str(canonical.get("category") or ""),
        "runRepeatGender": str(canonical.get("target_gender") or runrepeat_variant.get("gender") or ""),
        "runRepeatVariant": _json_compact(runrepeat_variant),
        "source": source,
        "sourceProductId": product_id,
        "sourceBrand": source_brand,
        "sourceProductName": product_name,
        "sourceSku": str(decision.get("sourceSku") or ""),
        "productUrl": product_url,
        "sourceCategory": "",
        "sourceGender": str(source_variant.gender or ""),
        "sourceVariant": _json_compact(source_variant.__dict__),
        "predictedStatus": str(decision.get("matchStatus") or ""),
        "matchConfidence": "" if decision.get("confidence") is None else f"{float(decision['confidence']):.2f}",
        "reasonCodes": "|".join(decision.get("reasonCodes") or ()),
        "matchReason": str(decision.get("reason") or ""),
        "humanLabel": "", "humanReason": "", "reviewerNote": "",
    }


def build_ground_truth_rows(payload: dict[str, Any], rejected_count: int = 15) -> list[dict[str, str]]:
    results = list(payload.get("results") or ())
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for status in ("MATCHED", "REVIEW_REQUIRED"):
        for result in results:
            if result.get("canonicalMatchStatus") == status:
                selected.append((result, _select_one(list((result.get("sources") or {}).values()), status)))
    selected.extend(_select_rejected(results, rejected_count))
    selected_keys = {
        (result["canonical"]["runrepeat_product_id"], decision.get("source"))
        for result, decision in selected
    }
    results_by_id = {result["canonical"]["runrepeat_product_id"]: result for result in results}
    for runrepeat_id, source in MANDATORY_SOURCE_CASES:
        if (runrepeat_id, source) in selected_keys:
            continue
        result = results_by_id.get(runrepeat_id)
        if result is None or source not in (result.get("sources") or {}):
            raise GroundTruthValidationError(f"mandatory audit case missing: {runrepeat_id} / {source}")
        selected.append((result, result["sources"][source]))
        selected_keys.add((runrepeat_id, source))
    rows = [_template_row(result, decision) for result, decision in selected]
    status_order = {"MATCHED": 0, "REVIEW_REQUIRED": 1, "REJECTED": 2, "NOT_FOUND": 3}
    def template_review_priority(row: dict[str, str]) -> int:
        codes = set(row["reasonCodes"].split("|"))
        if codes & {"SKU_COLLISION", "MULTIPLE_MATCHES"}:
            return 0
        if "VARIANT_INCOMPLETE" in codes:
            source_variant = json.loads(row["sourceVariant"])
            runrepeat_variant = json.loads(row["runRepeatVariant"])
            if row["sourceGender"] or runrepeat_variant.get("gender"):
                return 1
            if source_variant.get("construction") or runrepeat_variant.get("construction"):
                return 2
            return 3
        if "MODEL_NEAR_EXACT" in codes:
            return 4
        return 5

    rows.sort(key=lambda row: (
        status_order.get(row["predictedStatus"], 9),
        template_review_priority(row) if row["predictedStatus"] == "REVIEW_REQUIRED" else 0,
        -float(row["matchConfidence"] or -1), row["runRepeatModel"], row["source"],
    ))
    for index, row in enumerate(rows, 1):
        row["sampleId"] = f"GT-{index:03d}"
    return rows


def write_ground_truth_template(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_and_validate_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in GROUND_TRUTH_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise GroundTruthValidationError(f"missing required columns: {', '.join(missing)}")
        rows = []
        errors = []
        for csv_row_number, raw in enumerate(reader, 2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            label = row["humanLabel"]
            if label not in ALLOWED_HUMAN_LABELS:
                errors.append(f"row {csv_row_number} sampleId={row.get('sampleId', '')}: {row['humanLabel']!r}")
            row["humanLabel"] = label
            rows.append(row)
    if errors:
        raise GroundTruthValidationError("invalid humanLabel values:\n" + "\n".join(errors))
    return rows


def _label_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["humanLabel"] for row in rows if row["humanLabel"])
    return {label: counts[label] for label in HUMAN_LABELS}


def _reason_codes(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(code for code in row["reasonCodes"].split("|") if code)


def _variant_types(row: dict[str, str]) -> set[str]:
    result: set[str] = set()
    values = []
    for field in ("runRepeatVariant", "sourceVariant"):
        try:
            values.append(json.loads(row[field] or "{}"))
        except json.JSONDecodeError:
            values.append({})
    mapping = {
        "gender": "gender", "width": "width", "waterproof": "GTX/WP",
        "terrain": "trail/road", "generation": "generation",
        "construction": "construction suffix", "collaboration": "collaboration",
        "age_group": "kids/adult", "color": "colorway",
    }
    for key, label in mapping.items():
        if any(value.get(key) for value in values):
            result.add(label)
    reason_text = row["reasonCodes"] + " " + row.get("matchReason", "")
    for token, label in (("GENDER", "gender"), ("WIDTH", "width"), ("GTX", "GTX/WP"), ("GENERATION", "generation"), ("CONSTRUCTION", "construction suffix"), ("COLLAB", "collaboration"), ("KIDS", "kids/adult"), ("COLOR", "colorway")):
        if token in reason_text.upper():
            result.add(label)
    return result


def evaluate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["humanLabel"]]
    total, labeled_count = len(rows), len(labeled)
    confusion = {status: _label_counts(row for row in labeled if row["predictedStatus"] == status) for status in PREDICTED_STATUSES}
    matched = [row for row in labeled if row["predictedStatus"] == "MATCHED"]
    true_matched = sum(row["humanLabel"] == "TRUE_MATCH" for row in matched)
    false_matched = sum(row["humanLabel"] == "FALSE_MATCH" for row in matched)
    rejected_true = [row for row in labeled if row["predictedStatus"] == "REJECTED" and row["humanLabel"] == "TRUE_MATCH"]

    reason_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    review_reason_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labeled:
        for code in _reason_codes(row):
            reason_groups[code].append(row)
            if row["predictedStatus"] == "REVIEW_REQUIRED":
                review_reason_groups[code].append(row)

    confidence_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labeled:
        if "MODEL_NEAR_EXACT" in _reason_codes(row):
            tier = "0.75-0.80_NEAR_EXACT"
        else:
            tier = row["matchConfidence"] or "NONE"
        confidence_groups[tier].append(row)

    source_output = {}
    for source in ("MUSINSA", "ABC_MART", "29CM"):
        selected = [row for row in rows if row["source"] == source]
        evaluated = [row for row in labeled if row["source"] == source]
        source_output[source] = {
            "selectedCount": len(selected), "evaluatedCount": len(evaluated),
            "predictedMatched": sum(row["predictedStatus"] == "MATCHED" for row in selected),
            "trueMatch": sum(row["humanLabel"] == "TRUE_MATCH" for row in evaluated),
            "falseMatch": sum(row["humanLabel"] == "FALSE_MATCH" for row in evaluated),
            "reviewRequired": sum(row["predictedStatus"] == "REVIEW_REQUIRED" for row in selected),
            "rejectedFalseNegative": sum(row["predictedStatus"] == "REJECTED" and row["humanLabel"] == "TRUE_MATCH" for row in evaluated),
        }

    variant_output = {}
    for variant in ("gender", "width", "GTX/WP", "trail/road", "generation", "construction suffix", "collaboration", "kids/adult", "colorway"):
        group = [row for row in labeled if variant in _variant_types(row)]
        variant_output[variant] = {"total": len(group), **_label_counts(group)}

    return {
        "metadata": {
            "totalSamples": total, "labeledSamples": labeled_count,
            "unlabeledSamples": total - labeled_count,
            "completionRatePercent": round(100 * labeled_count / total, 2) if total else 0.0,
            "confidenceInterpretation": "RULE_BASED_EVIDENCE_TIER_NOT_PROBABILITY",
        },
        "matchedPrecision": {
            "value": round(true_matched / len(matched), 6) if matched else None,
            "labeledPredictedMatched": len(matched), "trueMatches": true_matched,
            "falseMatches": false_matched,
            "ambiguous": sum(row["humanLabel"] == "AMBIGUOUS" for row in matched),
        },
        "confusionStyleSummary": confusion,
        "reviewRequired": {
            **_label_counts(row for row in labeled if row["predictedStatus"] == "REVIEW_REQUIRED"),
            "byReasonCode": {key: {"total": len(value), **_label_counts(value)} for key, value in sorted(review_reason_groups.items())},
        },
        "rejectedFalseNegative": {
            "rejectedTrueMatches": len(rejected_true),
            "falseNegativeExamples": [{key: row[key] for key in ("sampleId", "runRepeatModel", "source", "sourceProductName", "reasonCodes", "humanReason")} for row in rejected_true],
        },
        "confidenceCalibration": {key: {"total": len(value), **_label_counts(value)} for key, value in sorted(confidence_groups.items())},
        "byReasonCode": {key: {"total": len(value), **_label_counts(value)} for key, value in sorted(reason_groups.items())},
        "bySource": source_output,
        "byVariant": variant_output,
    }


def error_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        error_type = None
        priority = 99
        if row["predictedStatus"] == "MATCHED" and row["humanLabel"] == "FALSE_MATCH":
            error_type, priority = "FALSE_POSITIVE", 0
        elif row["predictedStatus"] == "REJECTED" and row["humanLabel"] == "TRUE_MATCH":
            error_type, priority = "FALSE_NEGATIVE", 1
        elif row["predictedStatus"] == "REVIEW_REQUIRED" and row["humanLabel"] == "TRUE_MATCH":
            error_type, priority = "REVIEW_TRUE_MATCH", 2
        if error_type:
            selected.append((priority, {"errorType": error_type, **row}))
    return [row for _, row in sorted(selected, key=lambda value: (value[0], value[1]["sampleId"]))]


def write_evaluation_artifacts(rows: list[dict[str, str]], json_path: Path, errors_path: Path) -> dict[str, Any]:
    evaluation = evaluate_rows(rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    error_columns = ("errorType",) + GROUND_TRUTH_COLUMNS
    with errors_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_columns)
        writer.writeheader()
        writer.writerows(error_rows(rows))
    return evaluation
