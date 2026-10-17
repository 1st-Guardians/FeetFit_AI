from __future__ import annotations

import csv

import pytest

from app.services.shoe.canonical_v2.evaluation import (
    GROUND_TRUTH_COLUMNS,
    GroundTruthValidationError,
    error_rows,
    evaluate_rows,
    read_and_validate_labels,
)


def row(sample_id: str, predicted: str, label: str, *, reason="SKU_EXACT", source="MUSINSA", confidence="1.00"):
    value = {column: "" for column in GROUND_TRUTH_COLUMNS}
    value.update({
        "sampleId": sample_id, "runRepeatModel": f"Shoe {sample_id}",
        "source": source, "predictedStatus": predicted,
        "matchConfidence": confidence, "reasonCodes": reason,
        "humanLabel": label, "runRepeatVariant": "{}", "sourceVariant": "{}",
    })
    return value


def test_matched_precision_excludes_blank_but_keeps_ambiguous_in_denominator():
    result = evaluate_rows([
        row("1", "MATCHED", "TRUE_MATCH"),
        row("2", "MATCHED", "FALSE_MATCH"),
        row("3", "MATCHED", "AMBIGUOUS"),
        row("4", "MATCHED", ""),
    ])
    assert result["matchedPrecision"] == {
        "value": 0.333333, "labeledPredictedMatched": 3,
        "trueMatches": 1, "falseMatches": 1, "ambiguous": 1,
    }
    assert result["metadata"]["completionRatePercent"] == 75.0


def test_review_reason_rejected_false_negative_source_and_confidence_groups():
    rows = [
        row("1", "REVIEW_REQUIRED", "TRUE_MATCH", reason="SKU_COLLISION|MULTIPLE_MATCHES", confidence="1.00"),
        row("2", "REVIEW_REQUIRED", "FALSE_MATCH", reason="MODEL_NEAR_EXACT", source="29CM", confidence="0.77"),
        row("3", "REJECTED", "TRUE_MATCH", reason="SKU_CONFLICT", source="ABC_MART", confidence="0.05"),
    ]
    result = evaluate_rows(rows)
    assert result["reviewRequired"]["byReasonCode"]["SKU_COLLISION"]["TRUE_MATCH"] == 1
    assert result["rejectedFalseNegative"]["rejectedTrueMatches"] == 1
    assert result["confidenceCalibration"]["0.75-0.80_NEAR_EXACT"]["FALSE_MATCH"] == 1
    assert result["bySource"]["ABC_MART"]["rejectedFalseNegative"] == 1


def test_error_rows_have_required_priority():
    rows = [
        row("3", "REVIEW_REQUIRED", "TRUE_MATCH"),
        row("2", "REJECTED", "TRUE_MATCH"),
        row("1", "MATCHED", "FALSE_MATCH"),
        row("4", "MATCHED", "TRUE_MATCH"),
    ]
    assert [value["errorType"] for value in error_rows(rows)] == [
        "FALSE_POSITIVE", "FALSE_NEGATIVE", "REVIEW_TRUE_MATCH",
    ]


def test_invalid_label_reports_csv_row_and_value(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        writer.writerow(row("GT-001", "MATCHED", "MAYBE"))
    with pytest.raises(GroundTruthValidationError, match=r"row 2 sampleId=GT-001: 'MAYBE'"):
        read_and_validate_labels(path)


def test_blank_label_is_valid_and_not_evaluated(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        writer.writerow(row("GT-001", "MATCHED", ""))
    rows = read_and_validate_labels(path)
    assert evaluate_rows(rows)["metadata"] == {
        "totalSamples": 1, "labeledSamples": 0, "unlabeledSamples": 1,
        "completionRatePercent": 0.0,
        "confidenceInterpretation": "RULE_BASED_EVIDENCE_TIER_NOT_PROBABILITY",
    }
