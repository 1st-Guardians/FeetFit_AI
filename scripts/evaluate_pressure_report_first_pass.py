"""Compare pressure-report models with synthetic data and exactly one API attempt.

Examples (run from the repository root):
    python scripts/evaluate_pressure_report_first_pass.py --prepare-only
    python scripts/evaluate_pressure_report_first_pass.py --repeats 2
    python scripts/evaluate_pressure_report_first_pass.py --models gpt-4.1-mini gpt-4.1

Live runs use the configured OPENAI_API_KEY and incur API usage. No patient
photos or backend requests are used. Models see identical generated heatmaps.
Results include generated synthetic text for inspection; a validator pass alone
does not establish that every visual observation is accurate.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services import report_text_generation as reports
from app.services.pressure_heatmap_analysis import SavedPressureHeatmaps, create_pressure_heatmaps


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    left: list[float]
    right_canonical: list[float]


CASES = (
    SyntheticCase("balanced_broad", [1000] * 12, [1000] * 12),
    SyntheticCase("balanced_partial", [2000] * 5 + [500] * 7, [2000] * 5 + [500] * 7),
    SyntheticCase("strong_left_forefoot", [7000] + [0] * 11, [3000] + [0] * 11),
    SyntheticCase("right_heel", [300] * 12, [100] * 9 + [3000] * 3),
    SyntheticCase("balanced_different_regions", [0] * 9 + [5000] * 3,
                  [0] * 5 + [3750] * 4 + [0] * 3),
    SyntheticCase("mild_left_broad", [1100] * 12, [900] * 12),
)
FIELDS = ("leftRegionalObservation", "rightRegionalObservation",
          "overallBalanceObservation", "practicalTip")


def model_summary(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [record for record in records if record["model"] == model]
    durations = [record["seconds"] for record in selected]
    return {
        "model": model,
        "total": len(selected),
        "first_response_passed": sum(record["passed"] for record in selected),
        "http_attempts": sum(record["http_attempts"] for record in selected),
        "median_seconds": round(statistics.median(durations), 3) if durations else None,
        "max_seconds": max(durations) if durations else None,
    }


async def evaluate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reused_records = None
    if args.reuse_heatmaps_from:
        reused_records = json.loads(
            (args.reuse_heatmaps_from / "results.json").read_text(encoding="utf-8")
        )["records"]
    prepared = []
    for case in CASES:
        # Hardware sends right C0 at the heel; canonical examples start at toes.
        right_raw = list(reversed(case.right_canonical))
        facts = reports.build_pressure_balance_facts(case.left, right_raw)
        if facts["qualityStatus"] != "valid":
            raise ValueError(f"Invalid synthetic fixture: {case.name}")
        grounding = dict(zip(FIELDS, reports._pressure_fallback_components(facts)))
        reports._require_pressure_advice(grounding, facts)
        if reused_records is None:
            heatmaps = create_pressure_heatmaps(case.left, right_raw, output_dir / case.name)
        else:
            source = next((row for row in reused_records if row["case"] == case.name), None)
            if source is None or (source["left"], source["right_raw"], source["facts"]) != (
                case.left, right_raw, facts,
            ):
                raise ValueError(f"Reused heatmap inputs do not match: {case.name}")
            directory = args.reuse_heatmaps_from.resolve() / case.name
            left_path, right_path = directory / "left_pressure_heatmap.png", directory / "right_pressure_heatmap.png"
            heatmaps = SavedPressureHeatmaps(
                left_path=left_path, right_path=right_path,
                left_png=left_path.read_bytes(), right_png=right_path.read_bytes(),
                combined_path=directory / "both_pressure_heatmap.png",
            )
        prepared.append((case, right_raw, facts, heatmaps))
    print(json.dumps({"status": "prepared", "cases": len(prepared),
                      "api_calls": 0, "output_dir": str(output_dir)}), flush=True)
    if args.prepare_only:
        return 0
    if not reports._openai_enabled():
        print("Live evaluation not run: configure OPENAI_API_KEY and enable report text.",
              file=sys.stderr, flush=True)
        return 2
    if not settings.openai_report_include_images:
        print("Live evaluation not run: OPENAI_REPORT_INCLUDE_IMAGES must be enabled.",
              file=sys.stderr, flush=True)
        return 2

    models = args.models or [settings.openai_report_model]
    # These overrides apply only to this CLI process, never to .env or the server.
    settings.openai_report_max_attempts = 1
    settings.openai_report_fallback_enabled = False
    settings.openai_report_timeout_seconds = args.timeout_seconds
    logging.getLogger("httpx").setLevel(logging.WARNING)
    records: list[dict[str, Any]] = []
    active: dict[str, Any] = {}
    original_create = reports._create_structured_response
    result_path = output_dir / "results.json"

    def save_results() -> None:
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_inputs_only": True,
            "max_attempts": 1,
            "fallback_enabled": False,
            "planned_requests_per_model": len(CASES) * args.repeats,
            "summaries": [model_summary(records, model) for model in models],
            "records": records,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    async def capture(**kwargs: Any) -> dict[str, Any]:
        active["budget"] = kwargs["request_budget"]
        data = await original_create(**kwargs)
        active["raw_output"] = data
        return data

    reports._create_structured_response = capture
    try:
        for model in models:
            settings.openai_report_model = model
            for case, right_raw, facts, heatmaps in prepared:
                for repeat in range(1, args.repeats + 1):
                    active.clear()
                    record: dict[str, Any] = {
                        "model": model, "case": case.name, "repeat": repeat,
                        "left": case.left, "right_raw": right_raw, "facts": facts,
                    }
                    started = time.perf_counter()
                    try:
                        report = await reports.generate_pressure_balance_report(
                            left_pressure_values=case.left, right_pressure_values=right_raw,
                            pressure_facts=facts, left_heatmap_png=heatmaps.left_png,
                            right_heatmap_png=heatmaps.right_png,
                        )
                        record.update(passed=report.generation_source == "OPENAI", report=asdict(report))
                    except Exception as exc:
                        # Do not serialize exception messages, credentials, or API headers.
                        record.update(passed=False, error_code=getattr(exc, "code", type(exc).__name__),
                                      field=getattr(exc, "field", None), rule_code=getattr(exc, "rule_code", None),
                                      status_code=getattr(exc, "status_code", None))
                    record["seconds"] = round(time.perf_counter() - started, 3)
                    record["http_attempts"] = getattr(active.get("budget"), "attempts", 0)
                    record["raw_output"] = active.get("raw_output")
                    if record["http_attempts"] != 1:
                        record.update(passed=False, error_code="SINGLE_HTTP_ATTEMPT_REQUIRED")
                    records.append(record)
                    save_results()
                    print(json.dumps({key: record.get(key) for key in (
                        "model", "case", "repeat", "passed", "seconds", "http_attempts",
                        "error_code", "field", "rule_code", "status_code",
                    )}), flush=True)
                    if record.get("error_code") in {
                        "API_HTTP_ERROR", "API_TIMEOUT", "API_NETWORK_ERROR", "API_CONNECTION_ERROR",
                    }:
                        return 2
            print(json.dumps(model_summary(records, model)), flush=True)
    finally:
        reports._create_structured_response = original_create
    print(f"Saved evaluation: {result_path}", flush=True)
    return 0 if all(record["passed"] for record in records) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Defaults to the currently configured model.")
    parser.add_argument("--repeats", type=int, default=2, choices=range(1, 11))
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--prepare-only", action="store_true", help="Validate synthetic inputs without API calls.")
    parser.add_argument("--reuse-heatmaps-from", type=Path,
                        help="Reuse a completed evaluation's images after checking all sensor inputs and facts.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ".build" / "pressure_first_pass" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    if not 5 <= args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 5 and 600")
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
