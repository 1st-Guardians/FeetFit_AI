from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.shoe.canonical_v2.evaluation import build_ground_truth_rows, write_ground_truth_template  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human-label template from the immutable phase 2-A dry-run")
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/canonical-shoe-v2/phase-2a-dry-run.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/canonical-shoe-v2/phase-2a-ground-truth-template.csv")
    parser.add_argument("--rejected-count", type=int, default=15)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_ground_truth_rows(payload, args.rejected_count)
    write_ground_truth_template(rows, args.output)
    counts = Counter(row["predictedStatus"] for row in rows)
    print(json.dumps({"output": str(args.output), "totalSamples": len(rows), "predictedStatusCounts": dict(counts)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
