from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.shoe.canonical_v2.evaluation import (  # noqa: E402
    GroundTruthValidationError, read_and_validate_labels, write_evaluation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Canonical Shoe V2 labels without changing matching policy")
    parser.add_argument("input", type=Path, nargs="?", default=ROOT / "artifacts/canonical-shoe-v2/phase-2a-ground-truth-template.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts/canonical-shoe-v2/phase-2a-ground-truth-evaluation.json")
    parser.add_argument("--errors-csv", type=Path, default=ROOT / "artifacts/canonical-shoe-v2/phase-2a-ground-truth-errors.csv")
    args = parser.parse_args()
    try:
        rows = read_and_validate_labels(args.input)
    except (OSError, GroundTruthValidationError) as exc:
        parser.exit(2, f"label validation failed: {exc}\n")
    evaluation = write_evaluation_artifacts(rows, args.output_json, args.errors_csv)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
