from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.shoe.canonical_v2.adapters import (  # noqa: E402
    AbcMartArtifactAdapter, MusinsaArtifactAdapter, TwentyNineCmArtifactAdapter,
)
from app.services.shoe.canonical_v2.pipeline import load_runrepeat_pages, run_dry_run, write_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Canonical Shoe V2 phase 2-A artifact matching")
    parser.add_argument("--crawler-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=75, choices=range(50, 101), metavar="50..100")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "canonical-shoe-v2")
    args = parser.parse_args()
    rr = args.crawler_root / "selection" / "20260826-03" / "runrepeat-pages.json"
    musinsa = args.crawler_root / "selection" / "20260826-03" / "candidates.json"
    inventory = args.crawler_root / "artifacts" / "catalog_audits" / "abc-musinsa-29cm-model-family-inventory-2026-10-09.csv"
    sku_evidence = args.crawler_root / "artifacts" / "catalog_audits" / "abc-musinsa-29cm-overlap-2026-10-09.csv"
    for path in (rr, musinsa, inventory, sku_evidence):
        if not path.is_file():
            parser.error(f"required evidence artifact not found: {path}")
    payload = run_dry_run(load_runrepeat_pages(rr), [
        MusinsaArtifactAdapter(musinsa), AbcMartArtifactAdapter(inventory, sku_evidence), TwentyNineCmArtifactAdapter(inventory, sku_evidence),
    ], args.sample_size)
    json_path = args.output_dir / "phase-2a-dry-run.json"
    csv_path = args.output_dir / "phase-2a-dry-run-audit.csv"
    write_artifacts(payload, json_path, csv_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), **payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
