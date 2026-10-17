from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    load_dotenv(PROJECT_ROOT / ".env")
    port = int(os.getenv("NGROK_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("NGROK_PORT must be between 1 and 65535.")
    domain = os.getenv("NGROK_DOMAIN", "").strip().rstrip("/")
    if domain:
        os.environ["PUBLIC_BASE_URL"] = domain if "://" in domain else f"https://{domain}"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    print(f"Feetfit local API: http://127.0.0.1:{port}", flush=True)
    print(f"Swagger: http://127.0.0.1:{port}/docs", flush=True)
    print("Press Ctrl+C to stop only the local API. Run this command again to restart.", flush=True)
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1, reload=False)


if __name__ == "__main__":
    main()
