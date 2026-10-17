from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import ceil
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.shoe.shoe_server_client import ShoeServerClient


SHOE_COUNT = 338
REVIEWS_PER_SHOE = 140


def _foot_state() -> dict:
    return {
        "dailyFootAnalysis": {
            "balanceScore": 82.0,
            "leftPressurePercent": 49.0,
            "rightPressurePercent": 51.0,
            "measuredLeftFootSizeMm": 268.2,
            "measuredRightFootSizeMm": 269.1,
            "leftFootWidthMm": 102.4,
            "rightFootWidthMm": 103.1,
            "avgTemperatureCelsius": 31.2,
            "avgHumidityPercent": 54.0,
            "typeText": "균형형",
        },
        "tinaPedisAnalysis": {
            "fungalSuspicionSafetyScore": 91,
            "skinReactionSafetyScore": 88,
        },
        "halluxValgusAnalysis": {
            "leftToeAngleDegree": 11.2,
            "rightToeAngleDegree": 12.4,
            "riskScore": 30.0,
        },
        "staticPressureAnalyses": [
            {
                "analysisId": 501,
                "footSide": "LEFT",
                "leftPressureRatio": 49.0,
                "rightPressureRatio": 51.0,
                "forefootPressureRatio": 57.0,
                "rearfootPressureRatio": 43.0,
                "centerOfPressureX": 0.48,
                "centerOfPressureY": 0.61,
                "balanceScore": 82.0,
                "balanceStatus": "VERY_GOOD",
                "analysisText": None,
            }
        ],
        "pressureSensorReadings": [],
    }


def _metric(shoe_id: int, offset: int, characteristic: str) -> dict:
    return {
        "metricId": shoe_id * 100 + offset,
        "canonicalCharacteristic": characteristic,
        "sourceMetricName": characteristic,
        "value": "21.30",
        "averageValue": "19.80",
        "sourceMinValue": "10.00",
        "sourceMaxValue": "30.00",
        "unit": "benchmark-unit",
        "testedSize": "US 9",
        "methodName": "benchmark",
        "methodVersion": "new method",
        "location": "HEEL" if characteristic in {"SHOCK_ABSORPTION", "ENERGY_RETURN"} else None,
        "variant": "width" if characteristic == "TOEBOX_SPACE" else None,
        "comparisonSampleCount": 338,
        "comparisonCohort": "benchmark cohort",
        "rawValueText": "21.30 benchmark-unit",
    }


def _shoe(shoe_id: int) -> dict:
    review_text = (
        "발볼과 앞코 공간이 편하고 오래 걸어도 쿠션감이 안정적이며 "
        "통풍도 무난하다는 실제 착화 리뷰입니다."
    )
    reviews = [
        {
            "reviewId": shoe_id * 1_000 + review_index,
            "rating": 5.0,
            "reviewText": f"{review_text} 상품 {shoe_id} 리뷰 {review_index}.",
            "source": "MUSINSA",
            "collectedAt": "2026-10-18T00:00:00",
        }
        for review_index in range(1, REVIEWS_PER_SHOE + 1)
    ]
    characteristics = (
        "WIDTH_SPACE",
        "TOEBOX_SPACE",
        "SHOCK_ABSORPTION",
        "ENERGY_RETURN",
        "CUSHION",
        "BREATHABILITY",
    )
    return {
        "shoeId": shoe_id,
        "brandName": "Benchmark Brand",
        "shoeName": f"Benchmark Shoe {shoe_id}",
        "modelCode": f"BENCH-{shoe_id}",
        "musinsaUrl": f"https://example.test/products/{shoe_id}",
        "price": 129000,
        "imageUrl": None,
        "overallRating": 4.5,
        "reviewCount": REVIEWS_PER_SHOE,
        "reviews": reviews,
        "labMeasurements": [
            {
                "measurementId": shoe_id * 10,
                "source": "RUNREPEAT",
                "sourceUrl": f"https://example.test/runrepeat/{shoe_id}",
                "sourceBrandName": "Benchmark Brand",
                "sourceShoeName": f"Benchmark Shoe {shoe_id}",
                "sourceModelCode": f"BENCH-{shoe_id}",
                "testedSize": "US 9",
                "capturedAt": "2026-10-18T00:00:00",
                "parserVersion": "benchmark-v1",
                "internalLengthMm": None,
                "widthMm": 98.0,
                "toeboxWidthMm": None,
                "toeboxHeightMm": None,
                "insoleThicknessMm": None,
                "heelStackMm": None,
                "forefootStackMm": None,
                "rawMetrics": [
                    _metric(shoe_id, index, characteristic)
                    for index, characteristic in enumerate(characteristics, start=1)
                ],
            }
        ],
    }


def _page_bodies(page_size: int) -> tuple[dict[int, bytes], dict[int, bytes]]:
    shoes = [_shoe(shoe_id) for shoe_id in range(1, SHOE_COUNT + 1)]
    total_pages = ceil(SHOE_COUNT / page_size)
    raw_pages: dict[int, bytes] = {}
    gzip_pages: dict[int, bytes] = {}
    for page in range(total_pages):
        start = page * page_size
        result = {
            "measurementSessionId": 1,
            "userId": 1,
            "measurementStatus": "COMPLETED",
            "footState": _foot_state(),
            "shoes": shoes[start : start + page_size],
            "currentPage": page,
            "totalPages": total_pages,
            "totalElements": SHOE_COUNT,
            "hasNext": page + 1 < total_pages,
        }
        envelope = {
            "isSuccess": True,
            "code": "COMMON200",
            "message": "success",
            "result": result,
        }
        raw = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        raw_pages[page] = raw
        gzip_pages[page] = gzip.compress(raw, compresslevel=6)
    return raw_pages, gzip_pages


class _BenchmarkServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, page_size: int, use_gzip: bool):
        self.raw_pages, self.gzip_pages = _page_bodies(page_size)
        self.use_gzip = use_gzip
        self.request_count = 0
        self.transferred_bytes = 0
        self.connections: set[tuple[str, int]] = set()
        self.metrics_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _BenchmarkHandler)


class _BenchmarkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        page = int(parse_qs(urlsplit(self.path).query).get("page", ["0"])[0])
        server = self.server
        assert isinstance(server, _BenchmarkServer)
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        compressed = server.use_gzip and accepts_gzip
        body = server.gzip_pages[page] if compressed else server.raw_pages[page]
        with server.metrics_lock:
            server.request_count += 1
            server.transferred_bytes += len(body)
            server.connections.add(self.client_address)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


class _FreshConnectionClient:
    """Reproduce the pre-optimization client lifecycle for baseline timing."""

    async def request(self, method: str, url: str, **kwargs):
        timeout = kwargs.pop("timeout")
        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
        ) as client:
            return await client.request(method, url, **kwargs)


async def _run_once(
    endpoint: str,
    page_size: int,
    *,
    reuse_connections: bool,
) -> float:
    old_endpoint = settings.shoe_recommendation_context_endpoint
    old_page_size = settings.shoe_recommendation_context_page_size
    settings.shoe_recommendation_context_endpoint = endpoint
    settings.shoe_recommendation_context_page_size = page_size
    try:
        client = ShoeServerClient(
            "Bearer benchmark-token",
            internal_api_key="benchmark-internal-key",
            http_client=None if reuse_connections else _FreshConnectionClient(),
        )
        started_at = time.perf_counter()
        context = await client.fetch_recommendation_context(1)
        elapsed_ms = (time.perf_counter() - started_at) * 1_000
        if len(context.shoes) != SHOE_COUNT:
            raise RuntimeError(f"Expected {SHOE_COUNT} shoes, received {len(context.shoes)}")
        return elapsed_ms
    finally:
        settings.shoe_recommendation_context_endpoint = old_endpoint
        settings.shoe_recommendation_context_page_size = old_page_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, choices=(100, 200), required=True)
    parser.add_argument("--gzip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reuse-connections",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    server = _BenchmarkServer(page_size=args.page_size, use_gzip=args.gzip)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/recommendation-context"
    durations = []
    try:
        for _ in range(args.runs):
            durations.append(asyncio.run(_run_once(
                endpoint,
                args.page_size,
                reuse_connections=args.reuse_connections,
            )))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    total_raw_bytes = sum(len(body) for body in server.raw_pages.values()) * args.runs
    result = {
        "shoeCount": SHOE_COUNT,
        "reviewsPerShoe": REVIEWS_PER_SHOE,
        "pageSize": args.page_size,
        "gzip": args.gzip,
        "reuseConnections": args.reuse_connections,
        "runs": args.runs,
        "medianMs": round(statistics.median(durations), 2),
        "minMs": round(min(durations), 2),
        "maxMs": round(max(durations), 2),
        "requestCount": server.request_count,
        "uniqueTcpConnections": len(server.connections),
        "rawResponseBytes": total_raw_bytes,
        "transferredBytes": server.transferred_bytes,
        "transferRatio": round(server.transferred_bytes / total_raw_bytes, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
