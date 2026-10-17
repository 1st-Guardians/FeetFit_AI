import logging
from time import perf_counter

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

from app.api.router import api_router
from app.core.config import settings
from app.core.logging_config import configure_console_logging
from app.services.shoe.shoe_embedding import embedding_runtime_status
from app.services.shoe.shoe_fit_comment_service import (
    ollama_runtime_status,
    prewarm_ollama_model,
)


configure_console_logging()
logger = logging.getLogger(__name__)
_runtime_preflight_logged = False


public_base_url = (
    str(settings.public_base_url).rstrip("/")
    if settings.public_base_url is not None
    else None
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "하드웨어가 촬영한 발등·발바닥 사진과 좌우 압력 센서값, 측정 전후 온·습도 값을 "
        "분석하고 필요한 리포트와 지표를 백엔드 서버에 저장합니다."
    ),
    servers=(
        [{"url": public_base_url, "description": "하드웨어 공개 AI 서버"}]
        if public_base_url
        else None
    ),
    openapi_tags=[
        {
            "name": "Feetfit Analysis",
            "description": "하드웨어가 호출하는 사진·압력·환경 통합 분석 API",
        }
    ],
)

app.include_router(api_router, prefix="/api")


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    started_at = perf_counter()
    logger.info(
        "[HTTP 요청] AI 서버가 요청을 받았습니다. | method=%s path=%s",
        request.method,
        request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "[HTTP 실패] 요청 처리 중 오류가 발생했습니다. | method=%s path=%s elapsedMs=%.1f",
            request.method,
            request.url.path,
            (perf_counter() - started_at) * 1000,
        )
        raise
    logger.info(
        "[HTTP 완료] 요청 처리를 마쳤습니다. | method=%s path=%s statusCode=%s elapsedMs=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started_at) * 1000,
    )
    return response


@app.on_event("startup")
async def load_local_wound_model() -> None:
    if settings.tinea_inflammation_enabled and settings.wound_segmentation_enabled:
        from app.services.wound_analysis import get_wound_analyzer
        analyzer = await run_in_threadpool(get_wound_analyzer)
        logger.info("Local wound model ready: device=%s checkpoint=%s", analyzer.device, analyzer.checkpoint_path)


@app.on_event("startup")
async def log_ai_runtime_preflight() -> None:
    global _runtime_preflight_logged
    if _runtime_preflight_logged:
        return
    status = embedding_runtime_status()
    ollama = await ollama_runtime_status()
    logger.info(
        "Phase D runtime: CUDA available=%s GPU name=%s PyTorch CUDA version=%s "
        "BGE-M3 device=%s Ollama reachable=%s model=%s GPU in use=%s VRAM bytes=%s",
        status["cudaAvailable"],
        status["gpuName"],
        status["pytorchCudaVersion"],
        status["bgeM3ResolvedDevice"],
        ollama["reachable"],
        ollama["model"],
        ollama["gpuInUse"],
        ollama.get("sizeVram"),
    )
    if settings.ollama_prewarm_on_startup:
        logger.info(
            "[AI 서버] Ollama startup prewarm을 시작합니다. BGE-M3 배치와 동시에 실행하지 마세요."
        )
        await prewarm_ollama_model()
    _runtime_preflight_logged = True


@app.get("/health", tags=["상태 확인"], summary="서버 상태 확인")
def health_check():
    return {"status": "ok"}
