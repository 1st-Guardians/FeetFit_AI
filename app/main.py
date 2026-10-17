import logging
from time import perf_counter

from fastapi import FastAPI, Request

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


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "좌우 발 사진을 업로드하면 AI가 무좀 의심 영역과 염증/각질 관련 징후를 분석하고, "
        "의심 영역 지도와 원본 오버레이 이미지를 생성해 리포트 서버로 전송합니다."
    ),
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
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.exception(
            "[HTTP 실패] 요청 처리 중 오류가 발생했습니다. "
            "| method=%s path=%s elapsedMs=%.1f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "[HTTP 완료] 요청 처리를 마쳤습니다. "
        "| method=%s path=%s statusCode=%s elapsedMs=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
async def log_ai_runtime_preflight() -> None:
    global _runtime_preflight_logged
    if _runtime_preflight_logged:
        return
    status = embedding_runtime_status()
    ollama = await ollama_runtime_status()
    logger.info(
        "[AI 서버] 실행 준비를 완료했습니다. "
        "| cudaAvailable=%s gpuName=%s pytorchCudaVersion=%s "
        "bgeM3Device=%s ollamaReachable=%s ollamaModel=%s "
        "ollamaGpuInUse=%s vramBytes=%s",
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
            "[AI 서버] Ollama startup prewarm을 시작합니다. "
            "BGE-M3 배치와 동시에 실행하지 마세요."
        )
        await prewarm_ollama_model()
    _runtime_preflight_logged = True


@app.get("/health", tags=["상태 확인"], summary="서버 상태 확인")
def health_check():
    return {"status": "ok"}
