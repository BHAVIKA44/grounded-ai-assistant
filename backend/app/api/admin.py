from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
import httpx

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ValidationAppError
from app.core.logging import get_logger
from app.db.session import get_session
from app.fine_tuning.trainer import get_fine_tuning_service
from app.services.cache_service import get_cache_service
from app.services.llm_service import LLMProvider, get_llm_service
from app.services.observability import get_observability_service

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
_fine_tune_state: Dict[str, Any] = {"status": "idle", "dataset_path": None, "output_path": None, "error": None}
SUPPORTED_OLLAMA_MODELS = ["llama3.2", "phi3:mini", "qwen2.5:3b"]
SUPPORTED_GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


@router.get("/status")
async def status() -> Dict[str, Any]:
    settings = get_settings()
    redis_ok = False
    db_ok = False
    try:
        cache = await get_cache_service()
        redis_ok = await cache.ping()
    except Exception:
        redis_ok = False
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False
    obs = await get_observability_service()
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "services": {"db": db_ok, "redis": redis_ok, "langsmith": obs.is_enabled()},
        "llm": {
            "provider": "ollama",
            "model": settings.ollama_model,
            "openai_model": settings.openai_model,
            "groq_model": settings.groq_model,
        },
    }


@router.get("/cache")
async def cache_status() -> Dict[str, Any]:
    cache = await get_cache_service()
    return {"available": await cache.ping()}


@router.delete("/cache")
async def clear_cache(pattern: str = Query(default="*")) -> Dict[str, Any]:
    cache = await get_cache_service()
    deleted = await cache.clear_pattern(pattern)
    return {"deleted": deleted, "pattern": pattern}


@router.get("/llm")
async def llm_status() -> Dict[str, Any]:
    svc = await get_llm_service()
    return {
        "provider": svc.provider,
        "model": svc.model,
        "supported_providers": [p.value for p in LLMProvider],
        "supported_models": SUPPORTED_OLLAMA_MODELS,
        "supported_models_by_provider": {
            "ollama": SUPPORTED_OLLAMA_MODELS,
            "groq": SUPPORTED_GROQ_MODELS,
            "openai": [],
        },
    }


@router.post("/llm")
async def set_llm(provider: str = "ollama", model: str = "") -> Dict[str, Any]:
    svc = await get_llm_service()
    try:
        svc.provider = LLMProvider(provider)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": str(exc)},
        )
    if model:
        if svc.provider == LLMProvider.OLLAMA and model not in SUPPORTED_OLLAMA_MODELS:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "validation_error",
                    "message": f"Unsupported model '{model}'. Use one of: {', '.join(SUPPORTED_OLLAMA_MODELS)}",
                },
            )
        if svc.provider == LLMProvider.GROQ and model not in SUPPORTED_GROQ_MODELS:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "validation_error",
                    "message": f"Unsupported model '{model}'. Use one of: {', '.join(SUPPORTED_GROQ_MODELS)}",
                },
            )
        svc.model = model
    return {
        "provider": svc.provider,
        "model": svc.model,
        "supported_models": SUPPORTED_OLLAMA_MODELS,
    }


@router.post("/llm/pull")
async def pull_llm_model(model: str) -> Dict[str, Any]:
    if model not in SUPPORTED_OLLAMA_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "validation_error",
                "message": f"Unsupported model '{model}'. Use one of: {', '.join(SUPPORTED_OLLAMA_MODELS)}",
            },
        )
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/pull",
                json={"name": model, "stream": False},
            )
        if response.status_code != 200:
            raise ExternalServiceError(f"Ollama pull failed: {response.text}")
        return {"pulled": True, "model": model}
    except ExternalServiceError as exc:
        logger.error("admin_llm_pull_failed", model=model, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        )
    except httpx.HTTPError as exc:
        logger.error("admin_llm_pull_failed", model=model, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"code": "external_service_error", "message": str(exc)},
        )


@router.post("/fine-tuning/prepare")
async def prepare_fine_tuning(data: List[Dict[str, str]]) -> Dict[str, Any]:
    svc = await get_fine_tuning_service()
    path = svc.prepare_dataset(data=data, output_path="/app/data/fine_tuning/train.jsonl")
    _fine_tune_state["dataset_path"] = path
    _fine_tune_state["status"] = "prepared"
    return {"prepared": True, "dataset_path": path, "samples": len(data)}


@router.post("/fine-tuning/run")
async def run_fine_tuning(dataset_path: str | None = None) -> Dict[str, Any]:
    svc = await get_fine_tuning_service()
    active_dataset = dataset_path or _fine_tune_state.get("dataset_path")
    if not active_dataset:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": "No dataset prepared"},
        )
    _fine_tune_state["status"] = "running"
    _fine_tune_state["error"] = None
    try:
        output = await svc.fine_tune(active_dataset)
        _fine_tune_state["status"] = "completed"
        _fine_tune_state["output_path"] = output
        return {"status": "completed", "dataset_path": active_dataset, "output_path": output}
    except Exception as exc:
        _fine_tune_state["status"] = "failed"
        _fine_tune_state["error"] = str(exc)
        svc_logger = getattr(svc, "logger", None)
        if svc_logger is not None:
            svc_logger.exception("fine_tuning_run_failed", dataset_path=active_dataset)
        raise HTTPException(
            status_code=500,
            detail={"code": "fine_tuning_failed", "message": str(exc)},
        )


@router.get("/fine-tuning/status")
async def fine_tuning_status() -> Dict[str, Any]:
    return _fine_tune_state
